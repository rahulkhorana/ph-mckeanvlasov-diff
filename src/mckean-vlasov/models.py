from typing import Any, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as nn_attn


class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        h = nn.LayerNorm()(h)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, (1, 1))(x)
        return x + h


# ---------- Embeddings ----------


def time_embed(t: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """Sin-cos positional embedding. Pure (no nn.Dense), so no init/apply hassles.
    t: (B,) float in [0,1] or int timesteps; returns (B, dim)
    """
    t = t.astype(jnp.float32)
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(np.log(1.0), np.log(10000.0), half))
    angles = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if dim % 2:  # odd dim, pad
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def _film(g: jnp.ndarray, c: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Produce FiLM (scale, shift) from a conditioning vector g."""
    h = nn.gelu(nn.Dense(2 * c)(g))
    scale, shift = jnp.split(h, 2, axis=-1)
    return scale, shift  # (B,c), (B,c)


def _group_count(C: int) -> int:
    # pick num_groups that divides C; prefer 8 then 4 then 2 then 1
    for g in (8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1


# ---------- 2.5D UNet blocks (no downsample along K) ----------


def downsample_2d_keepK(x):  # (B,H,W,K,C)
    H, W = x.shape[1], x.shape[2]
    return jax.image.resize(
        x, (x.shape[0], H // 2, W // 2, x.shape[3], x.shape[4]), method="linear"
    )


def upsample_2d_keepK(x):  # (B,H,W,K,C)
    H, W = x.shape[1], x.shape[2]
    return jax.image.resize(
        x, (x.shape[0], H * 2, W * 2, x.shape[3], x.shape[4]), method="linear"
    )


class AttnBottleneck(nn.Module):
    heads: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (B,H,W,K,C) -> (B, HWK, C)
        B, H, W, K, C = x.shape
        y = x.reshape(B, H * W * K, C)
        attn = nn.SelfAttention(num_heads=self.heads, qkv_features=C)
        y = attn(nn.LayerNorm()(y))
        y = y + nn.Dense(C)(nn.gelu(nn.Dense(4 * C)(nn.LayerNorm()(y))))
        return y.reshape(B, H, W, K, C)


# ---------- 3D ResBlock keeping shapes with SAME padding ----------
class ResBlock3D(nn.Module):
    features: int
    kernel: tuple = (3, 3, 1)  # don’t mix across K aggressively; keep K small & stable

    @nn.compact
    def __call__(self, x):
        # x: (B,H,W,K,C)
        h = nn.GroupNorm(num_groups=1)(x)  # channels-last -> use few groups
        h = nn.swish(h)
        h = nn.Conv(self.features, self.kernel, padding="SAME")(h)
        h = nn.GroupNorm(num_groups=1)(h)
        h = nn.swish(h)
        h = nn.Conv(self.features, self.kernel, padding="SAME")(h)
        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, (1, 1, 1), padding="SAME")(x)
        return x + h


# ---------- Tiny 3D UNet w/ HW-only down/up, K fixed ----------
class TinyUNet3D(nn.Module):
    ch: int = 64
    m_dim: int = 256  # modules embedding dim

    def _inject(self, h, te, m_emb):
        # h: (B,H,W,K,C) ; te: (B,D) ; m_emb: (B,M)
        b, H, W, K, C = h.shape
        cond = jnp.concatenate([te, m_emb], axis=-1)  # (B, D+M)
        cond = nn.Dense(C)(cond)  # (B, C)
        cond = cond[:, None, None, None, :]  # (B,1,1,1,C)
        return h + cond

    @nn.compact
    def __call__(self, x, t_embed, m_emb):
        # x: (B, H, W, K, C) with C=channels=3 here; K=3
        # -------- Down --------
        h1 = ResBlock3D(self.ch)(x)
        h1 = self._inject(h1, t_embed, m_emb)
        d1 = nn.max_pool(h1, window_shape=(2, 2, 1), strides=(2, 2, 1), padding="SAME")

        h2 = ResBlock3D(self.ch * 2)(d1)
        h2 = self._inject(h2, t_embed, m_emb)
        d2 = nn.max_pool(h2, window_shape=(2, 2, 1), strides=(2, 2, 1), padding="SAME")

        h3 = ResBlock3D(self.ch * 4)(d2)
        h3 = self._inject(h3, t_embed, m_emb)

        # -------- Up --------
        u2 = nn.ConvTranspose(
            self.ch * 2, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding="SAME"
        )(h3)
        # shapes now match spatially with h2
        u2 = jnp.concatenate([u2, h2], axis=-1)
        u2 = ResBlock3D(self.ch * 2)(u2)
        u2 = self._inject(u2, t_embed, m_emb)

        u1 = nn.ConvTranspose(
            self.ch, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding="SAME"
        )(u2)
        # shapes now match spatially with h1
        u1 = jnp.concatenate([u1, h1], axis=-1)
        u1 = ResBlock3D(self.ch)(u1)
        u1 = self._inject(u1, t_embed, m_emb)

        # predict noise/v same shape as input
        eps = nn.Conv(x.shape[-1], kernel_size=(1, 1, 1), padding="SAME")(u1)
        return eps  # (B,H,W,K,C)


TinyUNet = TinyUNet3D


############# MODULES
class MHA(nn.Module):
    d: int
    h: int = 4

    @nn.compact
    def __call__(self, qx, kx=None, vx=None, q_mask=None, k_mask=None):
        if kx is None:
            kx = qx
        if vx is None:
            vx = kx
        q = nn.Dense(self.d)(qx)
        k = nn.Dense(self.d)(kx)
        v = nn.Dense(self.d)(vx)
        attn_mask = None
        if (q_mask is not None) and (k_mask is not None):
            attn_mask = nn_attn.make_attention_mask(q_mask, k_mask)  # (B,1,Q,K)
        return nn.MultiHeadDotProductAttention(num_heads=self.h)(
            q, k, v, mask=attn_mask
        )


class SAB(nn.Module):
    d: int
    h: int = 4

    @nn.compact
    def __call__(self, x, key_mask=None):
        B, S, _ = x.shape
        k_mask_bool = None
        if key_mask is not None:
            k_mask_bool = (
                (key_mask[..., 0] > 0) if key_mask.ndim == 3 else key_mask.astype(bool)
            )
        q_mask_bool = jnp.ones((B, S), dtype=bool)
        y = MHA(self.d, self.h)(x, None, None, q_mask=q_mask_bool, k_mask=k_mask_bool)
        x = x + y
        x = x + nn.Dense(self.d)(nn.gelu(nn.Dense(self.d * 4)(x)))
        return x


class PMA(nn.Module):
    d: int
    m: int = 1
    h: int = 4

    @nn.compact
    def __call__(self, x, key_mask=None):
        B, S, d = x.shape
        seeds = self.param("seed", nn.initializers.normal(0.02), (self.m, self.d))
        q = jnp.repeat(seeds[None, :, :], B, axis=0)  # (B,m,d)
        k_mask_bool = None
        if key_mask is not None:
            k_mask_bool = (
                (key_mask[..., 0] > 0) if key_mask.ndim == 3 else key_mask.astype(bool)
            )
        q_mask_bool = jnp.ones((B, self.m), dtype=bool)
        return MHA(self.d, self.h)(q, x, x, q_mask=q_mask_bool, k_mask=k_mask_bool)


# ---------- featurizer for ragged modules -> (T,S,F) ----------
def _robust_stats(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).astype(np.float32)
    finite = np.isfinite(a)
    fr = float(finite.mean()) if a.size else 0.0
    a = np.where(finite, a, np.nan)
    if a.size == 0 or fr == 0.0:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, fr, 0, 0], dtype=np.float32)
    mean = np.nanmean(a)
    std = np.nanstd(a)
    q25, q50, q75 = np.nanpercentile(a, [25, 50, 75])
    mn, mx = np.nanmin(a), np.nanmax(a)
    l1 = np.nansum(np.abs(a))
    l2 = float(np.sqrt(np.nansum(a * a)))
    max_abs = float(np.nanmax(np.abs(a)))
    numel = float(a.size)
    log_numel = float(np.log1p(numel))
    centered = a - mean
    m3 = np.nanmean(centered**3)
    skew = float(m3 / (std**3 + 1e-6))
    return np.array(
        [mean, std, mn, mx, q25, q50, q75, l1, l2, log_numel, fr, max_abs, skew],
        dtype=np.float32,
    )


def featurize_modules_trajectory(
    mods_trajectory: List[List[Any]],  # length T; each step is a list of tensors
    T_max: int = 1,
    S_max: int = 16,
    add_pos_ids: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    traj = mods_trajectory[:T_max]
    T = len(traj)
    feat_rows, set_masks = [], []
    for t in range(T_max):
        if t < T:
            elems = traj[t][:S_max]
            row = []
            for i, e in enumerate(elems):
                f = _robust_stats(np.array(e))
                if add_pos_ids:
                    oh = np.zeros((S_max,), dtype=np.float32)
                    oh[min(i, S_max - 1)] = 1.0
                    f = np.concatenate([f, oh], axis=0)
                row.append(f)
            if len(row) == 0:
                F = 13 + (S_max if add_pos_ids else 0)
                row = [np.zeros((F,), dtype=np.float32)]
            F = row[0].shape[0]
            if len(row) < S_max:
                row += [np.zeros((F,), dtype=np.float32)] * (S_max - len(row))
            feat_rows.append(np.stack(row, axis=0))
            set_masks.append(
                np.array(
                    [1.0] * min(len(traj[t]), S_max)
                    + [0.0] * (S_max - min(len(traj[t]), S_max))
                )[:, None]
            )
        else:
            F = (
                feat_rows[0].shape[1]
                if feat_rows
                else (13 + (S_max if add_pos_ids else 0))
            )
            feat_rows.append(np.zeros((S_max, F), dtype=np.float32))
            set_masks.append(np.zeros((S_max, 1), dtype=np.float32))
    feats = jnp.array(np.stack(feat_rows, axis=0))  # (T_max,S_max,F)
    set_mask = jnp.array(np.stack(set_masks, axis=0))  # (T_max,S_max,1)
    time_mask = jnp.array(np.array([1.0] * T + [0.0] * (T_max - T), dtype=np.float32))[
        :, None
    ]  # (T_max,1)
    return feats, set_mask, time_mask


# ---------- your encoder, unchanged ----------
class ModulesTrajectoryEncoder(nn.Module):
    d_set: int = 128
    d_time: int = 256
    out_dim: int = 256
    n_sab: int = 2
    n_layers_time: int = 2
    heads: int = 4
    m_pma: int = 1
    dropout_rate: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(
        self, feats: jnp.ndarray, set_mask: jnp.ndarray, time_mask: jnp.ndarray
    ):
        B, T, S, F = feats.shape
        x = feats.reshape(B * T, S, F)
        m = set_mask.reshape(B * T, S, 1)
        x = nn.Dense(self.d_set)(x)
        for _ in range(self.n_sab):
            x = SAB(self.d_set, self.heads)(x, key_mask=m)
        x = PMA(self.d_set, m=self.m_pma, h=self.heads)(x, key_mask=m).squeeze(1)
        x = x.reshape(B, T, self.d_set)
        x = nn.Dense(self.d_time)(x)
        tmask_bool = (
            (time_mask.squeeze(-1) > 0)
            if time_mask.ndim == 3
            else time_mask.astype(bool)
        )
        attn_mask = nn_attn.make_attention_mask(tmask_bool, tmask_bool)
        for _ in range(self.n_layers_time):
            h = nn.LayerNorm()(x)
            y = nn.SelfAttention(num_heads=self.heads, qkv_features=self.d_time)(
                h, mask=attn_mask
            )
            if self.dropout_rate > 0:
                y = nn.Dropout(self.dropout_rate)(y, deterministic=self.deterministic)
            x = x + y
            h = nn.LayerNorm()(x)
            y = nn.Dense(self.d_time * 4)(h)
            y = nn.gelu(y)
            y = nn.Dense(self.d_time)(y)
            if self.dropout_rate > 0:
                y = nn.Dropout(self.dropout_rate)(y, deterministic=self.deterministic)
            x = x + y
        time_w = time_mask
        denom = jnp.clip(jnp.sum(time_w, axis=1), 1e-6, None)
        x_mean = jnp.sum(x * time_w, axis=1) / denom
        z = nn.Dense(self.out_dim)(nn.gelu(nn.Dense(self.out_dim)(x_mean)))
        return z  # (B,out_dim)


# ---------- tiny builder that returns an embed_fn you can pass batches of ragged modules ----------
def build_modules_embedder(rng, out_dim=256, T_max=1, S_max=16):
    """
    Returns: (encoder_module, encoder_params, embed_fn)
      embed_fn(mods_batch: List[List[np.ndarray|torch.Tensor|jnp.ndarray]]) -> (B,out_dim) jnp.ndarray
    For now T_max=1 because your modules are flat lists; wrap to T=1 trajectories.
    """
    enc = ModulesTrajectoryEncoder(out_dim=out_dim)
    F = 13 + S_max  # robust stats + 1-of-S_max positional id
    feats_d = jnp.zeros((1, T_max, S_max, F), jnp.float32)
    set_d = jnp.zeros((1, T_max, S_max, 1), jnp.float32)
    time_d = jnp.ones((1, T_max, 1), jnp.float32)
    enc_params = enc.init(rng, feats_d, set_d, time_d)["params"]

    def as_T1_traj(mods_batch):
        # your dataset has a flat list per sample -> wrap as length-1 trajectory
        return [[mods] for mods in mods_batch]

    def embed_fn(mods_batch):
        traj_batch = as_T1_traj(mods_batch)
        feats_list, set_list, time_list = [], [], []
        for traj in traj_batch:
            f, s, t = featurize_modules_trajectory(traj, T_max=T_max, S_max=S_max)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        feats_b = jnp.stack(feats_list, 0)  # (B,T,S,F)
        set_b = jnp.stack(set_list, 0)  # (B,T,S,1)
        time_b = jnp.stack(time_list, 0)  # (B,T,1)
        return enc.apply({"params": enc_params}, feats_b, set_b, time_b)  # (B,out_dim)

    return enc, enc_params, embed_fn


### ENEGY


class EnergyNetwork(nn.Module):
    """E_phi(L, M): takes image L (B,H,W,C) and M-embedding (B,d) -> scalar per sample."""

    ch: int = 64
    m_dim: int = 256

    @nn.compact
    def __call__(self, L, m_emb):
        # Expect NHWC (B,H,W,C)
        assert L.ndim >= 4, f"EnergyNetwork expects NHWC input, got {L.shape}"

        h = ResBlock(self.ch)(L)
        h = nn.max_pool(h, (2, 2), strides=(2, 2), padding="SAME")
        h = ResBlock(self.ch * 2)(h)
        h = nn.max_pool(h, (2, 2), strides=(2, 2), padding="SAME")
        h = ResBlock(self.ch * 4)(h)  # (B, H', W', C')

        # Global average over all spatial axes (everything except batch & channels)
        spatial_axes = tuple(range(1, h.ndim - 1))  # e.g. (1, 2) for 2D images
        h = jnp.mean(h, axis=spatial_axes)  # -> (B, C')

        # Ensure m_emb is (B, d)
        assert (
            m_emb.ndim == 2 and m_emb.shape[0] == h.shape[0]
        ), f"m_emb must be (B,d), got {m_emb.shape}"

        joint = jnp.concatenate([h, m_emb], axis=-1)  # (B, C'+d)
        e = nn.Dense(256)(joint)
        e = nn.tanh(e)
        e = nn.Dense(1)(e)
        return e.squeeze(-1)  # (B,)
