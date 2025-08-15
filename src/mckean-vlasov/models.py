from typing import Any, List, Tuple
import numpy as np
import jax.numpy as jnp
import jax
import flax.linen as nn
from flax.linen import attention as nn_attn


# ---------- sinusoidal time embedding (pure fn) ----------
def time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """
    t_cont: (B,) in [0,1]
    returns: (B, dim)
    """
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000.0), half))
    ang = t_cont[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


# ---------- Modules trajectory featurization (stats + sets) ----------
def _robust_stats(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32).ravel()
    if a.size == 0:
        return np.zeros(13, dtype=np.float32)
    finite = np.isfinite(a)
    fr = float(finite.mean())
    a = a[finite]
    if a.size == 0:
        return np.zeros(13, dtype=np.float32)
    mean = float(a.mean())
    std = float(a.std() + 1e-6)
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    mn, mx = float(a.min()), float(a.max())
    l1 = float(np.abs(a).sum())
    l2 = float(np.sqrt((a * a).sum()))
    max_abs = float(np.abs(a).max())
    logN = float(np.log1p(a.size))
    skew = float(np.mean(((a - mean) / std) ** 3))
    return np.array(
        [mean, std, mn, mx, q25, q50, q75, l1, l2, logN, fr, max_abs, skew], np.float32
    )


def featurize_modules_trajectory(
    mods_traj: List[List[Any]],
    T_max: int = 1,
    S_max: int = 16,
    add_pos_ids: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    mods_traj: length-T list; each item is a list of tensors at that time.
    Returns:
      feats     (T_max,S_max,F)
      set_mask  (T_max,S_max,1)
      time_mask (T_max,1)
    """
    traj = mods_traj[:T_max]
    T = len(traj)
    slices, masks = [], []
    for t in range(T_max):
        if t < T:
            elems = traj[t][:S_max]
            row = []
            for i, e in enumerate(elems):
                f = _robust_stats(np.array(e))
                if add_pos_ids:
                    oh = np.zeros((S_max,), np.float32)
                    oh[min(i, S_max - 1)] = 1.0
                    f = np.concatenate([f, oh], -1)
                row.append(f)
            if not row:
                F = 13 + (S_max if add_pos_ids else 0)
                row = [np.zeros((F,), np.float32)]
            F = row[0].shape[0]
            if len(row) < S_max:
                row += [np.zeros((F,), np.float32)] * (S_max - len(row))
            slices.append(np.stack(row, 0))
            valid = min(len(traj[t]), S_max)
            masks.append(
                np.array([1.0] * valid + [0.0] * (S_max - valid), np.float32)[:, None]
            )
        else:
            F = slices[0].shape[1] if slices else 13 + (S_max if add_pos_ids else 0)
            slices.append(np.zeros((S_max, F), np.float32))
            masks.append(np.zeros((S_max, 1), np.float32))
    feats = jnp.array(np.stack(slices, 0))
    set_mask = jnp.array(np.stack(masks, 0))
    time_mask = jnp.array(
        np.array([1.0] * T + [0.0] * (T_max - T), np.float32)[:, None]
    )
    return feats, set_mask, time_mask


# ---------- Set/Time encoders ----------
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
        mask = None
        if (q_mask is not None) and (k_mask is not None):
            mask = nn_attn.make_attention_mask(q_mask, k_mask)
        return nn.MultiHeadDotProductAttention(num_heads=self.h)(q, k, v, mask=mask)


class SAB(nn.Module):
    d: int
    h: int = 4

    @nn.compact
    def __call__(self, x, key_mask=None):
        B, S, _ = x.shape
        k_mask = (key_mask[..., 0] > 0) if (key_mask is not None) else None
        q_mask = jnp.ones((B, S), dtype=bool)
        y = MHA(self.d, self.h)(x, None, None, q_mask=q_mask, k_mask=k_mask)
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
        q = jnp.repeat(seeds[None, :, :], B, 0)  # (B,m,d)
        k_mask = (key_mask[..., 0] > 0) if (key_mask is not None) else None
        q_mask = jnp.ones((B, self.m), dtype=bool)
        return MHA(self.d, self.h)(q, x, x, q_mask=q_mask, k_mask=k_mask)


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
        tmask = time_mask.squeeze(-1) > 0
        attn_mask = nn_attn.make_attention_mask(tmask, tmask)
        for _ in range(self.n_layers_time):
            h = nn.LayerNorm()(x)
            y = nn.SelfAttention(num_heads=self.heads, qkv_features=self.d_time)(
                h, mask=attn_mask
            )
            x = x + y
            h = nn.LayerNorm()(x)
            y = nn.Dense(self.d_time)(nn.gelu(nn.Dense(self.d_time * 4)(h)))
            x = x + y
        denom = jnp.clip(jnp.sum(time_mask, axis=1), 1e-6, None)
        x_mean = jnp.sum(x * time_mask, axis=1) / denom
        z = nn.Dense(self.out_dim)(nn.gelu(nn.Dense(self.out_dim)(x_mean)))
        return z  # (B,out_dim)


def build_modules_embedder(rng, out_dim: int = 256, T_max: int = 1, S_max: int = 16):
    enc = ModulesTrajectoryEncoder(out_dim=out_dim)
    F = 13 + S_max
    feats = jnp.zeros((1, T_max, S_max, F), jnp.float32)
    set_m = jnp.zeros((1, T_max, S_max, 1), jnp.float32)
    tim_m = jnp.ones((1, T_max, 1), jnp.float32)
    params = enc.init(rng, feats, set_m, tim_m)["params"]

    def embed_fn(mods_batch: List[Any]) -> jnp.ndarray:
        feats_list, set_list, time_list = [], [], []
        for mods in mods_batch:
            f, s, t = featurize_modules_trajectory([mods], T_max=T_max, S_max=S_max)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        Fb = jnp.stack(feats_list, 0)
        Sb = jnp.stack(set_list, 0)
        Tb = jnp.stack(time_list, 0)
        return enc.apply({"params": params}, Fb, Sb, Tb)  # type: ignore (B,out_dim)

    return embed_fn


# ---------- FiLM UNet-3D ----------
class FiLM(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, h, cond_vec):  # h: (B,H,W,K,C)
        g = nn.swish(nn.Dense(self.out_ch)(cond_vec))
        b = nn.swish(nn.Dense(self.out_ch)(cond_vec))
        g = g[:, None, None, None, :]
        b = b[:, None, None, None, :]
        return h * (1.0 + g) + b


class ResBlock3D(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x, cond_vec):
        h = nn.GroupNorm(num_groups=8)(x)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(h)
        h = FiLM(self.ch)(h, cond_vec)
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(h)
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1, 1))(x)
        return x + h


class UNet3D_FiLM(nn.Module):
    ch: int = 64
    depth: int = 3  # number of downs
    cond_dim: int = 256  # modules(256) + labels(one-hot) size — set at runtime

    @nn.compact
    def __call__(self, x, t_emb: jnp.ndarray, cond_vec: jnp.ndarray):
        """
        x: (B,H,W,K,C)   t_emb: (B,128)   cond_vec: (B,cond_dim)
        returns epsilon or v with same channel count as x (C)
        """
        # fuse time into cond
        t_h = nn.swish(nn.Dense(self.cond_dim)(t_emb))
        c = jnp.concatenate([cond_vec, t_h], axis=-1)
        c = nn.swish(nn.Dense(self.cond_dim)(c))  # final cond vector

        # encoder
        hs = []
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(x)
        h = ResBlock3D(self.ch)(h, c)
        hs.append(h)
        # downsample only H,W (keep K)
        ch = self.ch
        for _ in range(self.depth - 1):
            h = nn.max_pool(
                h, window_shape=(2, 2, 1), strides=(2, 2, 1), padding="SAME"
            )
            ch *= 2
            h = ResBlock3D(ch)(h, c)
            hs.append(h)

        # bottleneck
        h = ResBlock3D(ch * 2)(nn.Conv(ch * 2, (3, 3, 3), padding="SAME")(h), c)

        # decoder
        for skip in hs[::-1]:
            h = nn.ConvTranspose(ch, kernel_size=(2, 2, 1), strides=(2, 2, 1))(h)
            h = jnp.concatenate([h, skip], axis=-1)
            h = ResBlock3D(ch)(h, c)
            ch //= 2 if ch > self.ch else self.ch
        eps = nn.Conv(x.shape[-1], (1, 1, 1))(h)
        return eps


# ---------- Energy network on volumes ----------
class EnergyNetwork(nn.Module):
    ch: int = 64
    cond_dim: int = 256

    @nn.compact
    def __call__(self, x, cond_vec):  # x: (B,H,W,K,C)  cond_vec: (B,cond_dim)
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(x)
        h = ResBlock3D(self.ch)(h, cond_vec)
        h = nn.max_pool(h, (2, 2, 1), (2, 2, 1), padding="SAME")
        h = ResBlock3D(self.ch * 2)(h, cond_vec)
        h = nn.max_pool(h, (2, 2, 1), (2, 2, 1), padding="SAME")
        h = ResBlock3D(self.ch * 4)(h, cond_vec)
        h = jnp.mean(h, axis=(1, 2, 3))  # GAP
        h = jnp.concatenate([h, cond_vec], axis=-1)
        h = nn.tanh(nn.Dense(256)(h))
        e = nn.Dense(1)(h)
        return e.squeeze(-1)  # (B,)
