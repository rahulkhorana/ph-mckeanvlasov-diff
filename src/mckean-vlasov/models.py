import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as nn_attn
from typing import Any, List, Tuple


# ---------------- time embedding (param-free) ----------------
def time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """
    Param-free sinusoidal embed of continuous time t in [0,1].
    Returns (B, dim). Safe to call from Python (no flax params).
    """
    t = t_cont.astype(jnp.float32)
    half = dim // 2
    # geometric frequency range
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half))
    ang = t[:, None] * (1.0 / freqs[None, :])
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)  # (B, 2*half)
    if emb.shape[-1] < dim:
        # pad to requested dim
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb


# ----------- modules trajectory encoder (your version) -----------
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
    mods_trajectory: List[List[Any]],
    T_max: int = 1,
    S_max: int = 16,
    add_pos_ids: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    mods_trajectory: list length T of (list of module arrays/tensors)
    Returns:
      feats:     (T_max, S_max, F)
      set_mask:  (T_max, S_max, 1)
      time_mask: (T_max, 1)
    NOTE: caller will batch these into (B,T,S,F) etc.
    """
    traj = mods_trajectory[:T_max]
    T = len(traj)
    feat_slices, set_masks = [], []
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
            feat_slices.append(np.stack(row, axis=0))
            set_masks.append(
                np.array(
                    [1.0] * min(len(traj[t]), S_max)
                    + [0.0] * (S_max - min(len(traj[t]), S_max))
                )[:, None]
            )
        else:
            F = (
                feat_slices[0].shape[1]
                if feat_slices
                else (13 + (S_max if add_pos_ids else 0))
            )
            feat_slices.append(np.zeros((S_max, F), dtype=np.float32))
            set_masks.append(np.zeros((S_max, 1), dtype=np.float32))
    feats = jnp.array(np.stack(feat_slices, axis=0))  # (T_max,S_max,F)
    set_mask = jnp.array(np.stack(set_masks, axis=0))  # (T_max,S_max,1)
    time_mask = jnp.array(np.array([1.0] * T + [0.0] * (T_max - T), dtype=np.float32))[
        :, None
    ]  # (T_max,1)
    return feats, set_mask, time_mask


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
            attn_mask = nn_attn.make_attention_mask(q_mask, k_mask)
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
        seeds = self.param(
            "seed", nn.initializers.normal(stddev=0.02), (self.m, self.d)
        )
        q = jnp.repeat(seeds[None, :, :], B, axis=0)  # (B,m,d)
        k_mask_bool = None
        if key_mask is not None:
            k_mask_bool = (
                (key_mask[..., 0] > 0) if key_mask.ndim == 3 else key_mask.astype(bool)
            )
        q_mask_bool = jnp.ones((B, self.m), dtype=bool)
        return MHA(self.d, self.h)(q, x, x, q_mask=q_mask_bool, k_mask=k_mask_bool)


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
    def __call__(self, feats, set_mask, time_mask):
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
            x = x + y
            h = nn.LayerNorm()(x)
            y = nn.Dense(self.d_time)(nn.gelu(nn.Dense(self.d_time * 4)(h)))
            x = x + y

        denom = jnp.clip(jnp.sum(time_mask, axis=1), 1e-6, None)
        x_mean = jnp.sum(x * time_mask, axis=1) / denom
        z = nn.Dense(self.out_dim)(nn.gelu(nn.Dense(self.out_dim)(x_mean)))
        return z


# ---------------- 3D blocks + UNet FiLM ----------------


def _gn_groups(c: int) -> int:
    """Choose a GroupNorm group count that divides channels c."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class Res3D(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, gamma=None, beta=None):
        c1 = x.shape[-1]
        g1 = _gn_groups(c1)
        h = nn.GroupNorm(num_groups=g1)(x)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3, 3, 3), (1, 1, 1), padding="SAME")(h)

        c2 = h.shape[-1]
        g2 = _gn_groups(c2)
        h = nn.GroupNorm(num_groups=g2)(h)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3, 3, 3), (1, 1, 1), padding="SAME")(h)

        if (gamma is not None) and (beta is not None):
            h = h * (1.0 + gamma) + beta

        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, (1, 1, 1))(x)

        return x + h


class UNet3D_FiLM(nn.Module):
    """
    Conditional 3D UNet with FiLM conditioning from concatenated [t_emb, cond_vec].
    Input/Output: (B, H, W, K, C)
    Downsample H,W by 2 twice (K untouched), then upsample with ConvTranspose.
    """

    ch: int = 64

    @nn.compact
    def __call__(self, x, t_emb, cond_vec):
        B, H, W, K, C = x.shape
        hcond = jnp.concatenate([t_emb, cond_vec], axis=-1)

        # FiLM scales/biases for three stages
        g1 = nn.Dense(self.ch)(hcond)[:, None, None, None, :]
        b1 = nn.Dense(self.ch)(hcond)[:, None, None, None, :]
        g2 = nn.Dense(self.ch * 2)(hcond)[:, None, None, None, :]
        b2 = nn.Dense(self.ch * 2)(hcond)[:, None, None, None, :]
        g3 = nn.Dense(self.ch * 4)(hcond)[:, None, None, None, :]
        b3 = nn.Dense(self.ch * 4)(hcond)[:, None, None, None, :]

        # encoder
        h1 = Res3D(self.ch)(x, g1, b1)  # (B,H,W,K,ch)
        d1 = nn.max_pool(h1, window_shape=(2, 2, 1), strides=(2, 2, 1), padding="SAME")
        h2 = Res3D(self.ch * 2)(d1, g2, b2)  # (B,H/2,W/2,K,2ch)
        d2 = nn.max_pool(h2, window_shape=(2, 2, 1), strides=(2, 2, 1), padding="SAME")
        h3 = Res3D(self.ch * 4)(d2, g3, b3)  # (B,H/4,W/4,K,4ch)

        # decoder
        u2 = nn.ConvTranspose(self.ch * 2, kernel_size=(2, 2, 1), strides=(2, 2, 1))(h3)
        if (u2.shape[1] != h2.shape[1]) or (u2.shape[2] != h2.shape[2]):
            u2 = jax.image.resize(
                u2, (B, h2.shape[1], h2.shape[2], K, u2.shape[-1]), method="nearest"
            )
        u2 = jnp.concatenate([u2, h2], axis=-1)
        u2 = Res3D(self.ch * 2)(u2, g2, b2)

        u1 = nn.ConvTranspose(self.ch, kernel_size=(2, 2, 1), strides=(2, 2, 1))(u2)
        if (u1.shape[1] != h1.shape[1]) or (u1.shape[2] != h1.shape[2]):
            u1 = jax.image.resize(
                u1, (B, h1.shape[1], h1.shape[2], K, u1.shape[-1]), method="nearest"
            )
        u1 = jnp.concatenate([u1, h1], axis=-1)
        u1 = Res3D(self.ch)(u1, g1, b1)

        # final head: predict eps or v
        return nn.Conv(C, (1, 1, 1))(u1)


# ---------- Energy network on volumes (FiLM) ----------
class FiLM(nn.Module):
    feat: int

    @nn.compact
    def __call__(self, h, cond_vec):
        """
        h: (B,H,W,K,feat)
        cond_vec: (B, D)
        """
        gamma_beta = nn.Dense(2 * self.feat)(cond_vec)  # (B,2F)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)  # (B,F),(B,F)
        gamma = gamma[:, None, None, None, :]
        beta = beta[:, None, None, None, :]
        return h * (1.0 + gamma) + beta


class ResBlock3D(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x, cond_vec):
        g1 = _gn_groups(x.shape[-1])
        h = nn.GroupNorm(num_groups=g1)(x)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(h)

        h = FiLM(self.ch)(h, cond_vec)

        g2 = _gn_groups(h.shape[-1])
        h = nn.GroupNorm(num_groups=g2)(h)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(h)

        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1, 1))(x)
        return x + h


class EnergyNetwork(nn.Module):
    ch: int = 64
    cond_dim: int = 256  # D of cond_vec (label emb + modules emb)

    @nn.compact
    def __call__(self, x, cond_vec):  # x: (B,H,W,K,C)  cond_vec: (B,cond_dim)
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(x)
        h = ResBlock3D(self.ch)(h, cond_vec)
        h = nn.max_pool(h, (2, 2, 1), (2, 2, 1), padding="SAME")
        h = ResBlock3D(self.ch * 2)(h, cond_vec)
        h = nn.max_pool(h, (2, 2, 1), (2, 2, 1), padding="SAME")
        h = ResBlock3D(self.ch * 4)(h, cond_vec)
        h = jnp.mean(h, axis=(1, 2, 3))  # global average over H,W,K -> (B, 4ch)
        h = jnp.concatenate([h, cond_vec], axis=-1)
        h = nn.tanh(nn.Dense(256)(h))
        e = nn.Dense(1)(h)
        return e.squeeze(-1)  # (B,)
