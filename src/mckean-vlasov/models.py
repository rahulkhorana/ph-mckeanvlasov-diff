from __future__ import annotations
from typing import Tuple, List

import jax
import jax.numpy as jnp
from flax import linen as nn

_F32 = jnp.float32


# -----------------------------
# Utilities (robust numerics)
# -----------------------------
def _nn(x: jnp.ndarray) -> jnp.ndarray:
    # Replace NaN, +Inf, -Inf -> 0 and clamp large magnitudes to keep norms sane.
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.clip(x, -1e3, 1e3)


# -----------------------------
# Sinusoidal time embedding
# -----------------------------
def time_embed(t: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """
    t: (B,) float32 (any range). Returns (B, dim).
    """
    t = _nn(t.astype(_F32))
    half = dim // 2
    # Frequencies span 1 .. 10k in log-space
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half, dtype=_F32))
    ang = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if emb.shape[-1] < dim:
        pad = jnp.zeros((emb.shape[0], dim - emb.shape[-1]), dtype=_F32)
        emb = jnp.concatenate([emb, pad], axis=-1)
    return emb.astype(_F32)


# -----------------------------
# FiLM conditioning pieces
# -----------------------------
class FiLM(nn.Module):
    c: int  # channels to modulate
    cond_dim: int  # conditioning vector dim
    hidden: int = 256

    @nn.compact
    def __call__(self, h: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        h: (B,H,W,K,C)
        cond: (B, cond_dim)
        """
        cond = _nn(cond)
        x = nn.silu(nn.Dense(self.hidden)(cond))
        gb = nn.Dense(2 * self.c)(x)  # (B, 2C)
        gamma, beta = jnp.split(gb, 2, axis=-1)  # (B,C), (B,C)
        gamma = gamma[:, None, None, None, :]  # (B,1,1,1,C)
        beta = beta[:, None, None, None, :]
        return h * (1.0 + gamma) + beta


class ResBlock3D(nn.Module):
    c: int
    cond_dim: int
    groups: int = 8
    use_skip_proj: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        x: (B,H,W,K,Cin)
        """
        x = _nn(x)
        cond = _nn(cond)

        Cin = x.shape[-1]
        h = x
        h = nn.GroupNorm(num_groups=self.groups)(h)
        h = nn.silu(h)
        h = nn.Conv(self.c, kernel_size=(3, 3, 3), padding="SAME")(h)
        h = FiLM(self.c, self.cond_dim)(h, cond)

        h = nn.GroupNorm(num_groups=self.groups)(h)
        h = nn.silu(h)
        h = nn.Conv(self.c, kernel_size=(3, 3, 3), padding="SAME")(h)
        h = FiLM(self.c, self.cond_dim)(h, cond)

        if self.use_skip_proj or Cin != self.c:
            x = nn.Conv(self.c, kernel_size=(1, 1, 1), padding="SAME")(x)
        return x + h


class DownsampleHW(nn.Module):
    c: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _nn(x)
        # Downsample only H,W by 2; keep K
        return nn.Conv(
            self.c, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding="SAME"
        )(x)


class UpsampleHW(nn.Module):
    c: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _nn(x)
        # Nearest neighbor upsample on H,W then conv
        x = jnp.repeat(x, 2, axis=1)  # H
        x = jnp.repeat(x, 2, axis=2)  # W
        x = nn.Conv(self.c, kernel_size=(3, 3, 3), padding="SAME")(x)
        return x


# -----------------------------
# 3D U-Net with FiLM conditioning
# -----------------------------
class UNet3D_FiLM(nn.Module):
    ch: int = 48  # base channels
    cond_dim: int = 384  # e.g., y(128) + modules(256)
    t_dim: int = 128
    groups: int = 8

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, t_emb: jnp.ndarray, cond_vec: jnp.ndarray
    ) -> jnp.ndarray:
        """
        x: (B,H,W,K,C)
        t_emb: (B, t_dim)
        cond_vec: (B, cond_dim)
        returns: (B,H,W,K,C) (same C as input)
        """
        x = _nn(x)
        t_emb = _nn(t_emb)
        cond_vec = _nn(cond_vec)

        B, H, W, K, Cin = x.shape

        # Fused conditioning code (time + cond), sanitized
        t_h = nn.silu(nn.Dense(self.t_dim)(_nn(t_emb)))
        c_h = nn.silu(nn.Dense(self.t_dim)(_nn(cond_vec)))
        fused = nn.LayerNorm()(jnp.concatenate([t_h, c_h], axis=-1))  # (B, 2*t_dim)
        cond = nn.silu(nn.Dense(self.cond_dim)(fused))  # (B, cond_dim)

        # Stem
        h0 = nn.Conv(self.ch, kernel_size=(3, 3, 3), padding="SAME")(x)

        # Encoder
        h1 = ResBlock3D(self.ch, self.cond_dim, groups=self.groups)(h0, cond)
        h1 = ResBlock3D(self.ch, self.cond_dim, groups=self.groups)(h1, cond)
        d1 = DownsampleHW(self.ch * 2)(h1)  # (H/2,W/2)

        h2 = ResBlock3D(self.ch * 2, self.cond_dim, groups=self.groups)(d1, cond)
        h2 = ResBlock3D(self.ch * 2, self.cond_dim, groups=self.groups)(h2, cond)
        d2 = DownsampleHW(self.ch * 4)(h2)  # (H/4,W/4)

        h3 = ResBlock3D(self.ch * 4, self.cond_dim, groups=self.groups)(d2, cond)
        h3 = ResBlock3D(self.ch * 4, self.cond_dim, groups=self.groups)(h3, cond)

        # Bottleneck
        b = ResBlock3D(self.ch * 4, self.cond_dim, groups=self.groups)(h3, cond)

        # Decoder
        u2 = UpsampleHW(self.ch * 4)(b)
        u2 = jnp.concatenate([u2, h2], axis=-1)
        u2 = ResBlock3D(self.ch * 4, self.cond_dim, groups=self.groups)(u2, cond)
        u2 = ResBlock3D(self.ch * 4, self.cond_dim, groups=self.groups)(u2, cond)

        u1 = UpsampleHW(self.ch * 2)(u2)
        u1 = jnp.concatenate([u1, h1], axis=-1)
        u1 = ResBlock3D(self.ch * 2, self.cond_dim, groups=self.groups)(u1, cond)
        u1 = ResBlock3D(self.ch * 2, self.cond_dim, groups=self.groups)(u1, cond)

        out = nn.GroupNorm(num_groups=self.groups)(u1)
        out = nn.silu(out)
        out = nn.Conv(Cin, kernel_size=(3, 3, 3), padding="SAME")(out)
        return out.astype(_F32)


# -----------------------------
# Energy network (kept for compatibility)
# -----------------------------
class EnergyNetwork(nn.Module):
    ch: int = 48
    cond_dim: int = 384

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond_vec: jnp.ndarray) -> jnp.ndarray:
        x = _nn(x)
        cond_vec = _nn(cond_vec)

        B, H, W, K, C = x.shape
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(x)
        h = nn.silu(h)
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(h)
        h = nn.silu(h)
        # Global average pool
        h = jnp.mean(h, axis=(1, 2, 3))  # (B, ch)
        c = nn.silu(nn.Dense(self.ch)(cond_vec))  # (B, ch)
        z = jnp.concatenate([h, c], axis=-1)  # (B, 2ch)
        z = nn.silu(nn.Dense(self.ch)(z))
        e = nn.Dense(1)(z)
        return e.squeeze(-1).astype(_F32)  # (B,)


# -----------------------------
# Modules trajectory encoder
# -----------------------------
class ModulesTrajectoryEncoder(nn.Module):
    out_dim: int = 256
    hidden: int = 256
    elem_hidden: int = 128

    @nn.compact
    def __call__(
        self,
        feats: jnp.ndarray,  # (B, T, S, F)
        set_b: jnp.ndarray,  # (B, T, S, 1)  1.0 for valid, 0.0 for pad
        time_b: jnp.ndarray,  # (B, T, 1)
    ) -> jnp.ndarray:
        # Sanitize inputs
        feats = _nn(feats)
        set_b = jnp.clip(_nn(set_b), 0.0, 1.0)
        time_b = _nn(time_b)

        B, T, S, F = feats.shape

        # Per-element MLP (masked)
        x = nn.silu(nn.Dense(self.elem_hidden)(feats))  # (B,T,S,H)
        x = nn.silu(nn.Dense(self.elem_hidden)(x))  # (B,T,S,H)

        # Masked mean over S -> (B,T,1,H)
        x = x * set_b
        denom = jnp.maximum(jnp.sum(set_b, axis=2, keepdims=True), 1e-6)  # (B,T,1,1)
        pooled = jnp.sum(x, axis=2, keepdims=True) / denom  # (B,T,1,H)

        # Time projection: Dense(last=1)->H gives (B,T,H); add axis 2 → (B,T,1,H)
        tproj = nn.silu(nn.Dense(self.elem_hidden)(time_b))  # (B,T,H)
        tproj = tproj[:, :, None, :]  # (B,T,1,H)

        # Fuse and collapse T
        fused = jnp.concatenate([pooled, tproj], axis=-1)  # (B,T,1,2H)
        fused = fused.reshape((B, T, -1))  # (B,T,2H)
        fused = jnp.mean(fused, axis=1)  # (B,2H)
        fused = _nn(fused)

        z = nn.silu(nn.Dense(self.hidden)(fused))
        z = nn.silu(nn.Dense(self.hidden)(z))
        out = nn.Dense(self.out_dim)(z)  # (B,out_dim)
        return _nn(out.astype(_F32))


# -----------------------------
# Minimal safe featurizer
# -----------------------------
def featurize_modules_trajectory(
    mods_seq_list: List[List],  # list containing one sequence (you call with [mods])
    T_max: int = 1,
    S_max: int = 16,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      feats: (1, 1, S_max, F)  with F = 13 + S_max (positional one-hot)
      set_b: (1, 1, S_max, 1)  mask {0,1}
      time_b: (1, 1, 1)

    Conservative placeholder that *guarantees finite tensors* even if the
    input module sequence contains NaN/Inf. Swap with a richer featurizer
    as needed; keep the same signature/shapes.
    """
    F = 13 + S_max

    modules = mods_seq_list[0] if len(mods_seq_list) > 0 else []
    try:
        n_raw = len(modules)
    except Exception:
        n_raw = 0
    n = int(max(0, min(n_raw, S_max)))

    feats = jnp.zeros((1, 1, S_max, F), dtype=_F32)
    set_b = jnp.zeros((1, 1, S_max, 1), dtype=_F32)

    # Positional one-hot in the last S_max dims of F
    if n > 0:
        pos_eye = jnp.eye(S_max, dtype=_F32)[:n]  # (n, S_max)
        feats = feats.at[0, 0, :n, 13:].set(pos_eye)
        set_b = set_b.at[0, 0, :n, 0].set(1.0)

    # time scalar = 1.0 (finite)
    time_b = jnp.ones((1, 1, 1), dtype=_F32)

    # Final sanitize pass (defensive)
    feats = _nn(feats)
    set_b = jnp.clip(_nn(set_b), 0.0, 1.0)
    time_b = _nn(time_b)
    return feats, set_b, time_b
