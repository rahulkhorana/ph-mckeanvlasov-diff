# models.py
from __future__ import annotations
from typing import Tuple, List

import jax
import jax.numpy as jnp
from flax import linen as nn

# -----------------------------
# Dtypes and small helpers
# -----------------------------
_F32 = jnp.float32
_BF = jnp.bfloat16

# Parameters & activations in bf16 for memory; keep sensitive math in f32
_PARAM = _BF
_ACT = _BF
_NORM = _F32


def _san(x: jnp.ndarray) -> jnp.ndarray:
    """Sanitize numbers; keep dtype, caller casts as needed."""
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _clip(x: jnp.ndarray, v: float = 1e3) -> jnp.ndarray:
    return jnp.clip(x, -v, v)


# -----------------------------
# Sinusoidal time embedding
# -----------------------------
def time_embed(t: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """
    t: (B,) float32 (any range). Returns (B, dim) in float32.
    """
    t = _clip(_san(t.astype(_F32)))
    half = dim // 2
    # Frequencies span ~[1, 10k]
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half, dtype=_F32))
    ang = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if emb.shape[-1] < dim:
        pad = jnp.zeros((emb.shape[0], dim - emb.shape[-1]), dtype=_F32)
        emb = jnp.concatenate([emb, pad], axis=-1)
    return emb  # f32 on purpose


# -----------------------------
# FiLM conditioning
# -----------------------------
class FiLM(nn.Module):
    c: int  # channels to modulate
    cond_dim: int  # conditioning vector dim
    hidden: int = 256

    @nn.compact
    def __call__(self, h: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        h: (B,H,W,K,C)  (ACT dtype)
        cond: (B, cond_dim)  (ACT dtype)
        """
        cond = _clip(_san(cond)).astype(_ACT)
        x = nn.silu(nn.Dense(self.hidden, dtype=_ACT, param_dtype=_PARAM)(cond))
        gb = nn.Dense(2 * self.c, dtype=_ACT, param_dtype=_PARAM)(x)  # (B, 2C)
        gamma, beta = jnp.split(gb, 2, axis=-1)  # (B,C), (B,C)
        gamma = gamma[:, None, None, None, :]  # (B,1,1,1,C)
        beta = beta[:, None, None, None, :]
        return h * (1.0 + gamma) + beta


# -----------------------------
# 3D building blocks (HW down/up; light mixing over K)
# -----------------------------
class ResBlock3D(nn.Module):
    c: int
    cond_dim: int
    groups: int = 8
    use_skip_proj: bool = False
    k_kernel: int = (
        1  # set to 1 to save VRAM; 3x3x3 explodes memory at B=4, 128^2, K=3.
    )

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        x: (B,H,W,K,Cin)  (ACT dtype)
        cond: (B,cond_dim) (ACT dtype)
        """
        x = _clip(_san(x)).astype(_ACT)
        cond = _clip(_san(cond)).astype(_ACT)

        Cin = x.shape[-1]
        h = x

        # Norm in f32 for stability → cast back to ACT for convs.
        h = nn.GroupNorm(num_groups=self.groups, dtype=_NORM)(h.astype(_NORM))
        h = h.astype(_ACT)
        h = nn.silu(h)
        h = nn.Conv(
            self.c,
            kernel_size=(3, 3, self.k_kernel),
            padding="SAME",
            dtype=_ACT,
            param_dtype=_PARAM,
        )(h)
        h = FiLM(self.c, self.cond_dim)(h, cond)

        h = nn.GroupNorm(num_groups=self.groups, dtype=_NORM)(h.astype(_NORM))
        h = h.astype(_ACT)
        h = nn.silu(h)
        h = nn.Conv(
            self.c,
            kernel_size=(3, 3, self.k_kernel),
            padding="SAME",
            dtype=_ACT,
            param_dtype=_PARAM,
        )(h)
        h = FiLM(self.c, self.cond_dim)(h, cond)

        if self.use_skip_proj or Cin != self.c:
            x = nn.Conv(
                self.c,
                kernel_size=(1, 1, 1),
                padding="SAME",
                dtype=_ACT,
                param_dtype=_PARAM,
            )(x)
        return (x + h).astype(_ACT)


class DownsampleHW(nn.Module):
    c: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _clip(_san(x)).astype(_ACT)
        # Downsample only H,W by 2; keep K (stride K=1).
        return nn.Conv(
            self.c,
            kernel_size=(3, 3, 1),
            strides=(2, 2, 1),
            padding="SAME",
            dtype=_ACT,
            param_dtype=_PARAM,
        )(x)


class UpsampleHW(nn.Module):
    c: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _clip(_san(x)).astype(_ACT)
        # Nearest-neighbor upsample on H,W then conv
        x = jnp.repeat(x, 2, axis=1)  # H
        x = jnp.repeat(x, 2, axis=2)  # W
        x = nn.Conv(
            self.c,
            kernel_size=(3, 3, 1),
            padding="SAME",
            dtype=_ACT,
            param_dtype=_PARAM,
        )(x)
        return x


# -----------------------------
# 3D U-Net with FiLM conditioning
# -----------------------------
class UNet3D_FiLM(nn.Module):
    ch: int = 48  # base channels
    cond_dim: int = 384  # e.g., y(128) + modules(256)
    t_dim: int = 128
    groups: int = 8
    k_kernel: int = 1  # keep convs (3,3,1) to stay within VRAM

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, t_emb: jnp.ndarray, cond_vec: jnp.ndarray
    ) -> jnp.ndarray:
        """
        x: (B,H,W,K,C)
        t_emb: (B, t_dim) — f32
        cond_vec: (B, cond_dim)
        returns: (B,H,W,K,C) in f32
        """
        # Sanitize and cast inputs
        x = _clip(_san(x)).astype(_ACT)
        t_emb = _clip(_san(t_emb)).astype(_F32)  # keep in f32
        cond_vec = _clip(_san(cond_vec)).astype(_F32)  # fuse in f32

        B, H, W, K, Cin = x.shape

        # Fuse time + cond in f32, then map to ACT for the network
        t_h = nn.silu(nn.Dense(self.t_dim, dtype=_F32, param_dtype=_PARAM)(t_emb))
        c_h = nn.silu(nn.Dense(self.t_dim, dtype=_F32, param_dtype=_PARAM)(cond_vec))
        fused = nn.LayerNorm(dtype=_F32)(
            jnp.concatenate([t_h, c_h], axis=-1)
        )  # (B,2*t_dim)
        cond = nn.silu(
            nn.Dense(self.cond_dim, dtype=_F32, param_dtype=_PARAM)(fused)
        ).astype(_ACT)

        # Stem
        h0 = nn.Conv(
            self.ch,
            kernel_size=(3, 3, 1),
            padding="SAME",
            dtype=_ACT,
            param_dtype=_PARAM,
        )(x)

        # Encoder
        h1 = ResBlock3D(
            self.ch, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(h0, cond)
        h1 = ResBlock3D(
            self.ch, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(h1, cond)
        d1 = DownsampleHW(self.ch * 2)(h1)  # (H/2,W/2)

        h2 = ResBlock3D(
            self.ch * 2, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(d1, cond)
        h2 = ResBlock3D(
            self.ch * 2, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(h2, cond)
        d2 = DownsampleHW(self.ch * 4)(h2)  # (H/4,W/4)

        h3 = ResBlock3D(
            self.ch * 4, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(d2, cond)
        h3 = ResBlock3D(
            self.ch * 4, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(h3, cond)

        # Bottleneck
        b = ResBlock3D(
            self.ch * 4, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(h3, cond)

        # Decoder
        u2 = UpsampleHW(self.ch * 4)(b)
        u2 = jnp.concatenate([u2, h2], axis=-1).astype(_ACT)
        u2 = ResBlock3D(
            self.ch * 4, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(u2, cond)
        u2 = ResBlock3D(
            self.ch * 4, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(u2, cond)

        u1 = UpsampleHW(self.ch * 2)(u2)
        u1 = jnp.concatenate([u1, h1], axis=-1).astype(_ACT)
        u1 = ResBlock3D(
            self.ch * 2, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(u1, cond)
        u1 = ResBlock3D(
            self.ch * 2, self.cond_dim, groups=self.groups, k_kernel=self.k_kernel
        )(u1, cond)

        out = nn.GroupNorm(num_groups=self.groups, dtype=_NORM)(u1.astype(_NORM))
        out = nn.silu(out).astype(_ACT)
        out = nn.Conv(
            Cin, kernel_size=(3, 3, 1), padding="SAME", dtype=_ACT, param_dtype=_PARAM
        )(out)
        return out.astype(_F32)  # always return f32 to the loss


# -----------------------------
# Energy network (kept for compatibility)
# -----------------------------
class EnergyNetwork(nn.Module):
    ch: int = 48
    cond_dim: int = 384

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond_vec: jnp.ndarray) -> jnp.ndarray:
        x = _clip(_san(x)).astype(_ACT)
        cond_vec = _clip(_san(cond_vec)).astype(_F32)

        B, H, W, K, C = x.shape
        h = nn.Conv(self.ch, (3, 3, 1), padding="SAME", dtype=_ACT, param_dtype=_PARAM)(
            x
        )
        h = nn.silu(h)
        h = nn.Conv(self.ch, (3, 3, 1), padding="SAME", dtype=_ACT, param_dtype=_PARAM)(
            h
        )
        h = nn.silu(h)
        # Global average pool (compute in f32)
        h = jnp.mean(h.astype(_F32), axis=(1, 2, 3))  # (B, ch)
        c = nn.silu(
            nn.Dense(self.ch, dtype=_F32, param_dtype=_PARAM)(cond_vec)
        )  # (B, ch)
        z = jnp.concatenate([h, c], axis=-1)  # (B, 2ch)
        z = nn.silu(nn.Dense(self.ch, dtype=_F32, param_dtype=_PARAM)(z))
        e = nn.Dense(1, dtype=_F32, param_dtype=_PARAM)(z)
        return e.squeeze(-1).astype(_F32)  # (B,)


# -----------------------------
# Modules trajectory encoder
# -----------------------------
class ModulesTrajectoryEncoder(nn.Module):
    out_dim: int = 256
    hidden: int = 256
    elem_hidden: int = 128

    # --- tiny shape normalizers ---
    @staticmethod
    def _force_BT1(x: jnp.ndarray) -> jnp.ndarray:
        """
        Return (B,T,1) from anything vaguely like time info.
        Accepts (B,T), (B,T,1), (B,T,1,Dt), (B,1), (B,1,1,Dt), etc.
        Collapses trailing dims by mean, keeps a single length-1 axis at pos 2.
        """
        x = _clip(_san(x)).astype(_F32)
        if x.ndim == 1:  # (B,) -> (B,1,1)
            x = x[:, None, None]
        elif x.ndim == 2:  # (B,T) -> (B,T,1)
            x = x[:, :, None]
        elif x.ndim >= 3:  # (B,T,...) -> collapse ... -> (B,T,1)
            # reduce all trailing dims beyond axis=1
            axes = tuple(range(2, x.ndim))
            x = jnp.mean(x, axis=axes, keepdims=True)
        return x  # (B,T,1)

    @staticmethod
    def _force_BTS1(x: jnp.ndarray) -> jnp.ndarray:
        """
        Ensure (B,T,S,1) mask. Accepts (B,T,S), (B,T,S,1), (B,S,1), ...
        """
        x = _clip(_san(x)).astype(_F32)
        if x.ndim == 3:  # (B,T,S) -> (B,T,S,1)
            x = x[..., None]
        elif x.ndim == 2:  # (B,S) -> (B,1,S,1)
            x = x[:, None, :, None]
        return jnp.clip(x, 0.0, 1.0)

    @staticmethod
    def _force_BTSF(x: jnp.ndarray) -> jnp.ndarray:
        """
        Ensure (B,T,S,F) features. Accepts (B,S,F) -> (B,1,S,F) etc.
        """
        x = _clip(_san(x)).astype(_F32)
        if x.ndim == 3:  # (B,S,F) -> (B,1,S,F)
            x = x[:, None, :, :]
        elif x.ndim == 2:  # (S,F) -> (1,1,S,F)
            x = x[None, None, :, :]
        return x

    @nn.compact
    def __call__(
        self,
        feats: jnp.ndarray,  # (B,T,S,F) preferred
        set_b: jnp.ndarray,  # (B,T,S,1) preferred
        time_b: jnp.ndarray,  # (B,T,1)   preferred
    ) -> jnp.ndarray:
        # normalize shapes
        feats = self._force_BTSF(feats)  # (B,T,S,F)
        set_b = self._force_BTS1(set_b)  # (B,T,S,1)
        time_b = self._force_BT1(time_b)  # (B,T,1)

        B, T, S, F = feats.shape

        # Per-element MLP (masked)
        x = nn.silu(
            nn.Dense(self.elem_hidden, dtype=_F32, param_dtype=_PARAM)(feats)
        )  # (B,T,S,H)
        x = nn.silu(
            nn.Dense(self.elem_hidden, dtype=_F32, param_dtype=_PARAM)(x)
        )  # (B,T,S,H)

        # Masked mean over S
        x = x * set_b  # broadcast over last dim
        denom = jnp.maximum(jnp.sum(set_b, axis=2, keepdims=True), 1e-6)  # (B,T,1,1)
        pooled = jnp.sum(x, axis=2, keepdims=True) / denom  # (B,T,1,H)

        # Time projection → force (B,T,1,H) ***always***
        # Flatten any trailing content of time_b before Dense:
        tflat = time_b.reshape((time_b.shape[0], time_b.shape[1], -1))  # (B,T,Dt)
        tproj = nn.silu(
            nn.Dense(self.elem_hidden, dtype=_F32, param_dtype=_PARAM)(tflat)
        )  # (B,T,H)
        tproj = tproj[:, :, None, :]  # add S=1 axis → (B,T,1,H)

        fused = jnp.concatenate([pooled, tproj], axis=-1)  # (B,T,1,2H)

        # Collapse T
        fused = fused.reshape((B, T, -1))  # (B,T,2H)
        fused = jnp.mean(fused, axis=1)  # (B,2H)

        z = nn.silu(nn.Dense(self.hidden, dtype=_F32, param_dtype=_PARAM)(fused))
        z = nn.silu(nn.Dense(self.hidden, dtype=_F32, param_dtype=_PARAM)(z))
        out = nn.Dense(self.out_dim, dtype=_F32, param_dtype=_PARAM)(z)  # (B, out_dim)
        return _clip(_san(out)).astype(_F32)


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
      feats:  (1, 1, S_max, F)  with F = 13 + S_max (positional one-hot)
      set_b:  (1, 1, S_max, 1)  mask {0,1}
      time_b: (1, 1, 1)
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

    if n > 0:
        pos_eye = jnp.eye(S_max, dtype=_F32)[:n]  # (n, S_max)
        feats = feats.at[0, 0, :n, 13:].set(pos_eye)
        set_b = set_b.at[0, 0, :n, 0].set(1.0)

    time_b = jnp.ones((1, 1, 1), dtype=_F32)

    return _san(feats), jnp.clip(_san(set_b), 0.0, 1.0), _san(time_b)
