# models.py — Denoised GuidanceNet, Attention, and FiLM UNet3D with full Module Encoder
from typing import Any, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as nn_attn


# -------- sinusoidal time embedding --------
def time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000.0), half, dtype=jnp.float32))
    ang = t_cont[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


# -------- modules featurization --------
def _robust_stats(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, np.float32).ravel()
    if a.size == 0:
        return np.zeros(13, np.float32)
    finite = np.isfinite(a)
    fr = float(finite.mean())
    a = a[finite]
    if a.size == 0:
        return np.zeros(13, np.float32)
    mean, std = float(a.mean()), float(a.std() + 1e-6)
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    mn, mx = float(a.min()), float(a.max())
    l1, l2 = float(np.abs(a).sum()), float(np.sqrt((a * a).sum()))
    max_abs = float(np.abs(a).max())
    logN = float(np.log1p(a.size))
    skew = float(np.mean(((a - mean) / std) ** 3)) if std > 1e-6 else 0.0
    return np.array(
        [mean, std, mn, mx, q25, q50, q75, l1, l2, logN, fr, max_abs, skew], np.float32
    )


def featurize_modules_trajectory(
    mods_traj: List[List[Any]],
    T_max: int = 1,
    S_max: int = 16,
    add_pos_ids: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    feats, set_mask, time_mask = (
        jnp.array(np.stack(slices, 0)),
        jnp.array(np.stack(masks, 0)),
        jnp.array(np.array([1.0] * T + [0.0] * (T_max - T), np.float32)[:, None]),
    )
    return feats, set_mask, time_mask


class MHA(nn.Module):
    d: int
    h: int = 4

    @nn.compact
    def __call__(self, qx, kx=None, vx=None, q_mask=None, k_mask=None):
        kx = qx if kx is None else kx
        vx = kx if vx is None else vx
        q, k, v = nn.Dense(self.d)(qx), nn.Dense(self.d)(kx), nn.Dense(self.d)(vx)
        mask = (
            nn_attn.make_attention_mask(q_mask, k_mask)
            if (q_mask is not None) and (k_mask is not None)
            else None
        )
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
        q = jnp.repeat(seeds[None, :, :], B, 0)
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
        return nn.Dense(self.out_dim)(nn.gelu(nn.Dense(self.out_dim)(x_mean)))


def build_modules_embedder(rng, out_dim: int = 256, T_max: int = 1, S_max: int = 16):
    enc = ModulesTrajectoryEncoder(out_dim=out_dim)
    F = 13 + S_max
    feats, set_m, tim_m = (
        jnp.zeros((1, T_max, S_max, F)),
        jnp.zeros((1, T_max, S_max, 1)),
        jnp.ones((1, T_max, 1)),
    )
    params = enc.init(rng, feats, set_m, tim_m)["params"]

    def embed_fn(mods_batch: List[Any]) -> jnp.ndarray:
        feats_list, set_list, time_list = [], [], []
        for mods in mods_batch:
            f, s, t = featurize_modules_trajectory([mods], T_max=T_max, S_max=S_max)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        Fb, Sb, Tb = (
            jnp.stack(feats_list, 0),
            jnp.stack(set_list, 0),
            jnp.stack(time_list, 0),
        )
        return enc.apply({"params": params}, Fb, Sb, Tb)  # type: ignore

    return embed_fn


# ------------ UNet3D Components -------------
def _safe_gn_groups(c: int) -> int:
    if c >= 32 and c % 16 == 0:
        return 16
    if c % 8 == 0:
        return 8
    if c % 4 == 0:
        return 4
    return 1


class FiLM(nn.Module):
    feat: int

    @nn.compact
    def __call__(self, h, c):
        gb = nn.Dense(2 * self.feat, kernel_init=nn.initializers.zeros)(c)
        g, b = jnp.split(gb, 2, -1)
        return h * (1.0 + g[:, None, None, None, :]) + b[:, None, None, None, :]


def _pool_hw(x):
    return nn.max_pool(x, (2, 2, 1), (2, 2, 1), "SAME")


def _up_hw(x, f):
    return nn.ConvTranspose(f, (2, 2, 1), (2, 2, 1), "SAME")(x)


class Attention3D(nn.Module):
    ch: int
    h: int = 4

    @nn.compact
    def __call__(self, x):
        B, H, W, K, C = x.shape
        h_ = nn.GroupNorm(_safe_gn_groups(self.ch))(x).reshape(B, H * W * K, C)
        a = nn.SelfAttention(self.h, qkv_features=self.ch)(h_)
        return x + a.reshape(B, H, W, K, C)


class ResBlock3D(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x, c):
        h = nn.swish(nn.GroupNorm(_safe_gn_groups(x.shape[-1]))(x))
        h = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(h)
        h = FiLM(self.ch)(h, c)
        h = nn.swish(nn.GroupNorm(_safe_gn_groups(self.ch))(h))
        h = nn.Conv(
            self.ch, (3, 3, 3), padding="SAME", kernel_init=nn.initializers.zeros
        )(h)
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1, 1))(x)
        return (x + h) / jnp.sqrt(2.0)


# ------------ Main UNet ------------
class UNet3D_FiLM(nn.Module):
    ch: int = 64

    @nn.compact
    def __call__(self, x, t, c):
        cond = nn.swish(nn.Dense(self.ch * 4)(jnp.concatenate([t, c], -1)))
        h1 = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(x)
        h1 = ResBlock3D(self.ch)(h1, cond)
        h2 = _pool_hw(h1)
        h2 = ResBlock3D(self.ch * 2)(h2, cond)
        h3 = _pool_hw(h2)
        h3 = ResBlock3D(self.ch * 4)(h3, cond)
        b = ResBlock3D(self.ch * 4)(h3, cond)
        b = Attention3D(self.ch * 4)(b)
        b = ResBlock3D(self.ch * 4)(b, cond)
        u2 = _up_hw(b, self.ch * 2)
        u2 = ResBlock3D(self.ch * 2)(jnp.concatenate([u2, h2], -1), cond)
        u1 = _up_hw(u2, self.ch)
        u1 = ResBlock3D(self.ch)(jnp.concatenate([u1, h1], -1), cond)
        return nn.Conv(x.shape[-1], (1, 1, 1), kernel_init=nn.initializers.zeros)(u1)


# ------------ SOTA Denoised Guidance Model ------------
class GuidanceNet(nn.Module):
    ch: int = 16

    @nn.compact
    def __call__(self, x, t, c):
        cond = nn.swish(nn.Dense(self.ch * 4)(jnp.concatenate([t, c], -1)))
        h1 = nn.Conv(self.ch, (3, 3, 3), padding="SAME")(x)
        h1 = ResBlock3D(self.ch, name="g_res1")(h1, cond)
        h2 = _pool_hw(h1)
        h2 = ResBlock3D(self.ch * 2, name="g_res2")(h2, cond)
        b = ResBlock3D(self.ch * 2, name="g_res_bottle")(h2, cond)
        u1 = _up_hw(b, self.ch)
        u1 = ResBlock3D(self.ch, name="g_res_up1")(jnp.concatenate([u1, h1], -1), cond)
        return nn.Conv(x.shape[-1], (1, 1, 1), kernel_init=nn.initializers.zeros)(u1)
