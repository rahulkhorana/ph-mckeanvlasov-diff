from __future__ import annotations
from typing import Callable, Optional, Tuple, Literal

import jax
import jax.numpy as jnp
import numpy as np

from models import (
    time_embed as time_embed_fn,
)

_F32 = jnp.float32


# =============================== numerics helpers ===============================


def _nn(x: jnp.ndarray) -> jnp.ndarray:
    """Clamp & de-NaN for stability."""
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.clip(x, -1e6, 1e6).astype(_F32)


def _sched_factor(kind: str, t_idx: int, T: int, strength: float) -> jnp.ndarray:
    """Time schedule in [0,1]; larger late in sampling (low noise)."""
    Tm1 = max(1, T - 1)
    frac = 1.0 - (jnp.array(t_idx, _F32) / jnp.array(Tm1, _F32))  # 0->1
    if kind == "linear":
        f = frac
    elif kind == "cosine":
        f = jnp.sin(0.5 * jnp.pi * frac) ** 2
    elif kind == "exp":
        # convex ramp; 'strength' controls curvature
        k = jnp.array(max(1e-6, strength), _F32)
        f = (jnp.exp(frac * k) - 1.0) / (jnp.exp(k) - 1.0 + 1e-8)
    else:
        f = frac
    return jnp.clip(f, 0.0, 1.0).astype(_F32)


def _linspace_indices(T: int, steps: int) -> jnp.ndarray:
    """T discrete steps [0..T-1] → pick `steps` integers from T-1 down to 0."""
    steps = int(max(1, steps))
    t_float = jnp.linspace(T - 1, 0, steps)
    return jnp.round(t_float).astype(jnp.int32)


def _abar_at(alpha_bars: jnp.ndarray, idx: int) -> jnp.ndarray:
    """ᾱ_{idx}; if idx<0 return 1.0 (by convention for DDIM last jump)."""
    return jnp.where(idx >= 0, alpha_bars[idx], jnp.array(1.0, _F32))


def _to_eps_from_model(
    x_t: jnp.ndarray,
    model_out: jnp.ndarray,
    alpha_bar_t: jnp.ndarray,
    v_prediction: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    From model output (v or eps) return:
      eps_pred, x0_pred, sigma_t  (with sigma_t = sqrt(1-ᾱ_t))
    Relations for v-pred (Imagen/StableDiffusion convention):
      x_t = c0 x0 + c1 eps, v = c0 eps - c1 x0,
      c0 = sqrt(ᾱ_t), c1 = sqrt(1-ᾱ_t)
      => eps = c1*x + c0*v,  x0 = c0*x - c1*v
    """
    c0 = jnp.sqrt(jnp.clip(alpha_bar_t, 1e-8, 1.0))
    c1 = jnp.sqrt(jnp.clip(1.0 - alpha_bar_t, 1e-8, 1.0))
    if v_prediction:
        v = model_out
        eps = c1 * x_t + c0 * v
        x0 = c0 * x_t - c1 * v
    else:
        eps = model_out
        x0 = (x_t - c1 * eps) / jnp.maximum(c0, 1e-8)
    sigma_t = c1
    return _nn(eps), _nn(x0), sigma_t


def _ddim_sigma(
    alpha_bar_t: jnp.ndarray, alpha_bar_prev: jnp.ndarray, eta: float
) -> jnp.ndarray:
    """
    DDIM noise scale σ_t for jump t -> t-1 with stride Δ.
    σ_t = η * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_{t-1})) * sqrt(1-ᾱ_t)
        = η * sqrt( (1-ᾱ_{t-1}) - (1-ᾱ_t) * ᾱ_t/ᾱ_{t-1} )
    """
    eps = jnp.array(1e-8, _F32)
    a_t = jnp.clip(alpha_bar_t, eps, 1.0)
    a_s = jnp.clip(alpha_bar_prev, eps, 1.0)
    # use the standard two-factor form for better numerical stability
    frac = jnp.sqrt(jnp.clip((1.0 - a_s) / (1.0 - a_t), 0.0, 1e8))
    inner = jnp.sqrt(jnp.clip(1.0 - a_t / a_s, 0.0, 1.0))
    return jnp.array(eta, _F32) * frac * inner * jnp.sqrt(jnp.clip(1.0 - a_t, 0.0, 1.0))


# =============================== mean-field coupling ===============================


def _pad_edges(x: jnp.ndarray) -> jnp.ndarray:
    # replicate pad on (H,W,K) dims
    B, H, W, K, C = x.shape
    x0 = x[:, :1, :, :, :]
    x1 = x[:, -1:, :, :, :]
    x = jnp.concatenate([x0, x, x1], axis=1)
    y0 = x[:, :, :1, :, :]
    y1 = x[:, :, -1:, :, :]
    x = jnp.concatenate([y0, x, y1], axis=2)
    z0 = x[:, :, :, :1, :]
    z1 = x[:, :, :, -1:, :]
    x = jnp.concatenate([z0, x, z1], axis=3)
    return x  # (B,H+2,W+2,K+2,C)


def _neighbors6(x: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    """Return the 6 direct neighbors (±x, ±y, ±z) with edge replication."""
    B, H, W, K, C = x.shape
    p = _pad_edges(x)
    # shifts: (±1,0,0), (0,±1,0), (0,0,±1) relative to padded volume
    xp = p[:, 2:, 1:-1, 1:-1, :]  # +x
    xm = p[:, :-2, 1:-1, 1:-1, :]  # -x
    yp = p[:, 1:-1, 2:, 1:-1, :]  # +y
    ym = p[:, 1:-1, :-2, 1:-1, :]  # -y
    zp = p[:, 1:-1, 1:-1, 2:, :]  # +z
    zm = p[:, 1:-1, 1:-1, :-2, :]  # -z
    return xp, xm, yp, ym, zp, zm


def _mf_voxel(x: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """Laplacian-like smoothing gradient: sum(nei - 6*x)."""
    xp, xm, yp, ym, zp, zm = _neighbors6(x)
    lap = (xp + xm + yp + ym + zp + zm) - 6.0 * x
    # scale by bandwidth (acts like step size)
    return _nn(lap * jnp.array(bandwidth, _F32))


def _mf_rbf(x: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """
    Bilateral-like local interaction: Σ_nei exp(-||δ||^2 / (2 h^2)) * δ,
    δ = nei - x. This is ∇ of a local RBF-kernel energy → smoothing.
    """
    h2 = jnp.array(max(1e-6, bandwidth) ** 2, _F32)
    xp, xm, yp, ym, zp, zm = _neighbors6(x)
    out = jnp.zeros_like(x)
    for nei in (xp, xm, yp, ym, zp, zm):
        d = nei - x
        # channel-wise squared norm
        d2 = jnp.sum(d * d, axis=-1, keepdims=True)
        w = jnp.exp(-d2 / (2.0 * h2))
        out = out + w * d
    return _nn(out)


def _mean_field_grad(x: jnp.ndarray, mode: str, bandwidth: float) -> jnp.ndarray:
    if mode == "voxel":
        return _mf_voxel(x, bandwidth)
    elif mode == "rbf":
        return _mf_rbf(x, bandwidth)
    else:
        return jnp.zeros_like(x)


# =============================== energy guidance ===============================


def make_energy_guidance(
    E_apply: Callable,  # E({"params": p}, x, cond) -> (B,) energy
    eparams,
    cond_vec: jnp.ndarray,  # (B, D)
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a function grad_fn(x) = ∇_x ( -E(x, cond) ), batched over B.
    """

    def energy_single(x_i: jnp.ndarray, cond_i: jnp.ndarray) -> jnp.ndarray:
        # x_i: (H,W,K,C), cond_i: (D,)
        e = E_apply({"params": eparams}, x_i[None, ...], cond_i[None, ...]).reshape(())
        return -e  # ascend direction (adds to score)

    v_energy = jax.vmap(jax.grad(energy_single), in_axes=(0, 0))

    def grad_fn(x: jnp.ndarray) -> jnp.ndarray:
        return _nn(v_energy(x, cond_vec))

    return grad_fn


# =============================== main sampler ===============================


def mv_sde_sample(
    *,
    unet_apply: Callable,  # f({"params": p}, x, t_emb, cond) -> (B,H,W,K,C) output (v or eps)
    params,
    shape: Tuple[int, int, int, int, int],  # (B,H,W,K,C)
    betas: jnp.ndarray,  # (T,)
    alphas: jnp.ndarray,  # (T,)
    alpha_bars: jnp.ndarray,  # (T,)
    cond_vec: jnp.ndarray,  # (B,D)
    steps: int,
    rng: jnp.ndarray,
    v_prediction: bool,
    # CFG
    cfg_scale: float = 0.0,
    null_cond_vec: Optional[jnp.ndarray] = None,
    cfg_schedule: str = "cosine",
    cfg_strength: float = 5.0,
    # Energy guidance (score-like)
    guidance_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    guidance_scale: float = 0.0,
    guidance_schedule: str = "cosine",
    guidance_strength: float = 3.0,
    # Mean-field coupling
    mf_mode: Literal["none", "voxel", "rbf"] = "none",
    mf_lambda: float = 0.0,
    mf_bandwidth: float = 0.5,
    # Dynamics
    prob_flow_ode: bool = False,  # True → deterministic DDIM (η=0), False → MV-SDE (η>0)
    return_all: bool = False,
) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deterministic DDIM and MV-SDE sampler with mean-field coupling and classifier-free guidance.

    Score construction:
      - If v-pred: eps = c1*x + c0*v, score = -eps/sigma
      - If eps-pred: eps = model_out, score = -eps/sigma
      - Mean-field gradient g(x) (local neighbor-based) is added in score space.
      - Optional energy guidance h(x) = ∇_x(-E) added in score space.

    Update (DDIM form):
      x0 = (x - sigma_t * eps_total) / sqrt(ᾱ_t)
      σ_ddim = η * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1}) * sqrt(1-ᾱ_t)
      x_{t-1} = sqrt(ᾱ_{t-1}) * x0 + sqrt(1-ᾱ_{t-1} - σ_ddim^2) * eps_total + σ_ddim * z

    Where η=0 → deterministic (prob-flow ODE-ish), η>0 gives SDE-like sampling.
    """
    B, H, W, K, C = shape
    T = int(alpha_bars.shape[0])

    # choose η (noise strength)
    eta = 0.0 if prob_flow_ode else 1.0

    # timestep schedule (descending)
    ts = _linspace_indices(T, int(steps))

    # handy for time embedding: use raw index; your time_embed handles scaling internally
    def t_emb_for(idx: int) -> jnp.ndarray:
        return time_embed_fn(jnp.full((B,), float(idx), dtype=_F32), 128)

    # init x_T ~ N(0,I)
    rng, k = jax.random.split(rng)
    x = jax.random.normal(k, shape, dtype=_F32)

    traj = []
    if return_all:
        traj.append(x)

    # precompute null conditioning
    has_null = (
        (null_cond_vec is not None) and (cfg_scale is not None) and (cfg_scale > 0.0)
    )
    if not has_null:
        null_cond_vec = jnp.zeros_like(cond_vec)

    for si in range(int(ts.shape[0])):
        t_idx = int(ts[si])
        t_emb = t_emb_for(t_idx)

        # model evals (conditional & unconditional)
        out_c = unet_apply({"params": params}, x, t_emb, cond_vec)
        if cfg_scale > 0.0:
            out_u = unet_apply({"params": params}, x, t_emb, null_cond_vec)
            w = jnp.array(cfg_scale, _F32) * _sched_factor(
                cfg_schedule, t_idx, T, cfg_strength
            )
            model_out = out_u + w * (out_c - out_u)
        else:
            model_out = out_c

        # map to eps/x0 and build score
        a_bar_t = alpha_bars[t_idx]
        eps_pred, x0_pred, sigma_t = _to_eps_from_model(
            x, model_out, a_bar_t, v_prediction
        )

        # base score from model
        score = -eps_pred / jnp.maximum(sigma_t, 1e-8)

        # mean-field coupling (added in score space)
        if mf_lambda != 0.0 and mf_mode != "none":
            g = _mean_field_grad(x, mf_mode, mf_bandwidth)
            score = score + jnp.array(mf_lambda, _F32) * g

        # energy guidance (added in score space)
        if guidance_fn is not None and guidance_scale > 0.0:
            sE = jnp.array(guidance_scale, _F32) * _sched_factor(
                guidance_schedule, t_idx, T, guidance_strength
            )
            gE = guidance_fn(x)
            score = score + sE * gE

        # convert final score back to eps for DDIM update
        eps_total = -jnp.maximum(sigma_t, 1e-8) * score
        eps_total = _nn(eps_total)

        # next ᾱ
        t_prev = int(ts[si + 1]) if (si + 1) < ts.shape[0] else -1
        a_bar_prev = _abar_at(alpha_bars, t_prev)

        # DDIM stochasticity (η); η=0 → ODE-like
        sigma_ddim = _ddim_sigma(a_bar_t, a_bar_prev, eta)
        # deterministic coefficient on eps
        coeff_eps = jnp.sqrt(jnp.clip(1.0 - a_bar_prev - sigma_ddim**2, 0.0, 1.0))
        # predicted clean x0 from final eps
        x0 = (x - jnp.sqrt(jnp.clip(1.0 - a_bar_t, 0.0, 1.0)) * eps_total) / jnp.sqrt(
            jnp.clip(a_bar_t, 1e-8, 1.0)
        )

        # noise
        rng, k = jax.random.split(rng)
        z = jax.random.normal(k, x.shape, dtype=_F32)

        # update
        x = (
            jnp.sqrt(jnp.clip(a_bar_prev, 0.0, 1.0)) * x0
            + coeff_eps * eps_total
            + sigma_ddim * z
        )
        x = _nn(x)

        if return_all:
            traj.append(x)

    if return_all:
        return x, jnp.stack(traj, axis=1)  # (B, S+1, H, W, K, C)
    return x
