# sampling.py — DDIM + MV-SDE sampling with sequential-CFG and safe MF drifts
import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Callable

from models import time_embed


# ----------------------------- Module bridge -----------------------------
def make_module_bridge(E_apply, eparams, cond_vec, T: int):
    """
    cond_vec is *fixed* for sampling (B, cond_dim).
    Returns bridge_fn(x_t, t_cont, x0_hat) = ∇_{x0} E(x0, cond_vec).
    """

    def energy_mean(x0):
        e = E_apply({"params": eparams}, x0, cond_vec)  # (B,)
        return jnp.mean(e)

    grad_fn = jax.grad(energy_mean)

    def bridge_fn(x_t, t_cont, x0_hat):
        return grad_fn(x0_hat)

    return bridge_fn


# ----------------------------- Helpers -----------------------------------
def _cfg_blend(
    pred_cond: jnp.ndarray, pred_uncond: jnp.ndarray, scale: float
) -> jnp.ndarray:
    """
    Classic SD-style CFG on the *prediction* (eps or v):
      pred = pred_u + s * (pred_c - pred_u)
    """
    return pred_uncond + scale * (pred_cond - pred_uncond)


def _safe_int(x, lo=1, hi=255):
    try:
        xi = int(x)
    except Exception:
        xi = lo
    return max(lo, min(hi, xi))


# -------------------- Mean-field drift terms (MV-SDE) --------------------
def _mf_rbf_drift(x0_hat: jnp.ndarray, bandwidth: float = 0.5) -> jnp.ndarray:
    """
    Pairwise RBF smoothing across the *batch*.
    x0_hat: (B,H,W,K,C)
    Returns drift with same shape.
    """
    B = x0_hat.shape[0]
    x = x0_hat.reshape(B, -1)  # (B, M)
    # L2 distances in feature (voxel) space
    x2 = jnp.sum(x * x, axis=1, keepdims=True)  # (B,1)
    d2 = x2 + x2.T - 2.0 * (x @ x.T)  # (B,B)
    Kmat = jnp.exp(-d2 / (2.0 * (bandwidth**2) + 1e-6))  # (B,B)
    # Normalize row-wise
    Z = jnp.clip(jnp.sum(Kmat, axis=1, keepdims=True), 1e-6)
    W = Kmat / Z
    # Batch-mean field
    x_mean = W @ x  # (B,M)
    mean_field = x_mean.reshape(x0_hat.shape)
    return mean_field - x0_hat


def _depthwise_box_blur_hw(nhwc: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    nhwc: (B,H,W,C')  — blur over H/W ONLY with SAME pad, depthwise (groups=C').
    Returns same shape.
    """
    pad = k // 2
    Cprime = nhwc.shape[-1]
    # Kernel: (k, k, in_channels_per_group=1, out_channels=Cprime)
    w = jnp.ones((k, k, 1, Cprime), nhwc.dtype) / float(k * k)
    y = lax.conv_general_dilated(
        lhs=nhwc,
        rhs=w,
        window_strides=(1, 1),
        padding=[(pad, pad), (pad, pad)],  # SAME on H/W
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=Cprime,  # depthwise
    )
    return y


def _mf_voxel_drift(x0_hat: jnp.ndarray, kernel_size: int = 3) -> jnp.ndarray:
    """
    Local (per-sample) smoothing over H/W, *no* mixing across K,
    preserves shape exactly. Safe for tiny batches.
    x0_hat: (B,H,W,K,C)
    """
    B, H, W, K, C = x0_hat.shape
    k = _safe_int(kernel_size, lo=1)
    if (k % 2) == 0:
        k += 1  # force odd for symmetric SAME padding

    x_flat = x0_hat.reshape(B, H, W, K * C)  # NHWC with C' = K*C
    smooth_flat = _depthwise_box_blur_hw(x_flat, k)  # (B,H,W,K*C)
    smooth = smooth_flat.reshape(B, H, W, K, C)
    return smooth - x0_hat


# ----------------------------- DDIM sampler ------------------------------
def ddim_sample(
    unet_apply: Callable,
    params,
    shape,
    betas: jnp.ndarray,
    alphas: jnp.ndarray,
    alpha_bars: jnp.ndarray,
    cond_vec: jnp.ndarray,  # (B,D)
    steps: int = 50,
    rng=None,
    v_prediction: bool = True,
    bridge_fn: Optional[Callable] = None,  # bridge(x_t, t_cont, x0_hat) -> grad wrt x0
    bridge_scale: float = 0.0,
    cfg_scale: float = 0.0,
    cond_uncond_vec: Optional[jnp.ndarray] = None,  # (B,D) or None
    return_all: bool = False,
):
    """
    Deterministic DDIM (eta=0) with optional bridge + sequential CFG.
    """
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).round().astype(jnp.int32)
    key = jax.random.PRNGKey(0) if rng is None else rng
    x = jax.random.normal(key, shape)
    traj = []

    for t in ts:
        a_t = alpha_bars[t]
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)]
        sqrt_ab = jnp.sqrt(a_t)
        sqrt_1ab = jnp.sqrt(1.0 - a_t)
        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)
        temb = time_embed(jnp.full((shape[0],), t_cont), dim=128)

        # sequential CFG on prediction
        pred_c = unet_apply({"params": params}, x, temb, cond_vec)  # eps or v
        if cfg_scale != 0.0 and cond_uncond_vec is not None:
            pred_u = unet_apply({"params": params}, x, temb, cond_uncond_vec)
            pred = _cfg_blend(pred_c, pred_u, cfg_scale)
        else:
            pred = pred_c

        if v_prediction:
            eps = sqrt_ab * pred + sqrt_1ab * x
        else:
            eps = pred

        # x0 estimate
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # module bridge on x0_hat
        if bridge_fn is not None and bridge_scale != 0.0:
            g_b = bridge_fn(x, t_cont, x0_hat)
            w_t = jnp.sqrt(1.0 - a_t)  # stronger away from data manifold
            eps = eps - bridge_scale * w_t * g_b
            # refresh x0_hat after guidance
            x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # DDIM update (eta=0)
        x = jnp.sqrt(a_tm1) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x


# ----------------------------- MV-SDE sampler ----------------------------
def mv_sde_sample(
    unet_apply: Callable,
    params,
    shape,
    betas: jnp.ndarray,
    alphas: jnp.ndarray,
    alpha_bars: jnp.ndarray,
    cond_vec: jnp.ndarray,  # (B,D)
    steps: int = 50,
    rng=None,
    v_prediction: bool = True,
    mf_mode: str = "voxel",  # "none" | "rbf" | "voxel"
    mf_lambda: float = 0.05,
    mf_bandwidth: float = 0.5,
    mf_kernel_size: int = 3,
    bridge_fn: Optional[Callable] = None,
    bridge_scale: float = 0.0,
    cfg_scale: float = 0.0,
    cond_uncond_vec: Optional[jnp.ndarray] = None,  # (B,D) or None
    prob_flow_ode: bool = True,  # deterministic path recommended
    return_all: bool = False,
):
    """
    Mean-field SDE/ODE sampler.
    - prob_flow_ode=True: deterministic probability-flow ODE (stable).
    - mf_mode:
        * "none" : no mean-field
        * "rbf"  : batchwise RBF field
        * "voxel": per-sample local HW smoothing (SAFE for tiny batch)
    """
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).round().astype(jnp.int32)
    key = jax.random.PRNGKey(0) if rng is None else rng
    x = jax.random.normal(key, shape)
    traj = []

    for t in ts:
        a_t = alpha_bars[t]
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)]
        sqrt_ab = jnp.sqrt(a_t)
        sqrt_1ab = jnp.sqrt(1.0 - a_t)
        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)
        temb = time_embed(jnp.full((shape[0],), t_cont), dim=128)

        # sequential CFG on prediction
        pred_c = unet_apply({"params": params}, x, temb, cond_vec)
        if cfg_scale != 0.0 and cond_uncond_vec is not None:
            pred_u = unet_apply({"params": params}, x, temb, cond_uncond_vec)
            pred = _cfg_blend(pred_c, pred_u, cfg_scale)
        else:
            pred = pred_c

        if v_prediction:
            eps = sqrt_ab * pred + sqrt_1ab * x
        else:
            eps = pred

        # x0 estimate
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # module bridge (x0_hat)
        if bridge_fn is not None and bridge_scale != 0.0:
            g_b = bridge_fn(x, t_cont, x0_hat)
            w_t = jnp.sqrt(1.0 - a_t)
            eps = eps - bridge_scale * w_t * g_b
            x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # mean-field drift
        if mf_mode == "rbf":
            drift = _mf_rbf_drift(x0_hat, bandwidth=float(mf_bandwidth))
        elif mf_mode == "voxel":
            drift = _mf_voxel_drift(x0_hat, kernel_size=int(mf_kernel_size))
        else:
            drift = jnp.zeros_like(x0_hat)

        # combine
        drift = mf_lambda * drift

        if prob_flow_ode:
            # probability-flow ODE (deterministic)
            x = jnp.sqrt(a_tm1) * (x0_hat + drift)
        else:
            # Euler-Maruyama step (stochastic)
            dt = 1.0 / float(steps)
            key, sub = jax.random.split(key)
            z = jax.random.normal(sub, x.shape)
            # standard DDIM-like deterministic part
            x_det = jnp.sqrt(a_tm1) * x0_hat
            # add MF drift + small noise
            x = x_det + drift * dt + jnp.sqrt(dt) * z * jnp.sqrt(1.0 - a_tm1)

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x
