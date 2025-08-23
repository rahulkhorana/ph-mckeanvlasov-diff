# sampling.py — DDIM and MV-SDE (prob-flow) with proper CFG; no per-step clipping
import jax, jax.numpy as jnp
from models import time_embed


# ---------- helpers ----------
def _cfg_eps(unet_apply, params, x, temb, cond_vec, cfg_scale: float, cond_uncond_vec):
    if cond_uncond_vec is None or cfg_scale == 0.0:
        return unet_apply({"params": params}, x, temb, cond_vec)
    eps_u = unet_apply({"params": params}, x, temb, cond_uncond_vec)
    eps_c = unet_apply({"params": params}, x, temb, cond_vec)
    return eps_u + cfg_scale * (eps_c - eps_u)


# Optional mean-field drifts (start with none)
def _mf_none(*args, **kw):
    return 0.0


def _mf_rbf_drift(x0_hat, bandwidth: float = 0.5):
    """
    Weak smoothing drift in x0 space.
    x0_hat: (B,H,W,K,C)
    returns same shape drift (small)
    """
    # Depthwise 3D Gaussian approx via separable 1D kernels over H,W only (keep K)
    B, H, W, K, C = x0_hat.shape
    # simple 1D blur kernel of length 3
    k = jnp.array([0.25, 0.5, 0.25], jnp.float32)

    def blur2d(img):
        # conv over H then W; padding SAME via concat
        pad_h = jnp.pad(img, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))
        yh = k[0] * pad_h[:, :-2] + k[1] * pad_h[:, 1:-1] + k[2] * pad_h[:, 2:]
        pad_w = jnp.pad(yh, ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)))
        yw = k[0] * pad_w[:, :, :-2] + k[1] * pad_w[:, :, 1:-1] + k[2] * pad_w[:, :, 2:]
        return yw

    smooth = blur2d(x0_hat)
    return (smooth - x0_hat) * float(bandwidth)


def _choose_mf(mode: str):
    if mode == "rbf":
        return _mf_rbf_drift
    else:
        return _mf_none


# ---------- DDIM (ε-pred) ----------
def ddim_sample(
    unet_apply,
    params,
    shape,
    betas,
    alphas,
    alpha_bars,
    cond_vec,
    steps: int = 50,
    rng=None,
    cfg_scale: float = 0.0,
    cond_uncond_vec=None,
    return_all: bool = False,
):
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

        eps = _cfg_eps(
            unet_apply, params, x, temb, cond_vec, cfg_scale, cond_uncond_vec
        )
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)
        x = jnp.sqrt(a_tm1) * x0_hat  # deterministic (eta=0)

        if return_all:
            traj.append(x)
    return jnp.stack(traj, 1) if return_all else x


# ---------- MV-SDE (prob-flow ODE branch) ----------
def mv_sde_sample(
    unet_apply,
    params,
    shape,
    betas,
    alphas,
    alpha_bars,
    cond_vec,
    steps: int = 50,
    rng=None,
    cfg_scale: float = 0.0,
    cond_uncond_vec=None,
    mf_mode: str = "none",
    mf_lambda: float = 0.0,
    mf_bandwidth: float = 0.5,
    return_all: bool = False,
    prob_flow_ode: bool = True,  # deterministic path
):
    # For stability and speed we implement the prob-flow ODE equivalent, deterministic.
    # If you ever want the stochastic branch, you can add noise_t terms with dt scheduling.
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).round().astype(jnp.int32)
    key = jax.random.PRNGKey(0) if rng is None else rng
    x = jax.random.normal(key, shape)
    traj = []

    mf_fn = _choose_mf(mf_mode)

    for t in ts:
        a_t = alpha_bars[t]
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)]
        sqrt_ab = jnp.sqrt(a_t)
        sqrt_1ab = jnp.sqrt(1.0 - a_t)
        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)
        temb = time_embed(jnp.full((shape[0],), t_cont), dim=128)

        eps = _cfg_eps(
            unet_apply, params, x, temb, cond_vec, cfg_scale, cond_uncond_vec
        )
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # mean-field drift (very small; start with none)
        if mf_mode != "none" and mf_lambda != 0.0:
            drift = mf_fn(x0_hat, bandwidth=float(mf_bandwidth))
            x0_hat = x0_hat + float(mf_lambda) * drift

        # prob-flow deterministic update
        x = jnp.sqrt(a_tm1) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x
