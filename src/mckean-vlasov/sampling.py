# sampling.py
import jax, jax.numpy as jnp
from typing import Optional, Literal
from models import time_embed


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


# ------------------------------ helpers ------------------------------


def _time_embed_from_index(t_idx: jnp.ndarray, T: int, dim: int = 128) -> jnp.ndarray:
    # t_idx: (B,) int32 in [0, T-1]
    t_cont = (t_idx.astype(jnp.float32) + 0.5) / float(T)
    return time_embed(t_cont, dim=dim)


def _predict_eps(
    unet_apply, params, x, a_bar_t, t_idx, T, cond_vec, v_prediction: bool
):
    """
    Returns ε̂ at time t given model output head type.
    """
    temb = _time_embed_from_index(jnp.full((x.shape[0],), t_idx, dtype=jnp.int32), T)
    pred = unet_apply({"params": params}, x, temb, cond_vec)  # ε̂ or v̂ depending on head
    if v_prediction:
        sqrt_ab = jnp.sqrt(a_bar_t)
        sqrt_1ab = jnp.sqrt(1.0 - a_bar_t)
        eps_hat = sqrt_ab * pred + sqrt_1ab * x
    else:
        eps_hat = pred
    return eps_hat  # (B,H,W,K,C)


def _cfg_blend(
    unet_apply,
    params,
    x,
    a_bar_t,
    t_idx,
    T,
    cond_vec,
    cond_uncond_vec,
    v_prediction: bool,
    cfg_scale: float,
):
    """
    Classifier-free guidance on ε̂. If cond_uncond_vec is None or cfg_scale==0, returns conditional only.
    """
    eps_c = _predict_eps(
        unet_apply, params, x, a_bar_t, t_idx, T, cond_vec, v_prediction
    )
    if (cond_uncond_vec is None) or (cfg_scale == 0.0):
        return eps_c
    eps_u = _predict_eps(
        unet_apply, params, x, a_bar_t, t_idx, T, cond_uncond_vec, v_prediction
    )
    return eps_u + cfg_scale * (eps_c - eps_u)


def _x0_from_xt_eps(x, a_bar_t, eps_hat):
    sqrt_ab = jnp.sqrt(a_bar_t)
    sqrt_1ab = jnp.sqrt(1.0 - a_bar_t)
    return (x - sqrt_1ab * eps_hat) / jnp.clip(sqrt_ab, 1e-8)


def _mf_voxel_drift(x0_hat, kernel_size: int = 3) -> jnp.ndarray:
    """
    Local mean-field: drift toward blurred field ~ x0_hat - blur(x0_hat).
    Uses box blur via depthwise conv (uniform kernel).
    """
    assert kernel_size % 2 == 1, "kernel must be odd"
    B, H, W, K, C = x0_hat.shape
    k = kernel_size
    # Depthwise 3D box filter per-channel.
    filt = jnp.ones((k, k, k), dtype=x0_hat.dtype) / float(k * k * k)
    filt = filt[:, :, :, None, None]  # (k,k,k,1,1) depthwise
    # Reshape to (B,H,W,K,C) -> NHCWC (same)
    y = jax.lax.conv_general_dilated(
        x0_hat,
        filt,
        window_strides=(1, 1, 1),
        padding=[(k // 2, k // 2)] * 3,
        dimension_numbers=("NHWDC", "DHWIO", "NHWDC"),
        feature_group_count=C,  # depthwise per-channel
    )
    return x0_hat - y  # push away from local mean


def _mf_rbf_drift(x0_hat, bandwidth: float = 0.5) -> jnp.ndarray:
    """
    Batch mean-field: for each sample i, drift toward kernel-weighted batch mean.
    Embedding = global mean per sample (simple, light).
    """
    B = x0_hat.shape[0]
    feat = jnp.mean(x0_hat, axis=(1, 2, 3, 4))  # (B,) simple scalar (one per sample)
    feat = feat[:, None]  # (B,1)
    # pairwise squared distances
    diff = feat - feat.T  # (B,B,1)
    d2 = jnp.sum(diff * diff, axis=-1)  # (B,B)
    s2 = jnp.maximum(bandwidth * bandwidth, 1e-6)
    Kmat = jnp.exp(-d2 / (2.0 * s2))  # (B,B)
    # normalize rows
    Z = jnp.clip(jnp.sum(Kmat, axis=1, keepdims=True), 1e-6)
    W = Kmat / Z  # (B,B)
    # weighted batch mean in data space
    xm = jnp.tensordot(W, x0_hat, axes=(1, 0))  # (B,H,W,K,C)
    return (
        x0_hat - xm
    )  # pull toward / away from batch mean (repulsive if subtracted from eps)


# ------------------------------ samplers ------------------------------


def ddim_sample(
    unet_apply,
    params,
    shape,
    betas,
    alphas,
    alpha_bars,
    cond_vec,  # (B,cond_dim), fixed during sampling
    steps: int = 50,
    rng=None,
    v_prediction: bool = True,
    bridge_fn=None,
    bridge_scale: float = 0.0,  # energy bridge on x0_hat
    cfg_scale: float = 0.0,  # CFG
    cond_uncond_vec: Optional[jnp.ndarray] = None,
    return_all: bool = False,
):
    """
    Deterministic DDIM (η=0) with optional CFG and energy bridge.
    """
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).round().astype(jnp.int32)

    key = jax.random.PRNGKey(0) if rng is None else rng
    x = jax.random.normal(key, shape)
    traj = []

    for t in ts:
        a_t = alpha_bars[t]
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)]

        # ε̂ via (optionally) CFG
        eps_hat = _cfg_blend(
            unet_apply,
            params,
            x,
            a_t,
            int(t),
            T,
            cond_vec,
            cond_uncond_vec,
            v_prediction,
            float(cfg_scale),
        )

        # energy bridge at x0_hat
        x0_hat = _x0_from_xt_eps(x, a_t, eps_hat)
        if (bridge_fn is not None) and (bridge_scale != 0.0):
            g_b = bridge_fn(x, (float(t) + 0.5) / float(T), x0_hat)
            # weight larger near data (same schedule you had)
            w_t = jnp.sqrt(jnp.maximum(1.0 - a_t, 0.0))
            eps_hat = eps_hat - float(bridge_scale) * w_t * g_b

        # update (DDIM, η=0)
        x0_hat = _x0_from_xt_eps(x, a_t, eps_hat)
        x = jnp.sqrt(a_tm1) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x


def mv_sde_sample(
    unet_apply,
    params,
    shape,
    betas,
    alphas,
    alpha_bars,
    cond_vec,  # (B,cond_dim)
    steps: int = 50,
    rng=None,
    v_prediction: bool = True,
    mf_mode: Literal["none", "rbf", "voxel"] = "none",
    mf_lambda: float = 0.0,
    mf_bandwidth: float = 0.5,
    mf_kernel_size: int = 3,
    bridge_fn=None,
    bridge_scale: float = 0.0,
    cfg_scale: float = 0.0,
    cond_uncond_vec: Optional[jnp.ndarray] = None,
    prob_flow_ode: bool = True,  # True = deterministic (DDIM-like), False = stochastic
    return_all: bool = False,
):
    """
    MV-SDE with optional mean-field coupling + CFG + energy bridge.
    - prob_flow_ode=True: deterministic probability-flow ODE (like DDIM η=0).
    - prob_flow_ode=False: stochastic SDE (adds noise).
    Mean-field is applied in x0_hat space and reflected back onto ε̂.
    """
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).round().astype(jnp.int32)

    key = jax.random.PRNGKey(0) if rng is None else rng
    x = jax.random.normal(key, shape)
    traj = []

    for t in ts:
        a_t = alpha_bars[t]
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)]
        sqrt_1ab = jnp.sqrt(jnp.maximum(1.0 - a_t, 0.0))

        # ε̂ (CFG)
        eps_hat = _cfg_blend(
            unet_apply,
            params,
            x,
            a_t,
            int(t),
            T,
            cond_vec,
            cond_uncond_vec,
            v_prediction,
            float(cfg_scale),
        )

        # x0_hat
        x0_hat = _x0_from_xt_eps(x, a_t, eps_hat)

        # energy bridge (on x0_hat)
        if (bridge_fn is not None) and (bridge_scale != 0.0):
            g_b = bridge_fn(x, (float(t) + 0.5) / float(T), x0_hat)
            w_t = sqrt_1ab
            eps_hat = eps_hat - float(bridge_scale) * w_t * g_b
            x0_hat = _x0_from_xt_eps(x, a_t, eps_hat)  # refresh

        # mean-field coupling (reflected on ε̂)
        if (mf_mode != "none") and (mf_lambda != 0.0):
            if mf_mode == "voxel":
                drift = _mf_voxel_drift(x0_hat, kernel_size=int(mf_kernel_size))
            elif mf_mode == "rbf":
                drift = _mf_rbf_drift(x0_hat, bandwidth=float(mf_bandwidth))
            else:
                drift = 0.0
            # map drift in x0 space to ε̂ perturbation:
            # x0_hat = (x - sqrt(1-a_t) ε̂)/sqrt(a_t)  =>  ε̂ = (x - sqrt(a_t) x0_hat)/sqrt(1-a_t)
            # small change Δx0 induces Δε̂ ≈ -sqrt(a_t)/sqrt(1-a_t) * Δx0
            scale = -jnp.sqrt(jnp.maximum(a_t, 1e-8)) / jnp.clip(sqrt_1ab, 1e-8)
            eps_hat = eps_hat + float(mf_lambda) * scale * drift
            x0_hat = _x0_from_xt_eps(x, a_t, eps_hat)  # refresh

        # step
        if prob_flow_ode:
            # deterministic flow (DDIM-like, η=0)
            x = jnp.sqrt(a_tm1) * x0_hat
        else:
            # stochastic DDPM-like step
            key, key_n = jax.random.split(key)
            beta_t = betas[t]
            # variance preserving step (one of many valid discretizations)
            sigma_t = jnp.sqrt(jnp.maximum(beta_t, 1e-8))
            mean_xtm1 = (
                jnp.sqrt(a_tm1) * x0_hat
                + jnp.sqrt(jnp.maximum(1.0 - a_tm1 - (1.0 - a_t), 0.0)) * eps_hat * 0.0
            )
            # Above "mean_xtm1" uses classic DDIM mean; you can also use VP-DDPM mean:
            # mean = sqrt(α_t) * x - ( (1-α_t) / sqrt(1-ᾱ_t) ) * ε̂
            # but DDIM-style keeps consistency with prob_flow branch.
            noise = jax.random.normal(key_n, shape)
            x = mean_xtm1 + sigma_t * noise

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x
