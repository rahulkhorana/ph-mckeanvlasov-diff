import jax
import jax.numpy as jnp
from models import time_embed


# ------------------------- Energy bridge -------------------------
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
        # gradient wrt x0_hat; shape matches x0_hat
        return grad_fn(x0_hat)

    return bridge_fn


# ------------------------- Helpers -------------------------
def _flatten_vol(x: jnp.ndarray) -> jnp.ndarray:
    """(B,H,W,K,C) -> (B, D) without copying."""
    if x.ndim == 5:
        B = x.shape[0]
        return x.reshape(B, -1)
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"Expected x with ndim=5 or 2, got {x.shape} (ndim={x.ndim})")


def _mf_rbf_drift(x0_hat: jnp.ndarray, bandwidth: float = 0.5) -> jnp.ndarray:
    """
    Mean-field drift using an RBF kernel between *samples* in the batch.
    Safe for any B >= 1. If B == 1, returns zeros.
    Returns drift with the same shape as x0_hat.
    """
    B = x0_hat.shape[0]
    if B <= 1:
        return jnp.zeros_like(x0_hat)

    X = _flatten_vol(x0_hat)  # (B,D)
    # Pairwise squared distances (numerically safe)
    X2 = jnp.sum(X * X, axis=1, keepdims=True)  # (B,1)
    d2 = jnp.maximum(X2 + X2.T - 2.0 * (X @ X.T), 0.0)  # (B,B)

    h2 = jnp.maximum(float(bandwidth) ** 2, 1e-8)
    K = jnp.exp(-d2 / (2.0 * h2))  # (B,B)
    # remove self-interaction
    K = K * (1.0 - jnp.eye(B, dtype=K.dtype))

    Z = jnp.sum(K, axis=1, keepdims=True)  # (B,1)
    Z = jnp.maximum(Z, 1e-8)
    X_mean = (K @ X) / Z  # (B,D)

    drift_flat = X_mean - X  # (B,D)
    return drift_flat.reshape(x0_hat.shape)  # (B,H,W,K,C)


def _mf_voxel_drift(x0_hat: jnp.ndarray, kernel_size: int = 3) -> jnp.ndarray:
    """
    Simpler mean-field drift: toward the batch mean volume (per-voxel).
    Kernel_size kept for API parity (no-op here unless you later add local smoothing).
    """
    B = x0_hat.shape[0]
    if B <= 1:
        return jnp.zeros_like(x0_hat)
    mean_vol = jnp.mean(x0_hat, axis=0, keepdims=True)  # (1,H,W,K,C)
    return mean_vol - x0_hat  # (B,H,W,K,C)


def _apply_cfg_pred(
    unet_apply, params, x, temb, cond_vec, cfg_scale: float, cond_uncond_vec
):
    """
    Classifier-Free Guidance on the *prediction head* (eps or v).
    If cfg_scale == 0 or cond_uncond_vec is None, returns conditional pred.
    """
    if cfg_scale == 0.0 or cond_uncond_vec is None:
        return unet_apply({"params": params}, x, temb, cond_vec)

    # Two forward passes (conditional + unconditional)
    pred_c = unet_apply({"params": params}, x, temb, cond_vec)
    pred_u = unet_apply({"params": params}, x, temb, cond_uncond_vec)
    return pred_u + cfg_scale * (pred_c - pred_u)


# ------------------------- DDIM sampler -------------------------
def ddim_sample(
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
    bridge_fn=None,
    bridge_scale: float = 0.0,  # energy bridge weight
    cfg_scale: float = 0.0,
    cond_uncond_vec=None,  # (B,cond_dim) for CFG=uncond branch
    return_all: bool = False,
):
    """
    Deterministic DDIM (eta=0) with optional energy bridge and CFG.
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

        pred = _apply_cfg_pred(
            unet_apply, params, x, temb, cond_vec, float(cfg_scale), cond_uncond_vec
        )

        # map to noise prediction
        if v_prediction:
            eps = sqrt_ab * pred + sqrt_1ab * x
        else:
            eps = pred

        # energy bridge on x0_hat
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)
        if bridge_fn is not None and bridge_scale != 0.0:
            g_b = bridge_fn(x, t_cont, x0_hat)
            w_t = (1.0 - a_t) ** 0.5
            eps = eps - float(bridge_scale) * w_t * g_b
            x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # deterministic DDIM update (eta=0)
        x = jnp.sqrt(a_tm1) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x


# ------------------------- MV-SDE sampler -------------------------
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
    # mean-field knobs
    mf_mode: str = "rbf",
    mf_lambda: float = 0.05,
    mf_bandwidth: float = 0.5,
    mf_kernel_size: int = 3,
    # extras
    bridge_fn=None,
    bridge_scale: float = 0.0,
    cfg_scale: float = 0.0,
    cond_uncond_vec=None,
    prob_flow_ode: bool = True,  # deterministic by default
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

        pred = _apply_cfg_pred(
            unet_apply, params, x, temb, cond_vec, float(cfg_scale), cond_uncond_vec
        )

        # map to noise prediction
        if v_prediction:
            eps = sqrt_ab * pred + sqrt_1ab * x
        else:
            eps = pred

        # current x0 estimate
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # -------- mean-field drift on x0_hat (safe for any B) --------
        if mf_mode != "none" and mf_lambda != 0.0:
            if mf_mode == "rbf":
                drift = _mf_rbf_drift(x0_hat, bandwidth=float(mf_bandwidth))
            elif mf_mode == "voxel":
                drift = _mf_voxel_drift(x0_hat, kernel_size=int(mf_kernel_size))
            else:
                raise ValueError(f"Unknown mf_mode: {mf_mode}")

            # schedule the weight (stronger late)
            w_t = (1.0 - a_t) ** 0.5
            # incorporate drift as a correction to eps via x0_hat
            # (similar to energy bridge): eps <- eps - scale * drift_term
            eps = eps - float(mf_lambda) * w_t * drift
            x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # -------- optional energy bridge (usually off for you) --------
        if bridge_fn is not None and bridge_scale != 0.0:
            g_b = bridge_fn(x, t_cont, x0_hat)
            w_t = (1.0 - a_t) ** 0.5
            eps = eps - float(bridge_scale) * w_t * g_b
            x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)

        # -------- update --------
        if prob_flow_ode:
            # deterministic ODE update (DDIM eta=0)
            x = jnp.sqrt(a_tm1) * x0_hat
        else:
            # (Optional) stochastic branch; here just do deterministic step
            x = jnp.sqrt(a_tm1) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x
