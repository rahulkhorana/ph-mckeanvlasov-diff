import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Tuple, Literal

from models import time_embed


# -------------------------------------------------
# Energy guidance (modules+label already in cond_vec)
# -------------------------------------------------
def make_energy_guidance(
    E_apply: Callable,
    eparams,
    cond_vec: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Returns guidance(x, t_cont) -> ∇_x mean E(x, cond_vec).
    cond_vec is fixed for a sampling call: shape (B, D).
    """

    def e_mean(x: jnp.ndarray) -> jnp.ndarray:
        e = E_apply({"params": eparams}, x, cond_vec)  # (B,)
        return jnp.mean(e)

    grad_fn = jax.grad(e_mean)

    def guidance(x: jnp.ndarray, t_cont: jnp.ndarray) -> jnp.ndarray:
        # t_cont kept for API symmetry; you can time-gate guidance outside if desired.
        return grad_fn(x)

    return guidance


# -------------------------------------------------
# Small helpers
# -------------------------------------------------
def _v_to_eps(v_pred: jnp.ndarray, x: jnp.ndarray, a_bar: jnp.ndarray) -> jnp.ndarray:
    """
    Convert v-prediction to eps_hat.
      eps_hat = sqrt(a_bar) * v + sqrt(1 - a_bar) * x
    Shapes:
      v_pred, x: (B,H,W,K,C)
      a_bar:     (B,1,1,1,1)
    """
    sqrt_ab = jnp.sqrt(a_bar)
    sqrt_1ab = jnp.sqrt(jnp.clip(1.0 - a_bar, 1e-8, None))
    return sqrt_ab * v_pred + sqrt_1ab * x


def _score_from_eps(eps_hat: jnp.ndarray, a_bar: jnp.ndarray) -> jnp.ndarray:
    """
    VP score: s_theta(x,t) ≈ - eps_hat / sigma_t,  sigma_t = sqrt(1 - a_bar).
    """
    sigma_t = jnp.sqrt(jnp.clip(1.0 - a_bar, 1e-8, None))
    return -eps_hat / sigma_t


def _time_weight(
    t_cont: jnp.ndarray, kind: Literal["linear", "cosine", "exp"], strength: float
) -> jnp.ndarray:
    """
    Returns a scalar weight per batch element in [0, 1] to anneal guidance/CFG over time.
    t_cont in [0,1], 1 = early, 0 = final.
    """
    t = jnp.clip(t_cont, 0.0, 1.0)
    if kind == "linear":
        w = t  # strong early, fades to 0
    elif kind == "cosine":
        w = 0.5 * (1.0 + jnp.cos(jnp.pi * (1.0 - t)))  # smooth
    else:  # "exp"
        w = jnp.exp(-strength * (1.0 - t))
    return jnp.clip(w, 0.0, 1.0)


# -------------------------------------------------
# Mean-field interactions (McKean–Vlasov)
# -------------------------------------------------
def _mf_voxel_mean(x: jnp.ndarray, lam: float) -> jnp.ndarray:
    """
    Voxelwise mean-field repulsion:
      F_mf = lam * (x - mean_over_batch(x))
    This pushes each sample away from the batch mean at every voxel.
    """
    if lam <= 0.0:
        return jnp.zeros_like(x)
    mu = jnp.mean(x, axis=0, keepdims=True)  # (1,H,W,K,C)
    return lam * (x - mu)


def _mf_global_rbf(x: jnp.ndarray, lam: float, h: float) -> jnp.ndarray:
    """
    Global RBF repulsion in a compact embedding:
      - Summarize each sample by φ = mean over (H,W,K) -> (B,C).
      - Repel using RBF kernel in that C-dim space, broadcast to a bias field.
    """
    if lam <= 0.0 or h <= 0.0:
        return jnp.zeros_like(x)

    B = x.shape[0]
    phi = jnp.mean(x, axis=(1, 2, 3))  # (B,C)
    diff = phi[:, None, :] - phi[None, :, :]  # (B,B,C)
    d2 = jnp.sum(diff * diff, axis=-1)  # (B,B)
    h2 = jnp.clip(h * h, 1e-8, None)
    K = jnp.exp(-d2 / h2)  # (B,B)
    rep_phi = (K[..., None] * (-diff)).sum(axis=1) / h2  # (B,C)
    rep_phi = rep_phi / max(B, 1)
    rep_field = rep_phi[:, None, None, None, :]  # (B,1,1,1,C)
    return lam * rep_field


# -------------------------------------------------
# UNet → eps helper (handles v-pred)
# -------------------------------------------------
def _eps_hat_from_unet(
    unet_apply: Callable,
    params,
    x: jnp.ndarray,
    t_cont: jnp.ndarray,  # (B,)
    cond_vec: jnp.ndarray,  # (B,D)
    a_bar_b: jnp.ndarray,  # (B,1,1,1,1)
    v_prediction: bool,
) -> jnp.ndarray:
    temb = time_embed(t_cont, dim=128)  # (B,128)
    pred = unet_apply({"params": params}, x, temb, cond_vec)  # (B,H,W,K,C)
    if v_prediction:
        return _v_to_eps(pred, x, a_bar_b)
    else:
        return pred


# -------------------------------------------------
# McKean–Vlasov reverse SDE sampler (with CFG + energy)
# -------------------------------------------------
def mv_sde_sample(
    unet_apply: Callable,
    params,
    shape: Tuple[int, int, int, int, int],  # (B,H,W,K,C)
    betas: jnp.ndarray,
    alphas: jnp.ndarray,
    alpha_bars: jnp.ndarray,
    cond_vec: jnp.ndarray,  # (B,D)
    steps: int = 250,
    rng=None,
    v_prediction: bool = True,
    # --- Classifier-free guidance (CFG) ---
    cfg_scale: float = 0.0,  # 0 = disabled; typical 1..5
    null_cond_vec: Optional[jnp.ndarray] = None,  # (B,D) for uncond pass; default zeros
    cfg_schedule: Literal["linear", "cosine", "exp"] = "cosine",
    cfg_strength: float = 5.0,  # used only for "exp" schedule
    # --- Energy guidance ---
    guidance_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    guidance_scale: float = 0.0,
    guidance_schedule: Literal["linear", "cosine", "exp"] = "cosine",
    guidance_strength: float = 3.0,  # used only for "exp" schedule
    # --- Mean-field (McKean–Vlasov) ---
    mf_mode: Literal["none", "voxel", "rbf"] = "rbf",
    mf_lambda: float = 0.0,
    mf_bandwidth: float = 0.5,  # only for "rbf" mode
    # --- Integration ---
    prob_flow_ode: bool = False,  # True → deterministic PF-ODE (DDIM-like)
    return_all: bool = False,
) -> jnp.ndarray:
    """
    Reverse-time MV-SDE for VP diffusion with:
      - Classifier-free guidance (two UNet passes per step when cfg_scale>0)
      - Energy guidance via ∇_x E(x,cond) (modules+label embedding)
      - Mean-field interaction term depending on batch empirical law

    dx = [ -0.5 β x - β sθ(x,t)
           - w_E(t) * λ_E ∇_x E(x,cond)
           + F_mf(x; w_MF) ] dt
         + 1_{not PF-ODE} * sqrt(β) dW

    Scheduling:
      - CFG weight:   w_CFG(t) ∈ [0,1]
      - Energy weight w_E(t)   ∈ [0,1]
      We apply these to (eps_cond, eps_uncond) blending and energy drift respectively.
    """
    T = alpha_bars.shape[0]
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # integer timesteps descending
    ts_idx = jnp.linspace(T - 1, 0, steps, dtype=jnp.int32)
    dt = 1.0 / float(steps)

    # initial noise
    rng, kx = jax.random.split(rng)
    x = jax.random.normal(kx, shape, dtype=jnp.float32)

    # default null cond = zeros (unconditional token)
    if null_cond_vec is None:
        null_cond_vec = jnp.zeros_like(cond_vec)

    traj = []

    for i in range(steps):
        t_i = ts_idx[i]
        t_cont = (t_i.astype(jnp.float32) + 0.5) / float(T)  # in (0,1]
        t_b = jnp.full((shape[0],), t_cont, dtype=jnp.float32)

        a_bar = alpha_bars[t_i]
        beta_t = betas[t_i]
        a_bar_b = jnp.full((shape[0], 1, 1, 1, 1), a_bar, dtype=jnp.float32)

        # --------- Model prediction → epŝ (with CFG) ---------
        eps_cond = _eps_hat_from_unet(
            unet_apply, params, x, t_b, cond_vec, a_bar_b, v_prediction
        )

        if cfg_scale and cfg_scale != 0.0:
            eps_uncond = _eps_hat_from_unet(
                unet_apply, params, x, t_b, null_cond_vec, a_bar_b, v_prediction
            )
            w_cfg = _time_weight(t_b, cfg_schedule, cfg_strength)  # (B,)
            w_cfg = w_cfg[:, None, None, None, None]
            eps_hat = eps_uncond + (1.0 + w_cfg * (cfg_scale - 1.0)) * (
                eps_cond - eps_uncond
            )
        else:
            eps_hat = eps_cond

        # Score
        score = _score_from_eps(eps_hat, a_bar_b)  # (B,H,W,K,C)

        # Base reverse SDE drift (VP)
        drift = -0.5 * beta_t * x - beta_t * score

        # --------- Energy guidance ---------
        if guidance_fn is not None and guidance_scale != 0.0:
            g = guidance_fn(x, t_b)  # ∇_x E
            w_E = _time_weight(t_b, guidance_schedule, guidance_strength)
            w_E = w_E[:, None, None, None, None]
            drift = drift - (w_E * guidance_scale) * g

        # --------- Mean-field interaction ---------
        if mf_mode == "voxel" and mf_lambda > 0.0:
            drift = drift + _mf_voxel_mean(x, mf_lambda)
        elif mf_mode == "rbf" and mf_lambda > 0.0:
            drift = drift + _mf_global_rbf(x, mf_lambda, mf_bandwidth)

        # --------- Integrate ---------
        if not prob_flow_ode:
            rng, kn = jax.random.split(rng)
            noise = jax.random.normal(kn, x.shape, dtype=x.dtype)
            x = (
                x
                + drift * dt
                + jnp.sqrt(jnp.clip(beta_t, 1e-8, None)) * jnp.sqrt(dt) * noise
            )
        else:
            x = x + drift * dt

        if return_all:
            traj.append(x)

    if return_all:
        return jnp.stack(traj, axis=1)  # (B, steps, H, W, K, C)
    return x


# -------------------------------------------------
# Legacy deterministic wrapper (DDIM-like)
# -------------------------------------------------
def ddim_sample(
    unet_apply: Callable,
    params,
    shape: Tuple[int, int, int, int, int],
    betas: jnp.ndarray,
    alphas: jnp.ndarray,
    alpha_bars: jnp.ndarray,
    cond_vec: jnp.ndarray,
    steps: int = 50,
    rng=None,
    v_prediction: bool = True,
    guidance_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    guidance_scale: float = 0.0,
    cfg_scale: float = 0.0,
    null_cond_vec: Optional[jnp.ndarray] = None,
    return_all: bool = False,
) -> jnp.ndarray:
    """
    Keeps old call sites alive; routes to MV sampler with prob_flow_ode=True.
    """
    return mv_sde_sample(
        unet_apply=unet_apply,
        params=params,
        shape=shape,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        cond_vec=cond_vec,
        steps=steps,
        rng=rng,
        v_prediction=v_prediction,
        cfg_scale=cfg_scale,
        null_cond_vec=null_cond_vec,
        guidance_fn=guidance_fn,
        guidance_scale=guidance_scale,
        mf_mode="rbf",
        mf_lambda=0.0,
        prob_flow_ode=True,
        return_all=return_all,
    )
