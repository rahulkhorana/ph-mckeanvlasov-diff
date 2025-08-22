# sampling.py
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple


# ---------------------------
# Time embedding (used here)
# ---------------------------
def time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """Sinusoidal embedding for continuous t in [0,1]."""
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000.0), half, dtype=jnp.float32))
    ang = t_cont[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb.astype(jnp.float32)


# ---------------------------
# Energy bridge (module-wise)
# ---------------------------
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
        # We only use the gradient w.r.t. x0_hat (guidance towards lower energy)
        return grad_fn(x0_hat)

    return bridge_fn


# ---------------------------
# Helper: combine CFG
# ---------------------------
def _cfg_merge(
    pred_uncond: jnp.ndarray, pred_cond: jnp.ndarray, scale: float
) -> jnp.ndarray:
    # pred = u + s*(c - u)  (classifier-free guidance)
    return pred_uncond + scale * (pred_cond - pred_uncond)


# ---------------------------
# Helper: convert head -> eps
# ---------------------------
def _to_eps_from_pred(
    pred: jnp.ndarray,
    x_t: jnp.ndarray,
    sqrt_ab_t: jnp.ndarray,
    sqrt_one_ab_t: jnp.ndarray,
    v_prediction: bool,
) -> jnp.ndarray:
    """
    Given UNet output 'pred' as either epsilon or v, return epsilon.
    """
    if v_prediction:
        # v = (sqrt(a) * eps - sqrt(1-a) * x)  (one common definition)
        # -> eps = (v + sqrt(1-a) * x) / sqrt(a)
        # But we follow the stable variant used in code before:
        # eps = sqrt(a)*pred + sqrt(1-a)*x
        return sqrt_ab_t * pred + sqrt_one_ab_t * x_t
    else:
        return pred


# ---------------------------
# Helper: score from eps
# ---------------------------
def _score_from_eps(eps: jnp.ndarray, sigma_t: jnp.ndarray) -> jnp.ndarray:
    # ∇_x log p_t(x) ∝ - eps / sigma_t
    sigma_t = jnp.maximum(sigma_t, 1e-8)
    return -eps / sigma_t


# ---------------------------
# Mean-field coupling (local)
# ---------------------------
def _gaussian_kernel_3d(ks: int, sigma: float) -> jnp.ndarray:
    """(ks,ks,ks) normalized Gaussian kernel."""
    ax = jnp.arange(ks, dtype=jnp.float32) - (ks - 1) / 2.0
    g1 = jnp.exp(-(ax**2) / (2 * sigma * sigma))
    g1 = g1 / jnp.maximum(jnp.sum(g1), 1e-8)
    K = (g1[:, None, None] * g1[None, :, None] * g1[None, None, :]).astype(jnp.float32)
    K = K / jnp.maximum(jnp.sum(K), 1e-8)
    return K


def _depthwise_conv3d_same(
    x: jnp.ndarray, kernel: jnp.ndarray  # (B,H,W,K,C)  # (kh,kw,kk)
) -> jnp.ndarray:
    """Depthwise 3D conv per-channel, SAME padding, stride 1."""
    B, H, W, K, C = x.shape
    ker = kernel[:, :, :, None, None]  # (kh,kw,kk,1,1)

    def conv_one_channel(xc: jnp.ndarray) -> jnp.ndarray:
        # xc: (B,H,W,K) -> (B,H,W,K)
        xc = xc[..., None]  # (B,H,W,K,1)
        y = jax.lax.conv_general_dilated(
            xc,
            ker,
            window_strides=(1, 1, 1),
            padding="SAME",
            dimension_numbers=("NHWDC", "DHWIO", "NHWDC"),
        )
        return jnp.squeeze(y, axis=-1)

    # vmaps over channel
    x_c_first = jnp.moveaxis(x, -1, 0)  # (C,B,H,W,K)
    y_c = jax.vmap(conv_one_channel)(x_c_first)  # (C,B,H,W,K)
    y = jnp.moveaxis(y_c, 0, -1)  # (B,H,W,K,C)
    return y


def mean_field_force(
    x_t: jnp.ndarray,
    mode: str = "rbf",
    bandwidth: float = 0.5,
    kernel_size: int = 3,
    strength: float = 1.0,
) -> jnp.ndarray:
    """
    Compute a cheap per-batch mean-field force F[x_t].
    - 'rbf'   : local Gaussian smoothing, F = (Gσ * x) - x
    - 'voxel' : small uniform box avg,     F = Avg3x3x3(x) - x
    """
    if strength == 0.0:
        return jnp.zeros_like(x_t)

    if mode == "rbf":
        sigma = jnp.maximum(bandwidth, 1e-4)
        ks = int(max(3, kernel_size | 1))  # odd >=3
        ker = _gaussian_kernel_3d(ks, float(sigma))
        sm = _depthwise_conv3d_same(x_t, ker)
        return strength * (sm - x_t)

    elif mode == "voxel":
        ks = int(max(3, kernel_size | 1))
        ker = jnp.ones((ks, ks, ks), dtype=jnp.float32)
        ker = ker / jnp.maximum(jnp.sum(ker), 1e-8)
        sm = _depthwise_conv3d_same(x_t, ker)
        return strength * (sm - x_t)

    else:
        return jnp.zeros_like(x_t)


# ---------------------------
# DDIM sampler (η = 0)
# ---------------------------
def ddim_sample(
    unet_apply: Callable,
    params,
    shape: Tuple[int, int, int, int, int],
    betas: jnp.ndarray,
    alphas: jnp.ndarray,
    alpha_bars: jnp.ndarray,
    cond_vec: jnp.ndarray,  # (B, D), fixed during sampling
    steps: int = 50,
    rng: Optional[jax.Array] = None,
    v_prediction: bool = True,
    bridge_fn: Optional[Callable] = None,  # bridge(x_t, t_cont, x0_hat) -> grad wrt x0
    bridge_scale: float = 1.0,
    cfg_scale: float = 0.0,
    cond_uncond_vec: Optional[jnp.ndarray] = None,  # (B, D) for CFG
    return_all: bool = False,
):
    """
    Deterministic DDIM (η=0) with optional module-bridge (energy on x0_hat) and CFG.
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
        temb = time_embed(jnp.full((shape[0],), t_cont, dtype=jnp.float32), dim=128)

        # Predict (eps or v) with conditional (and optional unconditional for CFG)
        pred_c = unet_apply({"params": params}, x, temb, cond_vec)  # (B,....,C)
        if cfg_scale != 0.0 and cond_uncond_vec is not None:
            pred_u = unet_apply({"params": params}, x, temb, cond_uncond_vec)
            pred = _cfg_merge(pred_u, pred_c, float(cfg_scale))
        else:
            pred = pred_c

        # Convert to epsilon
        eps = _to_eps_from_pred(pred, x, sqrt_ab, sqrt_1ab, v_prediction)

        # Bridge on x0_hat
        x0_hat = (x - sqrt_1ab * eps) / jnp.maximum(sqrt_ab, 1e-8)
        if bridge_fn is not None and bridge_scale != 0.0:
            g_b = bridge_fn(x, t_cont, x0_hat)
            w_t = jnp.sqrt(jnp.maximum(1.0 - a_t, 0.0))  # stronger near noise
            eps = eps - bridge_scale * w_t * g_b

        # DDIM update (η=0)
        x0_hat = (x - sqrt_1ab * eps) / jnp.maximum(sqrt_ab, 1e-8)
        x = jnp.sqrt(jnp.maximum(a_tm1, 0.0)) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x


# ---------------------------
# MV-SDE sampler
# ---------------------------
def mv_sde_sample(
    unet_apply: Callable,
    params,
    shape: Tuple[int, int, int, int, int],
    betas: jnp.ndarray,  # (T,)
    alphas: jnp.ndarray,  # (T,)
    alpha_bars: jnp.ndarray,  # (T,)
    cond_vec: jnp.ndarray,  # (B,D)
    steps: int = 50,
    rng: Optional[jax.Array] = None,
    v_prediction: bool = True,
    # mean-field
    mf_mode: str = "rbf",
    mf_lambda: float = 0.05,
    mf_bandwidth: float = 0.5,
    mf_kernel_size: int = 3,
    # energy bridge on x0_hat
    bridge_fn: Optional[Callable] = None,
    bridge_scale: float = 1.0,
    # classifier-free guidance
    cfg_scale: float = 0.0,
    cond_uncond_vec: Optional[jnp.ndarray] = None,
    # ODE vs SDE
    prob_flow_ode: bool = True,
    # noise scale for SDE branch (relative to DDPM variance)
    sde_eta: float = 1.0,
    return_all: bool = False,
):
    """
    McKean–Vlasov SDE sampler with mean-field coupling and optional prob-flow ODE.
    Drift = DDPM/DDIM drift + mf_lambda * F[x_t].

    - If prob_flow_ode=True: integrates the probability-flow ODE (deterministic), equivalent to DDIM(η=0) + MF/bridge.
    - Else: Euler–Maruyama with diffusion term scaled by `sde_eta`.
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
        sigma_t = sqrt_1ab

        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)
        temb = time_embed(jnp.full((shape[0],), t_cont, dtype=jnp.float32), dim=128)

        # UNet predictions
        pred_c = unet_apply({"params": params}, x, temb, cond_vec)
        if cfg_scale != 0.0 and cond_uncond_vec is not None:
            pred_u = unet_apply({"params": params}, x, temb, cond_uncond_vec)
            pred = _cfg_merge(pred_u, pred_c, float(cfg_scale))
        else:
            pred = pred_c

        # epsilon + score
        eps = _to_eps_from_pred(pred, x, sqrt_ab, sqrt_1ab, v_prediction)
        score = _score_from_eps(eps, sigma_t)  # shape like x

        # Mean-field force at x_t
        F_mf = mean_field_force(
            x,
            mode=mf_mode,
            bandwidth=mf_bandwidth,
            kernel_size=mf_kernel_size,
            strength=1.0,
        )
        # Optional energy bridge on x0_hat (acts through eps correction)
        if bridge_fn is not None and bridge_scale != 0.0:
            x0_hat = (x - sqrt_1ab * eps) / jnp.maximum(sqrt_ab, 1e-8)
            g_b = bridge_fn(x, t_cont, x0_hat)
            w_t = jnp.sqrt(jnp.maximum(1.0 - a_t, 0.0))
            eps = eps - bridge_scale * w_t * g_b
            score = _score_from_eps(eps, sigma_t)

        if prob_flow_ode:
            # Probability-flow ODE update ≈ DDIM(η=0), add MF drift in x-space.
            x0_hat = (x - sqrt_1ab * eps) / jnp.maximum(sqrt_ab, 1e-8)
            x_det = jnp.sqrt(jnp.maximum(a_tm1, 0.0)) * x0_hat
            # Small MF push (discrete surrogate of ∫ λ F[x_t] dt across step)
            # Weight by w_t so it fades as we approach t=0
            w_t = jnp.sqrt(jnp.maximum(1.0 - a_t, 0.0))
            x = x_det + mf_lambda * w_t * F_mf
        else:
            # Euler–Maruyama step for reverse-time SDE
            # Reverse SDE drift ~ -0.5*beta_t*x_t - beta_t * score
            beta_t = 1.0 - alphas[t]
            drift = -0.5 * beta_t * x - beta_t * score + mf_lambda * F_mf
            # Δt per step across [0,1]: simple uniform partition
            dt = 1.0 / float(steps)
            # diffusion term magnitude ~ sqrt(beta_t) (variance-preserving SDE)
            g_t = jnp.sqrt(jnp.maximum(beta_t, 1e-8))
            key, sub = jax.random.split(key)
            z = jax.random.normal(sub, shape)
            x = x + drift * dt + sde_eta * g_t * jnp.sqrt(dt) * z

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x
