# losses_steps.py
from __future__ import annotations

from functools import partial
from dataclasses import replace as dc_replace
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import lax
import optax
from flax import struct
from flax.core import FrozenDict

# ===================== numerics / toggles =====================
_F32 = jnp.float32
_BF16 = jnp.bfloat16
SMALL = 1e-6

# bf16 forward (reduce activation mem) while keeping losses in fp32
USE_BF16_FORWARD = True

# gaussian blur (cheap separable 3x3 via shifts) mixed with original L
ENABLE_GAUSS_BLUR = True
BLUR_ALPHA = 0.25  # 0=no blur, 0.25 is mild augmentation

# Gumbel-Softmax exploration on negatives (denominator only)
ENABLE_GUMBEL = True
GUMBEL_T = 0.5  # lower => more exploration; 0.5–1.0 works well

# variance-spread regularizer (push energy distribution to use range)
VAR_TARGET = 0.25
VAR_WEIGHT = 0.05

# hard-negative margin weight
HINGE_MARGIN = 0.25
HINGE_WEIGHT = 0.5

FloatScalar = Union[float, jnp.ndarray]


# ===================== schedules =====================


def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = 0.008
    t = jnp.linspace(0.0, 1.0, T + 1, dtype=_F32)
    f = jnp.cos(((t + s) / (1 + s)) * jnp.pi * 0.5) ** 2
    f = f / jnp.maximum(f[0], 1e-12)
    alpha_bars = f[1:]
    alphas = alpha_bars / jnp.concatenate([jnp.array([1.0], _F32), alpha_bars[:-1]])
    betas = 1.0 - alphas
    return betas.astype(_F32), alphas.astype(_F32), alpha_bars.astype(_F32)


def linear_beta_schedule(
    T: int, start=1e-4, end=2e-2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    betas = jnp.linspace(start, end, T, dtype=_F32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas.astype(_F32), alphas.astype(_F32), alpha_bars.astype(_F32)


def vp_beta_schedule(
    T: int, beta_min: float = 0.1, beta_max: float = 20.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Variance-preserving continuous SDE discretization (DDPM++/VP):
    beta(t) = beta_min + t*(beta_max-beta_min), t in [0,1]
    alpha_bar(t) = exp(-∫_0^t beta(s) ds) = exp(-(beta_min*t + 0.5*(beta_max-beta_min)*t^2))
    """
    t = jnp.linspace(0.0, 1.0, T, dtype=_F32)
    integ = beta_min * t + 0.5 * (beta_max - beta_min) * (t * t)
    alpha_bars = jnp.exp(-integ)
    # back out discrete alphas/betas from alpha_bar
    alphas = alpha_bars / jnp.concatenate([jnp.array([1.0], _F32), alpha_bars[:-1]])
    betas = 1.0 - alphas
    return betas.astype(_F32), alphas.astype(_F32), alpha_bars.astype(_F32)


# ===================== State containers =====================


@struct.dataclass
class DiffusionState:
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    params: FrozenDict
    opt_state: optax.OptState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    ema_params: FrozenDict
    ema_decay: float = 0.999
    T: int = 1000
    v_prediction: bool = True

    def replace(self, **updates):
        return dc_replace(self, **updates)


@struct.dataclass
class EnergyState:
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    params: FrozenDict
    opt_state: optax.OptState
    tau: float = 0.10  # slightly warmer
    gp_lambda: float = 1e-4

    def replace(self, **updates):
        return dc_replace(self, **updates)


@struct.dataclass
class EncoderState:
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    params: FrozenDict
    opt_state: optax.OptState

    def replace(self, **updates):
        return dc_replace(self, **updates)


# ===================== State builders =====================


def create_diffusion_state(
    rng,
    apply_fn: Callable,
    init_params: FrozenDict,
    T: int = 1000,
    lr: float = 2e-4,
    v_prediction: bool = True,
    schedule: str = "vp",  # "cosine" | "linear" | "vp"
    ema_decay: float = 0.999,
) -> DiffusionState:
    if schedule == "cosine":
        betas, alphas, alpha_bars = cosine_beta_schedule(T)
    elif schedule == "linear":
        betas, alphas, alpha_bars = linear_beta_schedule(T)
    elif schedule == "vp":
        betas, alphas, alpha_bars = vp_beta_schedule(T)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    return DiffusionState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=int(T),
        v_prediction=bool(v_prediction),
        ema_params=init_params,
        ema_decay=float(ema_decay),
    )


def create_energy_state(
    apply_fn: Callable,
    init_params: FrozenDict,
    lr: float = 3e-4,
    tau: float = 0.10,
    gp_lambda: float = 1e-4,
) -> EnergyState:
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    return EnergyState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
        tau=float(tau),
        gp_lambda=float(gp_lambda),
    )


def create_encoder_state(
    apply_fn: Callable, init_params: FrozenDict, lr: float = 1e-3
) -> EncoderState:
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    return EncoderState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
    )


# ===================== helpers =====================


def _sinusoidal_time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10_000.0), half, dtype=_F32))
    angles = t_cont[:, None] * (1.0 / freqs[None, :])
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def _charbonnier(x: jnp.ndarray, eps: float = 1e-6, alpha: float = 0.5) -> jnp.ndarray:
    return jnp.power(x * x + eps, alpha)


def _reconstruct_x0_eps_from_v(
    xt: jnp.ndarray, v: jnp.ndarray, alpha_bar_t: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sa = jnp.sqrt(alpha_bar_t)[..., None, None, None, None]
    sb = jnp.sqrt(1.0 - alpha_bar_t)[..., None, None, None, None]
    x0_hat = sa * xt - sb * v
    eps_hat = sb * xt + sa * v
    return x0_hat, eps_hat


def _nan_to_num(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------- cheap separable 3x3 Gaussian-ish blur on (H,W,KS,C) ----------
def _blur2d_hw(x: jnp.ndarray) -> jnp.ndarray:
    """3x3 box-approx Gaussian on H,W dims. x: (H,W,KS,C)."""
    H, W, KS, C = x.shape
    xp = jnp.pad(x, ((1, 1), (1, 1), (0, 0), (0, 0)), mode="edge")
    # sum 3x3 neighborhood (no loops; 9 slices)
    s = (
        xp[0:H, 0:W]
        + xp[0:H, 1 : W + 1]
        + xp[0:H, 2 : W + 2]
        + xp[1 : H + 1, 0:W]
        + xp[1 : H + 1, 1 : W + 1]
        + xp[1 : H + 1, 2 : W + 2]
        + xp[2 : H + 2, 0:W]
        + xp[2 : H + 2, 1 : W + 1]
        + xp[2 : H + 2, 2 : W + 2]
    )
    return s / 9.0


# ---------- deterministic gumbel (stateless) ----------
def _det_uniform(v: jnp.ndarray) -> jnp.ndarray:
    """
    Deterministic pseudo-uniform (0,1) from a vector v (no RNG).
    Uses a sinusoidal hash -> fractional part in (0,1).
    """
    proj = jnp.sum(
        jax.lax.stop_gradient(v) * jnp.linspace(0.123, 0.987, v.shape[-1], dtype=_F32),
        axis=-1,
    )
    u = jnp.mod(jnp.sin(proj * 12345.678) * 43758.5453, 1.0)  # (..,)
    # ensure (0,1) open interval
    return jnp.clip(u * 0.999999 + 1e-6, 1e-6, 1.0 - 1e-6)


def _gumbel_from_cond(cond_seg: jnp.ndarray, t: float) -> jnp.ndarray:
    """Gumbel noise from cond rows. cond_seg: (chunk,D) -> gumbel: (chunk,)"""
    u = _det_uniform(cond_seg)
    g = -jnp.log(-jnp.log(u))
    return g / jnp.maximum(jnp.array(t, _F32), 1e-6)


# ===================== diffusion loss & step =====================


def _ddpm_loss(
    unet_apply: Callable,
    params: FrozenDict,
    rng,
    x0: jnp.ndarray,  # (B,H,W,KS,C) standardized
    alpha_bars: jnp.ndarray,  # (T,)
    alphas: jnp.ndarray,  # (T,)  (kept for completeness)
    v_prediction: bool,
    cond_vec: jnp.ndarray,  # (B, D)
    t_dim: int = 128,
    lambda_x0: float = 0.1,
    x0_weight_pow: float = 1.5,
) -> jnp.ndarray:
    x0 = _nan_to_num(x0)
    cond_vec = _nan_to_num(cond_vec)

    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)

    # sample t
    t_int = jax.random.randint(key_t, (B,), 0, alpha_bars.shape[0], dtype=jnp.int32)
    a_bar = alpha_bars[t_int].astype(_F32)  # (B,)

    # forward
    eps = jax.random.normal(key_eps, x0.shape, dtype=_F32)
    sa = jnp.sqrt(a_bar)[..., None, None, None, None]
    sb = jnp.sqrt(1.0 - a_bar)[..., None, None, None, None]
    xt = sa * x0 + sb * eps

    # temb
    t_cont = (t_int.astype(_F32) + 0.5) / float(alpha_bars.shape[0])
    temb = _sinusoidal_time_embed(t_cont, dim=t_dim)

    # bf16 forward
    if USE_BF16_FORWARD:
        pred = unet_apply(
            {"params": params},
            xt.astype(_BF16),
            temb.astype(_BF16),
            cond_vec.astype(_BF16),
        ).astype(_F32)
    else:
        pred = unet_apply({"params": params}, xt, temb, cond_vec).astype(_F32)

    # targets
    if v_prediction:
        target = sa * eps - sb * x0
    else:
        target = eps

    main = _charbonnier(pred - target).mean()

    # x0 consistency (weighted near t≈0)
    x0_hat, _ = _reconstruct_x0_eps_from_v(xt, pred, a_bar)
    w_x0 = jnp.power(1.0 - a_bar, x0_weight_pow)[..., None, None, None, None]
    aux = (w_x0 * _charbonnier(x0_hat - x0)).mean()
    return main + lambda_x0 * aux


@partial(jax.jit, static_argnames=("v_prediction",))
def diffusion_train_step(
    state: DiffusionState,
    x0: jnp.ndarray,  # (B,H,W,KS,C)
    rng,
    v_prediction: bool,
    cond_vec: jnp.ndarray,  # (B,D)
):
    def loss_fn(p):
        return _ddpm_loss(
            state.apply_fn,
            p,
            rng,
            x0,
            state.alpha_bars,
            state.alphas,
            v_prediction,
            cond_vec,
            t_dim=128,
            lambda_x0=0.1,
            x0_weight_pow=1.5,
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_ema = tree_map(
        lambda e, q: state.ema_decay * e + (1.0 - state.ema_decay) * q,
        state.ema_params,
        new_params,
    )
    new_state = state.replace(params=new_params, opt_state=new_opt, ema_params=new_ema)
    return new_state, loss


# ======================= ENERGY: streaming, non-saturating =======================


def _stream_logsumexp(
    acc_m_s: Tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray, mask: jnp.ndarray
):
    """Update running logsumexp using max-trick with masked entries."""
    m, s = acc_m_s
    xm = jnp.where(mask, x, -jnp.inf)
    m_new = jnp.maximum(m, jnp.max(xm))
    s_new = jnp.exp(m - m_new) * s + jnp.sum(jnp.where(mask, jnp.exp(x - m_new), 0.0))
    return (m_new, s_new)


def _row_pass(E_apply, params, Li, cond_all, tau: FloatScalar, chunk: int):
    """
    Single row: CE with positive at index 0, + hard-negative hinge + variance-spread.
    - Li: (H,W,KS,C)   (NaN-safe)
    - cond_all: (B_all, D) with positive at position 0.
    """
    Li = _nan_to_num(Li)
    cond_all = _nan_to_num(cond_all)

    # light blur augmentation blend on Li
    if ENABLE_GAUSS_BLUR:
        Li_blur = _blur2d_hw(Li.astype(_F32))
        Li = (1.0 - BLUR_ALPHA) * Li + BLUR_ALPHA * Li_blur

    B_all, D = cond_all.shape
    tau = jnp.asarray(tau, _F32)

    # pad to multiple of chunk
    nseg = (B_all + chunk - 1) // chunk
    B_pad = nseg * chunk
    pad_n = B_pad - B_all
    cond_pad = jnp.pad(cond_all, ((0, pad_n), (0, 0)))
    valid = jnp.concatenate(
        [jnp.ones((B_all,), dtype=bool), jnp.zeros((pad_n,), dtype=bool)], 0
    )

    # accumulators
    m0 = jnp.array(-jnp.inf, _F32)
    s0 = jnp.array(0.0, _F32)
    sum_e0 = jnp.array(0.0, _F32)
    sumsq_e0 = jnp.array(0.0, _F32)
    min_neg0 = jnp.array(jnp.inf, _F32)

    def body(carry, s):
        m, ssum, sum_e, sumsq_e, min_neg = carry
        j0 = s * chunk
        cond_seg = lax.dynamic_slice(cond_pad, (j0, 0), (chunk, D))
        mask_seg = lax.dynamic_slice(valid, (j0,), (chunk,))

        LiB = jnp.broadcast_to(Li, (chunk,) + Li.shape)
        if USE_BF16_FORWARD:
            e_seg = E_apply(
                {"params": params}, LiB.astype(_BF16), cond_seg.astype(_BF16)
            ).astype(_F32)
        else:
            e_seg = E_apply({"params": params}, LiB, cond_seg).astype(_F32)

        logits_seg = -e_seg / jnp.maximum(tau, 1e-6)

        # add Gumbel noise on denominator only (exploration); positive added later cleanly
        if ENABLE_GUMBEL:
            g = _gumbel_from_cond(cond_seg, GUMBEL_T)  # (chunk,)
            logits_seg = logits_seg + g

        # stream logsumexp
        m_new, s_new = _stream_logsumexp((m, ssum), logits_seg, mask_seg)

        # raw energy stats
        e_valid = jnp.where(mask_seg, e_seg, 0.0)
        sum_e_new = sum_e + e_valid.sum()  # type: ignore
        sumsq_e_new = sumsq_e + (e_valid * e_valid).sum()  # type: ignore

        # min negative across segment (we'll exclude pos later); we don't know its pos here
        min_neg_new = jnp.minimum(min_neg, jnp.where(mask_seg, e_seg, jnp.inf).min())  # type: ignore

        return (m_new, s_new, sum_e_new, sumsq_e_new, min_neg_new), None

    (m_fin, s_fin, sum_e_fin, sumsq_e_fin, min_neg_global), _ = lax.scan(
        body, (m0, s0, sum_e0, sumsq_e0, min_neg0), jnp.arange(nseg)
    )

    # exact positive at index 0
    e_pos = E_apply(
        {"params": params},
        Li[None, ...].astype(_BF16) if USE_BF16_FORWARD else Li[None, ...],
        cond_all[:1, :].astype(_BF16) if USE_BF16_FORWARD else cond_all[:1, :],
    ).astype(_F32)[0]
    logit_pos = -e_pos / jnp.maximum(tau, 1e-6)

    # CE
    ce = -(logit_pos - (m_fin + jnp.log(jnp.maximum(s_fin, 1e-12))))

    # variance
    n_valid = jnp.array(B_all, _F32)
    mean_e = sum_e_fin / jnp.maximum(n_valid, 1.0)
    var_e = jnp.maximum(sumsq_e_fin / jnp.maximum(n_valid, 1.0) - mean_e * mean_e, 0.0)

    # hinge on hardest negative
    hinge = jnp.maximum(0.0, e_pos - min_neg_global + HINGE_MARGIN)

    return ce, hinge, var_e


def _gp_subset(E_apply, params, L, cond_vec, gp_subset: int):
    """Small gradient penalty wrt volume (first M samples)."""
    M = int(min(L.shape[0], gp_subset))

    def e_mean(L_, C_):
        return E_apply({"params": params}, L_, C_).mean()

    L_sub = _nan_to_num(L[:M])
    C_sub = _nan_to_num(cond_vec[:M])
    g = jax.grad(e_mean, argnums=0)(L_sub, C_sub)
    return jnp.mean(jnp.square(g))


def _coerce_1d_energy(x, n_expected: int | None = None) -> jnp.ndarray:
    """Accept scalar/1D array or (energy, aux) tuple; return 1D float32 with NaNs cleaned."""
    if isinstance(x, (tuple, list)):
        x = x[0]
    x = jnp.asarray(x, dtype=_F32)
    if n_expected is not None and (x.ndim != 1 or x.shape[0] != n_expected):
        x = x.reshape((n_expected,))
    return _nan_to_num(x)


def _row_logits_chunked(
    E_apply,
    params,
    Li: jnp.ndarray,  # (H,W,KS,C)  one volume row
    cond_vec: jnp.ndarray,  # (B,D)
    tau: jnp.ndarray | float,
    chunk: int,
) -> jnp.ndarray:
    """
    Returns logits (B,) = -E(Li, cond_j)/tau in memory-bounded chunks.
    Uses lax.dynamic_slice with static slice sizes; safe under jit/scan.
    """
    tau = jnp.asarray(tau, _F32)
    B, D = cond_vec.shape
    H, W, KS, C = Li.shape  # static sizes flow from Li's shape

    # pad conds to multiple of chunk so slice sizes are static
    nseg = (B + chunk - 1) // chunk  # Python ints (static)
    B_pad = nseg * chunk
    pad_n = B_pad - B

    cond_pad = jnp.pad(cond_vec, ((0, pad_n), (0, 0)))
    valid_mask = jnp.concatenate(
        [jnp.ones((B,), dtype=bool), jnp.zeros((pad_n,), dtype=bool)], axis=0
    )

    # init output buffer
    logits0 = jnp.full((B_pad,), -jnp.inf, dtype=_F32)

    def body(logits, s):
        j0 = s * chunk  # segment start (scalar tracer)
        # static sizes for slices
        cond_seg = lax.dynamic_slice(cond_pad, (j0, 0), (chunk, D))  # (chunk,D)
        mask_seg = lax.dynamic_slice(valid_mask, (j0,), (chunk,))  # (chunk,)
        # broadcast Li -> (chunk,H,W,KS,C)
        LiB = jnp.broadcast_to(Li, (chunk, H, W, KS, C))
        # energy for this segment and coercion to clean 1D array
        raw_e_seg = E_apply({"params": params}, LiB, cond_seg)
        e_seg = _coerce_1d_energy(raw_e_seg, n_expected=chunk)  # (chunk,)
        # logits = -E/tau; invalidate padded positions
        logit_seg = -e_seg / jnp.maximum(tau, 1e-6)
        logit_seg = jnp.where(mask_seg, logit_seg, -jnp.inf)
        # write back into the padded buffer
        logits = lax.dynamic_update_slice_in_dim(logits, logit_seg, j0, axis=0)
        return logits, None

    logits_pad, _ = lax.scan(body, logits0, jnp.arange(nseg))
    return logits_pad[:B]


def energy_total_loss_chunked(
    E_apply,
    params,
    L,  # (B,H,W,KS,C)
    cond_vec,  # (B,D)
    tau: float | jnp.ndarray,
    gp_lambda: float | jnp.ndarray = 1e-4,
    margin: float = 0.1,
    hinge_weight: float = 0.25,
    chunk: int = 8,
    gp_subset: int = 4,
) -> jnp.ndarray:
    """Row-wise CE + hinge + optional GP. JAX-safe dynamic indexing."""
    tau = jnp.asarray(tau, _F32)
    gp_lambda = jnp.asarray(gp_lambda, _F32)

    B, D = cond_vec.shape
    H, W, KS, C = map(int, L.shape[1:])  # static sizes for dynamic_slice

    def row_body(carry, i):
        # Li = L[i] with dynamic_slice (shape -> (1,H,W,KS,C) then squeeze)
        Li1 = lax.dynamic_slice(L, (i, 0, 0, 0, 0), (1, H, W, KS, C))
        Li = Li1[0]

        # logits for this row against all conds (chunked to save mem)
        logits = _row_logits_chunked(E_apply, params, Li, cond_vec, tau, chunk)  # (B,)

        # cross-entropy with label i
        ce_i = optax.softmax_cross_entropy_with_integer_labels(
            logits[None, :], jnp.array([i], jnp.int32)
        ).squeeze()

        # hinge on raw energies
        E_row = -tau * logits
        pos = E_row[i]
        diff = E_row - pos
        mask = 1.0 - jax.nn.one_hot(i, B)
        hinge_i = jnp.sum(jnp.maximum(0.0, margin - diff) * mask) / jnp.clip(
            jnp.sum(mask), 1.0
        )

        # simple variance regularizer over negatives (non-saturating signal)
        neg_vals = jnp.where(mask > 0, E_row, jnp.nan)
        mean_neg = jnp.nanmean(neg_vals)  # type: ignore
        var_neg = jnp.nanmean((neg_vals - mean_neg) ** 2)

        ce_sum, hinge_sum, var_sum = carry
        return (ce_sum + ce_i, hinge_sum + hinge_i, var_sum + var_neg), None

    (ce_tot, hinge_tot, var_tot), _ = lax.scan(row_body, (0.0, 0.0, 0.0), jnp.arange(B))
    ce = ce_tot / B
    hinge = hinge_tot / B
    var_reg = var_tot / B * 0.01  # tiny weight; helps avoid collapse

    # optional GP without branching on Python bools
    def _gp_branch(_):
        M = int(min(B, gp_subset))
        L_sub = lax.dynamic_slice(L, (0, 0, 0, 0, 0), (M, H, W, KS, C))
        C_sub = lax.dynamic_slice(cond_vec, (0, 0), (M, D))

        def e_mean(L_, C_):  # mean scalar energy
            e = E_apply({"params": params}, L_, C_)
            if isinstance(e, (tuple, list)):
                e = e[0]
            return jnp.mean(jnp.asarray(e, _F32))

        g = jax.grad(e_mean, argnums=0)(L_sub, C_sub)
        return jnp.mean(jnp.square(g)).astype(_F32)

    gp_term = lax.cond(
        gp_lambda > 0.0, _gp_branch, lambda _: jnp.array(0.0, _F32), operand=None
    )

    return ce + hinge_weight * hinge + var_reg + gp_lambda * gp_term


# ===================== train steps (jit) =====================


@partial(jax.jit, static_argnames=("E_apply", "chunk", "gp_subset"))
def energy_step_E(
    e_state,
    L: jnp.ndarray,
    cond_vec: jnp.ndarray,
    E_apply,
    chunk: int = 8,
    gp_subset: int = 4,
):
    tau = jnp.asarray(e_state.tau, _F32)
    gp_lambda = jnp.asarray(e_state.gp_lambda, _F32)

    def loss_e(params):
        return energy_total_loss_chunked(
            E_apply,
            params,
            L,
            cond_vec,
            tau=tau,
            gp_lambda=gp_lambda,
            margin=0.1,
            hinge_weight=0.25,
            chunk=chunk,
            gp_subset=gp_subset,
        )

    loss, grads = jax.value_and_grad(loss_e)(e_state.params)
    updates, new_opt = e_state.tx.update(grads, e_state.opt_state, e_state.params)
    new_params = optax.apply_updates(e_state.params, updates)
    new_state = e_state.replace(params=new_params, opt_state=new_opt)
    return new_state, loss


@partial(jax.jit, static_argnames=("E_apply", "chunk"))
def energy_step_encoder(
    enc_state,
    y_emb: jnp.ndarray,  # (B, y_dim)
    feats_b: jnp.ndarray,  # (B, T_max, S_max, F)
    set_b: jnp.ndarray,  # (B, T_max, S_max, 1)
    time_b: jnp.ndarray,  # (B, T_max, 1)
    E_apply,  # callable: ({'params': eparams}, Lb, Cb)->(B,) or (B,...)
    eparams: FrozenDict,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    tau: float | jnp.ndarray,  # scalar temperature
    chunk: int = 8,
):
    """
    Encoder update = row-wise InfoNCE (cross-entropy) using energy as logits.
    Memory-safe: computes logits for each row Li vs all conds in fixed-size chunks.
    No dynamic Python slicing; everything uses lax.scan and static slice sizes.
    """
    tau = jnp.asarray(tau, jnp.float32)
    B = L.shape[0]

    def loss_enc(enc_params):
        # embed modules -> m_emb, then build cond_vec = [y_emb | m_emb]
        m_emb = enc_state.apply_fn(
            {"params": enc_params}, feats_b, set_b, time_b
        )  # (B, d_m)
        cond_vec = jnp.concatenate([y_emb, m_emb], axis=-1)  # (B, D)

        # per-row CE without building a full BxB matrix
        def row_ce(carry, i):
            Li = L[i]  # (H,W,KS,C); single row volume
            logits = _row_logits_chunked(
                E_apply, eparams, Li, cond_vec, tau, chunk
            )  # (B,)

            # CE with label==i (stable softmax in optax)
            ce_i = optax.softmax_cross_entropy_with_integer_labels(
                logits[None, :], jnp.array([i], jnp.int32)
            ).squeeze()
            return carry + ce_i, None

        ce_sum, _ = lax.scan(row_ce, jnp.array(0.0, jnp.float32), jnp.arange(B))
        return ce_sum / jnp.maximum(B, 1)

    loss, grads = jax.value_and_grad(loss_enc)(enc_state.params)
    updates, new_opt = enc_state.tx.update(grads, enc_state.opt_state, enc_state.params)
    new_params = optax.apply_updates(enc_state.params, updates)
    new_state = enc_state.replace(params=new_params, opt_state=new_opt)
    return new_state, loss
