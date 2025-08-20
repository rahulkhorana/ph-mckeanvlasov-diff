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
def _norm_cond(cond_vec: jnp.ndarray, target_rms: float = 1.0) -> jnp.ndarray:
    # Cap per-batch RMS to stabilize FiLM
    rms = jnp.sqrt(jnp.mean(jnp.square(cond_vec)) + 1e-8)
    scale = jnp.minimum(1.0, target_rms / rms)
    cond = cond_vec * scale
    return jnp.nan_to_num(cond, posinf=1e6, neginf=-1e6)


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
def diffusion_train_step(state: DiffusionState, x0, rng, v_prediction: bool, cond_vec):
    cond_vec = _norm_cond(cond_vec, target_rms=1.0)

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


def _nan_to_num_clip(x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    # Replace NaN/±Inf then clip
    x = jnp.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)
    return jnp.clip(x, lo, hi)


def _safe_cond(cond_vec: jnp.ndarray, clip_val: float = 100.0) -> jnp.ndarray:
    # Keep condition vectors finite and bounded; no renorm (to preserve semantics)
    return _nan_to_num_clip(cond_vec, -clip_val, clip_val)


def _safe_volume(L: jnp.ndarray, clip_val: float = 20.0) -> jnp.ndarray:
    # Volumes may contain NaN (from upstream persistence transforms). Zero them and clip.
    return _nan_to_num_clip(L, -clip_val, clip_val)


def _safe_energy(e: jnp.ndarray, clip_val: float = 1e4) -> jnp.ndarray:
    # Clamp & denan just in case model spikes
    return _nan_to_num_clip(e, -clip_val, clip_val)


def _row_logits_chunked(
    E_apply,
    params,
    Li: jnp.ndarray,
    cond_vec: jnp.ndarray,
    tau: jnp.ndarray,
    chunk: int,
    gumbel_scale: float,
) -> jnp.ndarray:
    """
    Li (H,W,KS,C) vs all cond rows in fixed-size chunks.
    - Static slice sizes; no BxB alloc.
    - Debiased per-chunk energies: center & scale.
    - Always add Gumbel jitter scaled by gumbel_scale (0 allowed).
    """
    B, D = cond_vec.shape
    nseg = (B + chunk - 1) // chunk
    B_pad = nseg * chunk
    pad_n = B_pad - B

    cond_pad = jnp.pad(cond_vec, ((0, pad_n), (0, 0)))
    valid = jnp.concatenate(
        [jnp.ones((B,), dtype=bool), jnp.zeros((pad_n,), dtype=bool)], 0
    )

    logits0 = jnp.full((B_pad,), -jnp.inf, jnp.float32)
    gscale = jnp.asarray(gumbel_scale, jnp.float32)
    tau_safe = jnp.maximum(tau, 1e-6)

    def body(carry, s):
        j0 = s * chunk
        seg = lax.dynamic_slice(cond_pad, (j0, 0), (chunk, D))
        mask = lax.dynamic_slice(valid, (j0,), (chunk,))

        LiB = jnp.broadcast_to(Li, (chunk,) + Li.shape)
        e_seg = E_apply({"params": params}, LiB, seg)  # (chunk,)
        e_seg = jnp.nan_to_num(e_seg, posinf=1e6, neginf=-1e6)

        # Debias per-segment (approx per-row standardization)
        mu = jnp.mean(e_seg)
        sd = jnp.sqrt(jnp.mean((e_seg - mu) ** 2) + 1e-6)
        e_n = (e_seg - mu) / sd

        logits = -e_n / tau_safe

        # Tiny always-on Gumbel
        k = jax.random.PRNGKey(s)
        u = jax.random.uniform(k, logits.shape, minval=1e-6, maxval=1 - 1e-6)
        g = -jnp.log(-jnp.log(u))
        logits = logits + gscale * g

        logits = jnp.where(mask, logits, -jnp.inf)
        carry = lax.dynamic_update_slice_in_dim(carry, logits, j0, axis=0)
        return carry, None

    logits_pad, _ = lax.scan(body, logits0, jnp.arange(nseg))
    return logits_pad[:B]


def _row_losses_non_sat(
    E_apply,
    params,
    L: jnp.ndarray,
    cond_vec: jnp.ndarray,
    tau: jnp.ndarray,
    margin: float,
    hinge_topk: int,
    chunk: int,
    gumbel_scale: float,
):
    """
    For each row i:
      - CE with target i (InfoNCE) on debiased logits.
      - Hinge on top-k hardest negatives using normalized energies.
    """
    B = L.shape[0]
    tau = jnp.asarray(tau, jnp.float32)
    k = int(hinge_topk)
    can_hinge = jnp.array(B >= (k + 1))

    def body(carry, i):
        Li = L[i]
        logits = _row_logits_chunked(
            E_apply, params, Li, cond_vec, tau, chunk, gumbel_scale
        )  # (B,)

        ce_i = optax.softmax_cross_entropy_with_integer_labels(
            logits[None, :], jnp.array([i], jnp.int32)
        ).squeeze()

        # back to normalized energies to compute hinge
        E_row = -tau * logits
        pos = E_row[i]
        idx = jnp.arange(B)
        negs = jnp.where(idx == i, jnp.inf, E_row)
        negs = jnp.nan_to_num(negs, nan=jnp.inf)

        scores = -negs  # smaller energy => larger score

        def do_hinge(_):
            topk_vals, _ = lax.top_k(scores, k)  # static k
            hard_negs = -topk_vals
            rel = jnp.maximum(0.0, margin - (hard_negs - pos))
            return jnp.mean(rel)

        hinge_i = lax.cond(
            can_hinge, do_hinge, lambda _: jnp.array(0.0, jnp.float32), operand=None
        )

        return (carry[0] + ce_i, carry[1] + hinge_i), None

    (ce_sum, hinge_sum), _ = lax.scan(body, (0.0, 0.0), jnp.arange(B))
    invB = 1.0 / jnp.maximum(B, 1)
    return ce_sum * invB, hinge_sum * invB


# Optional tiny gradient penalty on a very small subset
def _gp_subset(E_apply, params, L: jnp.ndarray, cond_vec: jnp.ndarray, gp_subset: int):
    K = int(gp_subset)  # ensure static
    Lh = lax.dynamic_slice_in_dim(L, 0, K, axis=0)
    Ch = lax.dynamic_slice_in_dim(cond_vec, 0, K, axis=0)

    def e_mean(Lhh, Chh):
        return jnp.mean(E_apply({"params": params}, Lhh, Chh))

    g = jax.grad(e_mean, argnums=0)(Lh, Ch)
    return jnp.mean(jnp.square(g))


def energy_total_loss_chunked(
    E_apply,
    params,
    L,
    cond_vec,
    tau,
    gp_lambda,
    margin: float,
    hinge_weight: float,
    chunk: int,
    gp_subset: int,
    hinge_topk: int,
    gumbel_scale: float,
):
    tau = jnp.asarray(tau, jnp.float32)
    gp_lambda = jnp.asarray(gp_lambda, jnp.float32)

    # stabilize cond for energy too
    cond_vec = _norm_cond(jnp.asarray(cond_vec, jnp.float32), target_rms=1.0)

    ce, hinge = _row_losses_non_sat(
        E_apply, params, L, cond_vec, tau, margin, hinge_topk, chunk, gumbel_scale
    )

    def do_gp(_):
        return _gp_subset(E_apply, params, L, cond_vec, gp_subset)

    gp = lax.cond(
        gp_lambda > 0.0, do_gp, lambda _: jnp.array(0.0, jnp.float32), operand=None
    )

    return ce + hinge_weight * hinge + gp_lambda * gp


@partial(jax.jit, static_argnames=("E_apply", "chunk", "gp_subset", "hinge_topk"))
def energy_step_E(
    e_state: EnergyState,
    L: jnp.ndarray,
    cond_vec: jnp.ndarray,
    E_apply,
    chunk: int = 4,  # small to control VRAM
    gp_subset: int = 2,  # must be <= batch
    hinge_topk: int = 2,  # must be <= batch-1
    margin: float = 0.25,
    hinge_weight: float = 0.5,
    gumbel_scale: float = 0.02,
):
    tau = jnp.asarray(e_state.tau, jnp.float32)
    gp_lambda = jnp.asarray(e_state.gp_lambda, jnp.float32)

    def loss_e(params):
        return energy_total_loss_chunked(
            E_apply,
            params,
            L,
            cond_vec,
            tau,
            gp_lambda,
            margin=margin,
            hinge_weight=hinge_weight,
            chunk=chunk,
            gp_subset=gp_subset,
            hinge_topk=hinge_topk,
            gumbel_scale=gumbel_scale,
        )

    loss, grads = jax.value_and_grad(loss_e)(e_state.params)
    updates, new_opt = e_state.tx.update(grads, e_state.opt_state, e_state.params)
    new_params = optax.apply_updates(e_state.params, updates)
    new_state = e_state.replace(params=new_params, opt_state=new_opt)
    return new_state, loss


# --- Enc step: update the modules encoder only (keep E frozen) ---


@partial(jax.jit, static_argnames=("E_apply", "chunk"))
def energy_step_encoder(
    enc_state: EncoderState,
    y_emb,
    feats_b,
    set_b,
    time_b,
    E_apply,
    eparams: FrozenDict,
    L,
    tau,
    chunk: int = 4,
):
    tau = jnp.asarray(tau, jnp.float32)

    def loss_enc(enc_params):
        m = enc_state.apply_fn({"params": enc_params}, feats_b, set_b, time_b)
        cond = _norm_cond(jnp.concatenate([y_emb, m], -1), target_rms=1.0)
        # fast CE-only with margin 0 for encoder — lower variance gradients
        B = L.shape[0]

        def body(ce_sum, i):
            Li = L[i]
            logits = _row_logits_chunked(
                E_apply, eparams, Li, cond, tau, chunk, gumbel_scale=0.0
            )
            ce_i = optax.softmax_cross_entropy_with_integer_labels(
                logits[None, :], jnp.array([i], jnp.int32)
            ).squeeze()
            return ce_sum + ce_i, None

        ce_tot, _ = lax.scan(body, 0.0, jnp.arange(B))
        return ce_tot / jnp.maximum(B, 1)

    loss, grads = jax.value_and_grad(loss_enc)(enc_state.params)
    updates, new_opt = enc_state.tx.update(grads, enc_state.opt_state, enc_state.params)
    new_params = optax.apply_updates(enc_state.params, updates)
    new_state = enc_state.replace(params=new_params, opt_state=new_opt)
    return new_state, loss
