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
USE_BF16_FORWARD = True

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
