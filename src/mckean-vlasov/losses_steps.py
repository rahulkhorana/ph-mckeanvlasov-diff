# losses_steps.py
from typing import Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from dataclasses import replace as dc_replace


# ---------------- Schedules ----------------
def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    steps = jnp.arange(T + 1, dtype=jnp.float32)
    s = 0.008
    alphas_bar = jnp.cos(((steps / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    a_bar_t = alphas_bar[1:]  # length T
    a_bar_tm1 = alphas_bar[:-1]  # length T
    betas = jnp.clip(1.0 - (a_bar_t / a_bar_tm1), 1e-6, 0.999)
    alphas = 1.0 - betas
    return betas, alphas, jnp.cumprod(alphas)


def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    return betas, alphas, jnp.cumprod(alphas)


# ---------------- Train States ----------------
@struct.dataclass
class DiffusionState:
    train: TrainState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    T: int
    v_prediction: bool
    ema_params: FrozenDict
    ema_decay: float


@struct.dataclass
class EnergyState:
    train: TrainState


def create_diffusion_state(
    rng,
    apply_fn: Callable,
    init_params,
    T: int = 1000,
    lr: float = 2e-4,
    v_prediction: bool = True,
    schedule: str = "cosine",
    ema_decay: float = 0.999,
) -> DiffusionState:
    if schedule == "cosine":
        betas, alphas, alpha_bars = cosine_beta_schedule(T)
    else:
        betas, alphas, alpha_bars = linear_beta_schedule(T)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    train = TrainState.create(apply_fn=apply_fn, params=init_params, tx=tx)
    return DiffusionState(
        train=train,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=T,
        v_prediction=v_prediction,
        ema_params=init_params,
        ema_decay=ema_decay,
    )


def create_energy_state(
    apply_fn: Callable, init_params, lr: float = 1e-3
) -> EnergyState:
    tx = optax.adam(lr)
    train = TrainState.create(apply_fn=apply_fn, params=init_params, tx=tx)
    return EnergyState(train=train)


# ---------------- Loss helpers (no tracer booleans) ----------------
# Fixed hyperparams (can be tweaked here without touching main.py)
_CHARB_EPS = 1e-3  # Charbonnier epsilon
_CHARB_ALPHA = 0.3  # mix between MSE (1-a) and Charbonnier (a)
_SNR_GAMMA = 0.5  # exponent for SNR weighting
_SNR_CLIP = 5.0  # clamp SNR to stabilize weights


def _snr_weight(alpha_bar_t: jnp.ndarray) -> jnp.ndarray:
    """Per-sample SNR weight scalar for timestep t (shape (B,))."""
    snr = alpha_bar_t / jnp.clip(1.0 - alpha_bar_t, 1e-8, 1.0)
    snr = jnp.clip(snr, 1e-6, _SNR_CLIP)
    w = jnp.power(snr, _SNR_GAMMA)
    # Normalize to ~[0,1]-ish scale (optional); keep as-is to emphasize late steps
    return w


def _charbonnier(residual: jnp.ndarray, eps: float = _CHARB_EPS) -> jnp.ndarray:
    # mean over all dims
    return jnp.mean(jnp.sqrt(residual * residual + eps * eps))


def _mse(residual: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(residual * residual)


def _ddpm_loss(
    unet_apply: Callable,
    params,
    rng,
    x0: jnp.ndarray,  # (B,H,W,K,C)
    cond_vec: jnp.ndarray,  # (B,D)
    alpha_bars: jnp.ndarray,
    alphas: jnp.ndarray,
    v_prediction: bool,
) -> jnp.ndarray:
    """
    v_prediction: static at trace time (controlled by jit static_argnames)
    Loss = SNR-weighted blend of MSE and Charbonnier on target residual.
    """
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)
    T = alpha_bars.shape[0]
    t = jax.random.randint(key_t, (B,), 0, T)  # (B,)
    t_f = (t.astype(jnp.float32) + 0.5) / float(T)

    eps = jax.random.normal(key_eps, x0.shape)
    a_bar = alpha_bars[t][:, None, None, None, None]  # (B,1,1,1,1)
    sqrt_ab = jnp.sqrt(a_bar)
    sqrt_1ab = jnp.sqrt(1.0 - a_bar)

    xt = sqrt_ab * x0 + sqrt_1ab * eps
    # Time embedding is supplied by caller's UNet; we pass t_f directly as an input embedding there.
    # Here, we emulate what main.py does: UNet expects a precomputed time embedding,
    # but since main.py constructs it, we just follow the same API in main.
    # -> Keep call signature consistent with models.UNet3D_FiLM: (x, t_emb, cond_vec)
    # However, this helper doesn't create t_emb; diffusion_train_step does it before calling _ddpm_loss.
    # To keep API clean here, we will create it here using the same sinusoidal function:
    from models import time_embed

    t_emb = time_embed(t_f, dim=128)

    pred = unet_apply({"params": params}, xt, t_emb, cond_vec)  # eps or v

    if v_prediction:
        # Convert predicted v -> eps_hat following common formulation
        eps_hat = sqrt_ab * pred + sqrt_1ab * xt
        residual = eps - eps_hat
    else:
        residual = eps - pred  # direct eps prediction

    # Per-sample SNR weights
    a_bar_vec = alpha_bars[t]  # (B,)
    w_snr = _snr_weight(a_bar_vec)  # (B,)
    w_snr = w_snr[:, None, None, None, None]  # broadcast to residual

    # Compute both losses (no branching)
    mse_loss = _mse(w_snr * residual)
    charb_loss = _charbonnier(w_snr * residual, eps=_CHARB_EPS)

    # Blend
    loss = (1.0 - _CHARB_ALPHA) * mse_loss + _CHARB_ALPHA * charb_loss
    return loss


# ---------------- Train steps ----------------
@partial(jax.jit, static_argnames=("v_prediction",))
def diffusion_train_step(
    state: DiffusionState,
    batch_vol: jnp.ndarray,
    cond_vec: jnp.ndarray,
    rng,
    v_prediction: bool,
):
    def loss_fn(params):
        return _ddpm_loss(
            state.train.apply_fn,
            params,
            rng,
            batch_vol,
            cond_vec,
            state.alpha_bars,
            state.alphas,
            v_prediction,
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.train.params)
    new_train = state.train.apply_gradients(grads=grads)
    new_ema = optax.incremental_update(
        new_train.params, state.ema_params, 1.0 - state.ema_decay
    )
    new_state = dc_replace(state, train=new_train, ema_params=new_ema)
    return new_state, loss


def energy_contrastive_loss(E_apply, eparams, x, cond_vec, neg_k: int = 4):
    e_pos = E_apply({"params": eparams}, x, cond_vec)  # (B,)
    losses = []
    for k in range(1, neg_k + 1):
        cond_neg = jnp.roll(cond_vec, k, axis=0)
        e_neg = E_apply({"params": eparams}, x, cond_neg)
        losses.append(jnp.logaddexp(0.0, e_pos - e_neg))
    return jnp.mean(jnp.stack(losses, 0))


@jax.jit
def energy_train_step(state: EnergyState, x, cond_vec):
    def loss_fn(params):
        return energy_contrastive_loss(state.train.apply_fn, params, x, cond_vec)

    loss, grads = jax.value_and_grad(loss_fn)(state.train.params)
    new_train = state.train.apply_gradients(grads=grads)
    new_state = dc_replace(state, train=new_train)
    return new_state, loss
