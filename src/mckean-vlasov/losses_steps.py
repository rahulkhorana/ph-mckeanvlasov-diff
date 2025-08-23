# losses_steps.py — schedules, states, ε-pred loss and steps
from typing import Callable, Tuple
import jax, jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from dataclasses import replace as dc_replace
import optax
from models import time_embed
from functools import partial


# --------- schedules ---------
def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    steps = jnp.arange(T + 1, dtype=jnp.float32)
    s = 0.008
    alphas_bar = jnp.cos(((steps / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    a_bar_t = alphas_bar[1:]
    a_bar_tm1 = alphas_bar[:-1]
    betas = jnp.clip(1.0 - (a_bar_t / a_bar_tm1), 1e-6, 0.999)
    alphas = 1.0 - betas
    return betas, alphas, jnp.cumprod(alphas)


def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    return betas, alphas, jnp.cumprod(alphas)


# --------- states ---------
@struct.dataclass
class DiffusionState:
    train: TrainState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    T: int
    ema_params: FrozenDict
    ema_decay: float


def create_diffusion_state(
    rng,
    apply_fn: Callable,
    init_params,
    T: int = 1000,
    lr: float = 2e-4,
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
        ema_params=init_params,
        ema_decay=ema_decay,
    )


# --------- ε-prediction loss ---------
def _ddpm_loss(unet_apply, params, rng, x0, cond_vec, alpha_bars):
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)
    T = alpha_bars.shape[0]
    t = jax.random.randint(key_t, (B,), 0, T)
    t_cont = (t.astype(jnp.float32) + 0.5) / float(T)

    eps = jax.random.normal(key_eps, x0.shape)
    a_bar = alpha_bars[t][:, None, None, None, None]
    sqrt_ab = jnp.sqrt(a_bar)
    sqrt_1ab = jnp.sqrt(1.0 - a_bar)
    xt = sqrt_ab * x0 + sqrt_1ab * eps

    temb = time_embed(t_cont, dim=128)
    pred_eps = unet_apply({"params": params}, xt, temb, cond_vec)  # predict ε
    loss = jnp.mean((eps - pred_eps) ** 2)
    return loss


@partial(jax.jit, static_argnames=())
def diffusion_train_step(state: DiffusionState, batch_vol, cond_vec, rng):
    def loss_fn(params):
        return _ddpm_loss(
            state.train.apply_fn, params, rng, batch_vol, cond_vec, state.alpha_bars
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.train.params)
    new_train = state.train.apply_gradients(grads=grads)
    new_ema = optax.incremental_update(
        new_train.params, state.ema_params, 1.0 - state.ema_decay
    )
    new_state = dc_replace(state, train=new_train, ema_params=new_ema)
    return new_state, loss
