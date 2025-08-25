# losses_steps.py — Denoised guidance loss and unified train step
from typing import Callable, Tuple, Any
import jax, jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import optax
from models import time_embed
from functools import partial
from dataclasses import replace as dc_replace


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


# --------- states & state creation ---------
class TrainStateWithEMA(TrainState):
    ema_params: FrozenDict


@struct.dataclass
class FullTrainState:
    unet_state: TrainStateWithEMA
    guidance_state: TrainStateWithEMA
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    T: int
    ema_decay: float

    def replace(self, **updates):
        return dc_replace(self, **updates)


def create_full_train_state(
    rng,
    unet_apply_fn: Callable,
    unet_params,
    guidance_apply_fn: Callable,
    guidance_params,
    total_steps: int,
    T: int = 1000,
    lr: float = 2e-4,
    lr_guidance: float = 1e-5,
    schedule: str = "cosine",
    ema_decay: float = 0.999,
    warmup_steps: int = 500,
) -> FullTrainState:
    betas, alphas, alpha_bars = cosine_beta_schedule(T)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup_steps, total_steps, 1e-7
    )
    lr_guidance_schedule = optax.warmup_cosine_decay_schedule(
        0.0, lr_guidance, warmup_steps, total_steps, 1e-7
    )

    tx_unet = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule))
    tx_guidance = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(lr_guidance_schedule)
    )

    unet_state = TrainStateWithEMA.create(
        apply_fn=unet_apply_fn, params=unet_params, tx=tx_unet, ema_params=unet_params
    )
    guidance_state = TrainStateWithEMA.create(
        apply_fn=guidance_apply_fn,
        params=guidance_params,
        tx=tx_guidance,
        ema_params=guidance_params,
    )

    return FullTrainState(
        unet_state=unet_state,
        guidance_state=guidance_state,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=T,
        ema_decay=ema_decay,
    )  # Correctly store ema_decay


# --------- losses ---------
def _diffusion_loss(unet_apply, params, rng, x0, cond_vec, alpha_bars, v_pred: bool):
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)
    T = alpha_bars.shape[0]
    t = jax.random.randint(key_t, (B,), 0, T)
    t_cont = (t.astype(jnp.float32) + 0.5) / float(T)
    eps = jax.random.normal(key_eps, x0.shape)
    a_bar = alpha_bars[t][:, None, None, None, None]
    sqrt_ab, sqrt_1ab = jnp.sqrt(a_bar), jnp.sqrt(1.0 - a_bar)
    xt = sqrt_ab * x0 + sqrt_1ab * eps
    temb = time_embed(t_cont, dim=128)
    prediction = unet_apply({"params": params}, xt, temb, cond_vec)
    target = (sqrt_ab * eps - sqrt_1ab * x0) if v_pred else eps
    return jnp.mean((target - prediction) ** 2), (xt, temb)


def _denoised_guidance_loss(guidance_apply, params, xt, temb, cond, x0):
    x0_pred = guidance_apply({"params": params}, xt, temb, cond)
    return jnp.mean((x0 - x0_pred) ** 2)


# --------- Combined Training Step ---------
def train_step(
    state: FullTrainState, batch: dict, rng, v_pred: bool, guidance_loss_weight: float
):
    vol, cond = batch["vol"], batch["cond"]

    (diff_loss, (xt, temb)), unet_grads = jax.value_and_grad(
        lambda p: _diffusion_loss(
            state.unet_state.apply_fn, p, rng, vol, cond, state.alpha_bars, v_pred
        ),
        has_aux=True,
    )(state.unet_state.params)

    if guidance_loss_weight > 0:
        guidance_loss, guidance_grads = jax.value_and_grad(
            lambda p: _denoised_guidance_loss(
                state.guidance_state.apply_fn, p, xt, temb, cond, vol
            )
        )(state.guidance_state.params)
        total_loss = diff_loss + guidance_loss_weight * guidance_loss
    else:
        guidance_loss = 0.0
        guidance_grads = jax.tree_util.tree_map(
            jnp.zeros_like, state.guidance_state.params
        )
        total_loss = diff_loss

    new_unet_state = state.unet_state.apply_gradients(grads=unet_grads)
    new_guidance_state = state.guidance_state.apply_gradients(grads=guidance_grads)

    # Correctly access ema_decay from the FullTrainState
    new_unet_state = new_unet_state.replace(
        ema_params=optax.incremental_update(
            new_unet_state.params, new_unet_state.ema_params, state.ema_decay
        )
    )
    new_guidance_state = new_guidance_state.replace(
        ema_params=optax.incremental_update(
            new_guidance_state.params, new_guidance_state.ema_params, state.ema_decay
        )
    )

    new_full_state = state.replace(
        unet_state=new_unet_state, guidance_state=new_guidance_state
    )
    metrics = {
        "total_loss": total_loss,
        "diff_loss": diff_loss,
        "guidance_loss": guidance_loss,
    }
    return new_full_state, metrics
