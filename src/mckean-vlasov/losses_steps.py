# losses_steps.py
import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training.train_state import TrainState
from jax import jit, value_and_grad
import optax


def time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """Sinusoidal time embedding. t_cont: (B,) in [0,1]. Tracer-safe."""
    t_cont = t_cont.astype(jnp.float32)
    half = dim // 2
    freqs = jnp.exp(
        jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half, dtype=jnp.float32)
    )
    angles = t_cont[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Nichol & Dhariwal cosine schedule."""
    s = 0.008
    t = jnp.arange(T + 1, dtype=jnp.float32) / T
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * (jnp.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alpha_bars = alphas_cumprod[1:]  # length T
    betas = 1.0 - (
        alpha_bars
        / jnp.concatenate([jnp.array([1.0], dtype=jnp.float32), alpha_bars[:-1]])
    )
    betas = jnp.clip(betas, 1e-6, 0.999)
    alphas = 1.0 - betas
    return betas, alphas, alpha_bars


def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


class DiffusionState(TrainState):
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    T: int
    v_prediction: bool = (
        False  # if True, predict v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0
    )


class EnergyState(TrainState):
    pass


def ddpm_loss(unet_apply, params, rng, x0, T, alpha_bars, v_prediction: bool):
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)
    t = jax.random.randint(key_t, (B,), 0, T)
    t_cont = (t.astype(jnp.float32) + 0.5) / T

    eps = jax.random.normal(key_eps, x0.shape)
    a_bar = alpha_bars[t][:, None, None, None]
    sqrt_ab = jnp.sqrt(a_bar)
    sqrt_1ab = jnp.sqrt(1.0 - a_bar)
    xt = sqrt_ab * x0 + sqrt_1ab * eps

    temb = time_embed(t_cont, dim=128)
    pred = unet_apply({"params": params}, xt, temb)

    mask = jnp.asarray(v_prediction, dtype=jnp.bool_)  # shape=(), tracers OK
    eps_from_v = (pred + sqrt_1ab * xt) / jnp.clip(sqrt_ab, 1e-8)
    eps_hat = jnp.where(mask, eps_from_v, pred)

    return jnp.mean((eps_hat - eps) ** 2)


@jit
def diffusion_train_step(state, batch_imgs, rng):
    def loss_fn(params):
        return ddpm_loss(
            state.apply_fn,
            params,
            rng,
            batch_imgs,
            state.T,
            state.alpha_bars,
            state.v_prediction,
        )

    loss, grads = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def energy_contrastive_loss(E_apply, eparams, L, m_emb, neg_k: int = 4):
    """InfoNCE-style: matched (L,m_emb) vs rolled negatives."""
    e_pos = E_apply({"params": eparams}, L, m_emb)  # (B,)
    losses = []
    for k in range(1, neg_k + 1):
        m_neg = jnp.roll(m_emb, k, axis=0)
        e_neg = E_apply({"params": eparams}, L, m_neg)
        losses.append(jnp.logaddexp(0.0, e_pos - e_neg))
    return jnp.mean(jnp.stack(losses, 0))


@jit
def energy_train_step(est, L, m_emb):
    def loss_fn(eparams):
        return energy_contrastive_loss(est.apply_fn, eparams, L, m_emb)

    loss, grads = value_and_grad(loss_fn)(est.params)
    new_est = est.apply_gradients(grads=grads)
    return new_est, loss


def create_diffusion_state(
    rng, model_apply, params, T=1000, schedule="cosine", lr=2e-4, v_prediction=False
):
    if schedule == "cosine":
        betas, alphas, alpha_bars = cosine_beta_schedule(T)
    else:
        betas, alphas, alpha_bars = linear_beta_schedule(T)
    tx = optax.adam(lr)
    return DiffusionState.create(
        apply_fn=model_apply,
        params=params,
        tx=tx,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=T,
        v_prediction=v_prediction,
    )


def create_energy_state(model_apply, params, lr=1e-3):
    tx = optax.adam(lr)
    return EnergyState.create(
        apply_fn=model_apply,
        params=params,
        tx=tx,
    )
