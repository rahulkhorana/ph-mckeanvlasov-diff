from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import flax
from flax import struct
from functools import partial


# -------- Time embed (reuse from models) --------
from models import time_embed as time_embed_fn

# -------- Schedules --------


def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Nichol & Dhariwal's cosine schedule
    s = 0.008
    t = jnp.linspace(0, T, T + 1, dtype=jnp.float32) / T
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alpha_bars = alphas_cumprod[1:]  # length T
    alphas = alpha_bars / jnp.concatenate(
        [jnp.array([1.0], jnp.float32), alpha_bars[:-1]], axis=0
    )
    betas = 1.0 - alphas
    return betas, alphas, alpha_bars


def _cosine_schedule(T):
    import numpy as np

    s = 0.008
    t = np.linspace(0, T, T + 1, dtype=np.float32) / T
    alphas_cum = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    alpha_bars = jnp.array(alphas_cum[1:])  # length T
    alphas = alpha_bars[1:] / alpha_bars[:-1]
    alphas = jnp.concatenate([alphas[:1], alphas], axis=0)
    betas = 1.0 - alphas
    return jnp.clip(betas, 1e-5, 0.999)


def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)
    return betas, alphas, alpha_bars


# -------- Diffusion state with EMA --------


@struct.dataclass
class DiffusionState(TrainState):
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    T: int
    v_prediction: bool


def create_diffusion_state(
    rng, apply_fn, init_params, T=1000, lr=2e-4, v_prediction=True, schedule="cosine"
):
    if schedule == "cosine":
        betas = _cosine_schedule(T)
    else:
        betas = jnp.linspace(1e-4, 2e-2, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    return DiffusionState.create(
        apply_fn=apply_fn,
        params=init_params,
        tx=tx,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=int(T),
        v_prediction=bool(v_prediction),
    )


# -------- MinSNR weighting (optional) --------


def _minsnr_weight(alpha_bar_t: jnp.ndarray, gamma: float = 5.0):
    snr = alpha_bar_t / jnp.clip(1.0 - alpha_bar_t, 1e-8, None)
    w = jnp.minimum(snr, jnp.array(gamma, dtype=jnp.float32)) / jnp.clip(
        snr, 1e-8, None
    )
    return w


# -------- DDPM loss with v-pred --------


def _targets(x0, eps, a_bar_t, v_prediction: bool):
    if v_prediction:
        # v = sqrt(a_bar) * eps - sqrt(1 - a_bar) * x0
        return (
            jnp.sqrt(a_bar_t)[:, None, None, None, None] * eps
            - jnp.sqrt(1.0 - a_bar_t)[:, None, None, None, None] * x0
        )
    else:
        return eps


def _ddpm_loss(
    unet_apply,
    params,
    rng,
    x0,  # (B,H,W,K,C)
    alpha_bars,  # (T,)
    alphas,  # (T,)  (kept for completeness if you use it later)
    v_prediction: bool,  # Python bool (static at trace time)
    m_dim: int = 256,
):
    B = x0.shape[0]
    T = alpha_bars.shape[0]  # derive T from schedule (array shape), not from state

    key_t, key_eps = jax.random.split(rng)
    t = jax.random.randint(key_t, (B,), 0, T)  # (B,)
    eps = jax.random.normal(key_eps, x0.shape)  # (B,H,W,K,C)

    a_bar = alpha_bars[t][..., None, None, None, None]  # (B,1,1,1,1)
    xt = jnp.sqrt(a_bar) * x0 + jnp.sqrt(1.0 - a_bar) * eps

    # continuous time in [0,1]
    t_cont = (t.astype(jnp.float32) + 0.5) / jnp.float32(T)
    temb = time_embed_fn(t_cont, dim=128)  # (B,128)

    # unconditional diffusion: zero modules embedding
    m_emb = jnp.zeros((B, m_dim), x0.dtype)

    pred = unet_apply({"params": params}, xt, temb, m_emb)  # (B,H,W,K,C)

    if v_prediction:
        # v-pred target using alpha_bar
        target = jnp.sqrt(a_bar) * eps - jnp.sqrt(1.0 - a_bar) * x0
    else:
        target = eps

    return jnp.mean((pred - target) ** 2)


@partial(jax.jit, static_argnames=("v_prediction",))
def diffusion_train_step(state, batch_imgs, rng, v_prediction: bool):
    """state must carry: apply_fn, params, alpha_bars, alphas"""

    def loss_fn(params):
        return _ddpm_loss(
            state.apply_fn,
            params,
            rng,
            batch_imgs,
            state.alpha_bars,
            state.alphas,
            v_prediction,
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


# -------- Energy training (unchanged API) --------


@struct.dataclass
class EnergyState(TrainState):
    pass


def create_energy_state(apply_fn, params, lr: float = 1e-3) -> EnergyState:
    tx = optax.adam(lr)
    return EnergyState.create(apply_fn=apply_fn, params=params, tx=tx)


@jax.jit
def energy_train_step(est: EnergyState, E_apply, imgs, m_emb):
    def loss_fn(params):
        # simple NCE-ish: push matched lower than rolled negatives
        e_pos = E_apply({"params": params}, imgs, m_emb)  # (B,)
        loss = 0.0
        K = 4
        for k in range(1, K + 1):
            e_neg = E_apply({"params": params}, imgs, jnp.roll(m_emb, k, axis=0))
            loss += jnp.mean(jnp.logaddexp(0.0, e_pos - e_neg))
        return loss / K

    loss, grads = jax.value_and_grad(loss_fn)(est.params)
    est = est.apply_gradients(grads=grads)
    return est, loss
