# losses_steps.py
import math
from flax import struct

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.tree_util import tree_map

import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState
import flax.linen as nn


# --------- time embedding (functional) ----------


def time_embed(t: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """
    Sinusoidal embedding; t is float in [0,1] shape (B,).
    Returns (B, dim).
    """
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half))
    angles = t[:, None] / freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


# --------- schedules ----------


def cosine_beta_schedule(T: int, s: float = 0.008) -> jnp.ndarray:
    steps = jnp.arange(T + 1, dtype=jnp.float32)
    f = (jnp.cos(((steps / T) + s) / (1.0 + s) * jnp.pi * 0.5)) ** 2
    f = f / f[0]
    betas = 1.0 - (f[1:] / f[:-1])
    return jnp.clip(betas, 1e-5, 0.999)


def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2) -> jnp.ndarray:
    return jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)


# --------- states ----------


@struct.dataclass
class DiffusionState(TrainState):
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    T: int
    v_prediction: bool
    ema_params: FrozenDict
    ema_decay: float  # <-- put this in the state


@struct.dataclass
class EnergyState(TrainState):
    pass


def create_diffusion_state(
    rng,
    unet,
    input_shape,
    lr: float = 2e-4,
    T: int = 1000,
    schedule: str = "cosine",
    v_prediction: bool = True,
    ema_decay: float = 0.999,
):
    # make betas/alphas/alpha_bars exactly as before...
    betas = cosine_beta_schedule(T) if schedule == "cosine" else linear_beta_schedule(T)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)

    B = input_shape[0]
    x_d = jnp.zeros(input_shape, jnp.float32)
    t_d = jnp.zeros((B,), jnp.float32)
    temb = time_embed(t_d, dim=128)
    params = unet.init(rng, x_d, temb)["params"]

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    state = DiffusionState.create(
        apply_fn=unet.apply,
        params=params,
        tx=tx,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        T=T,
        v_prediction=v_prediction,
        ema_params=params,
        ema_decay=ema_decay,  # <-- pass into state here
    )
    return state


@jit
def diffusion_train_step(state: DiffusionState, batch_imgs, rng):
    def loss_fn(p):
        return _ddpm_loss(
            state.apply_fn, p, rng, batch_imgs, state.alpha_bars, state.v_prediction
        )

    loss, grads = value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)

    d = state.ema_decay
    from jax.tree_util import tree_map

    new_ema = tree_map(
        lambda e, n: e * d + (1.0 - d) * n, state.ema_params, new_state.params
    )
    new_state = new_state.replace(ema_params=new_ema)
    return new_state, loss


def create_energy_state(apply_fn, params, lr: float = 1e-3) -> EnergyState:
    tx = optax.adam(lr)
    return EnergyState.create(apply_fn=apply_fn, params=params, tx=tx)


# --------- diffusion loss & train step ----------


def _ddpm_loss(apply_fn, params, rng, x0, alpha_bars, v_prediction):
    T_py = alpha_bars.shape[0]
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)

    t_int = jax.random.randint(key_t, (B,), 0, T_py, dtype=jnp.int32)
    eps = jax.random.normal(key_eps, x0.shape)

    a_t = alpha_bars[t_int]  # (B,)
    sqrt_ab = jnp.sqrt(a_t)[:, None, None, None]
    sqrt_1ab = jnp.sqrt(1.0 - a_t)[:, None, None, None]
    xt = sqrt_ab * x0 + sqrt_1ab * eps

    # time embedding
    T_f = jnp.asarray(T_py, jnp.float32)
    t_cont = (t_int.astype(jnp.float32) + 0.5) / jnp.maximum(T_f, 1.0)
    temb = time_embed(t_cont, dim=128)

    pred = apply_fn({"params": params}, xt, temb)  # eps or v

    # compute both losses, then select without Python branching
    eps_hat = (pred + sqrt_1ab * xt) / jnp.clip(sqrt_ab, 1e-8)  # v->eps
    loss_v = jnp.mean((eps - eps_hat) ** 2)  # for v-pred
    loss_eps = jnp.mean((eps - pred) ** 2)  # for eps-pred

    vp = jnp.asarray(v_prediction, dtype=jnp.float32)  # scalar 0/1
    loss = vp * loss_v + (1.0 - vp) * loss_eps
    return loss


# --------- energy loss & train step ----------


def energy_contrastive_loss(E_apply, eparams, L, m_emb, neg_k=4):
    """
    L: (B,H,W,C), m_emb: (B,D)
    """
    e_pos = E_apply({"params": eparams}, L, m_emb)  # (B,)
    losses = []
    for k in range(1, neg_k + 1):
        e_neg = E_apply({"params": eparams}, L, jnp.roll(m_emb, k, axis=0))
        # push matched energy lower: log(1+exp(e_pos - e_neg))
        losses.append(jnp.logaddexp(0.0, e_pos - e_neg))
    return jnp.mean(jnp.stack(losses, 0))


@jit
def energy_train_step(est: EnergyState, L, m_emb):
    def loss_fn(eparams):
        return energy_contrastive_loss(est.apply_fn, eparams, L, m_emb)

    loss, grads = value_and_grad(loss_fn)(est.params)
    est = est.apply_gradients(grads=grads)
    return est, loss
