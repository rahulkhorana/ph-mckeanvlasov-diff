from __future__ import annotations
from typing import Callable, Tuple
from dataclasses import replace as dc_replace

import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from flax.core import FrozenDict
import optax
from jax.tree_util import tree_map

# ------------------ dtypes ------------------
_F32 = jnp.float32
_I32 = jnp.int32


# ------------------ cosine schedule ------------------
def _cosine_alpha_bars(T: int, s: float = 0.008) -> jnp.ndarray:
    steps = jnp.arange(T + 1, dtype=_F32)
    f = jnp.cos(((steps / T + s) / (1.0 + s)) * jnp.pi * 0.5) ** 2
    f = f / f[0]
    alpha_bars = f[1:]
    alpha_bars = jnp.clip(alpha_bars, 1e-6, 1.0)
    return alpha_bars.astype(_F32)


def _alphas_from_alpha_bars(alpha_bars: jnp.ndarray) -> jnp.ndarray:
    prev = jnp.concatenate([jnp.array([1.0], _F32), alpha_bars[:-1]], axis=0)
    alphas = jnp.sqrt(alpha_bars / prev)
    alphas = jnp.clip(alphas, 1e-6, 1.0)
    return alphas


def _betas_from_alphas(alphas: jnp.ndarray) -> jnp.ndarray:
    betas = 1.0 - (alphas**2)
    betas = jnp.clip(betas, 1e-8, 0.999)
    return betas


# ------------------ time embedding (sinusoidal) ------------------
def _positional_embedding_sin(x: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half, dtype=_F32))
    ang = x[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if emb.shape[-1] < dim:
        pad = jnp.zeros((emb.shape[0], dim - emb.shape[-1]), dtype=_F32)
        emb = jnp.concatenate([emb, pad], axis=-1)
    return emb.astype(_F32)


# ------------------ Charbonnier loss ------------------
def _charbonnier(x: jnp.ndarray, eps: float = 1e-3) -> jnp.ndarray:
    eps32 = jnp.array(eps, _F32)
    return jnp.sqrt(x * x + eps32 * eps32)


# ------------------ SNR utilities ------------------
def _snr_from_alpha_bar(alpha_bar_t: jnp.ndarray) -> jnp.ndarray:
    return alpha_bar_t / jnp.maximum(1.0 - alpha_bar_t, jnp.array(1e-6, _F32))


def _snr_weight(alpha_bar_t: jnp.ndarray, snr_clip: float) -> jnp.ndarray:
    snr = _snr_from_alpha_bar(alpha_bar_t)
    return jnp.minimum(snr, jnp.array(snr_clip, _F32))


# ------------------ Diffusion state ------------------
@struct.dataclass
class DiffusionState:
    # non-pytree
    apply_fn: Callable = struct.field(pytree_node=False)  # unet.apply
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # pytree
    params: FrozenDict
    ema_params: FrozenDict
    opt_state: optax.OptState

    # schedule
    betas: jnp.ndarray  # (T,)
    alphas: jnp.ndarray  # (T,)
    alpha_bars: jnp.ndarray  # (T,)

    # config
    v_prediction: jnp.ndarray  # () bool-array
    ema_decay: jnp.ndarray  # () f32
    snr_clip: jnp.ndarray  # () f32
    use_snr_weight: jnp.ndarray  # () bool-array
    charbonnier_eps: jnp.ndarray  # () f32
    t_embed_dim: jnp.ndarray  # () i32

    def replace(self, **updates):
        return dc_replace(self, **updates)


def create_diffusion_state(
    *,
    rng,
    apply_fn: Callable,
    init_params: FrozenDict,
    T: int,
    lr: float,
    v_prediction: bool,
    schedule: str = "cosine",
    ema_decay: float = 0.999,
    snr_clip: float = 5.0,
    use_snr_weight: bool = True,
    charbonnier_eps: float = 1e-3,
    t_embed_dim: int = 128,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
) -> DiffusionState:
    if schedule != "cosine":
        raise ValueError("Only 'cosine' schedule is implemented in this file.")

    alpha_bars = _cosine_alpha_bars(T)
    alphas = _alphas_from_alpha_bars(alpha_bars)
    betas = _betas_from_alphas(alphas)

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay),
    )

    return DiffusionState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        ema_params=init_params,
        opt_state=tx.init(init_params),
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        v_prediction=jnp.array(bool(v_prediction)),
        ema_decay=jnp.array(ema_decay, _F32),
        snr_clip=jnp.array(snr_clip, _F32),
        use_snr_weight=jnp.array(bool(use_snr_weight)),
        charbonnier_eps=jnp.array(charbonnier_eps, _F32),
        t_embed_dim=jnp.array(int(t_embed_dim), _I32),
    )


def _gather_t(arr: jnp.ndarray, t_idx: jnp.ndarray) -> jnp.ndarray:
    return arr[t_idx]


@jax.jit
def diffusion_train_step(
    state: DiffusionState,
    x0: jnp.ndarray,  # (B,H,W,KS,C)
    rng: jnp.ndarray,
    v_prediction_flag: bool,  # unused; kept for API compatibility
    cond_vec: jnp.ndarray,  # (B, cond_dim)
) -> Tuple[DiffusionState, jnp.ndarray]:
    B = x0.shape[0]
    T = state.alpha_bars.shape[0]

    rng, k_t, k_eps = jax.random.split(rng, 3)
    t_idx = jax.random.randint(k_t, (B,), 0, T, dtype=_I32)  # [0..T-1]
    alpha_bar_t = _gather_t(state.alpha_bars, t_idx)  # (B,)
    a = jnp.sqrt(alpha_bar_t)[..., None, None, None, None]
    s = jnp.sqrt(jnp.maximum(1.0 - alpha_bar_t, jnp.array(1e-6, _F32)))[
        ..., None, None, None, None
    ]

    eps = jax.random.normal(k_eps, shape=x0.shape, dtype=_F32)
    xt = a * x0 + s * eps

    t_frac = (t_idx.astype(_F32) + 0.5) / jnp.array(T, _F32)
    t_emb = _positional_embedding_sin(t_frac, int(state.t_embed_dim))

    def _loss_with_params(p):
        pred = state.apply_fn({"params": p}, xt, t_emb, cond_vec).astype(_F32)
        if bool(state.v_prediction):
            v_tgt = a * eps - s * x0
            diff = pred - v_tgt
        else:
            diff = pred - eps

        char = _charbonnier(diff, float(state.charbonnier_eps))
        per_sample = jnp.mean(char, axis=tuple(range(1, char.ndim)))  # (B,)

        if bool(state.use_snr_weight):
            w = _snr_weight(alpha_bar_t, float(state.snr_clip))
            per_sample = per_sample * w

        return jnp.mean(per_sample).astype(_F32)

    loss, grads = jax.value_and_grad(_loss_with_params)(state.params)

    updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    ema = float(state.ema_decay)
    new_ema_params = tree_map(
        lambda e, p: ema * e + (1.0 - ema) * p, state.ema_params, new_params
    )

    new_state = state.replace(
        params=new_params, ema_params=new_ema_params, opt_state=new_opt
    )
    return new_state, loss
