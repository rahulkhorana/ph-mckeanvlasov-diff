# losses_steps.py
from __future__ import annotations
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
import optax
from dataclasses import replace as dc_replace


_F32 = jnp.float32
_I32 = jnp.int32


# -----------------------------
# Sinusoidal time embedding
# -----------------------------
def _time_embed(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """t: (B,) -> (B, dim), JIT-safe (dim is a Python int)."""
    t = jnp.asarray(t, _F32)
    half = dim // 2
    # frequencies in log space [1, 10k]
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half, dtype=_F32))
    ang = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if emb.shape[-1] < dim:
        pad = jnp.zeros((emb.shape[0], dim - emb.shape[-1]), dtype=_F32)
        emb = jnp.concatenate([emb, pad], axis=-1)
    return emb.astype(_F32)


# -----------------------------
# Noise schedules (cosine / linear)
# -----------------------------
def _make_cosine_schedule(
    T: int, s: float = 0.008
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns betas, alphas, alpha_bars of length T (float32).
    """
    steps = jnp.arange(T + 1, dtype=_F32)
    t = steps / jnp.array(T, _F32)
    f = jnp.cos((t + s) / (1.0 + s) * jnp.pi / 2.0) ** 2
    alpha_bars = (f / f[0]).clip(1e-7, 1.0)  # (T+1,)
    ab_t = alpha_bars[1:]
    ab_prev = alpha_bars[:-1]
    betas = (1.0 - (ab_t / ab_prev)).clip(1e-7, 0.999)
    alphas = 1.0 - betas
    return betas.astype(_F32), alphas.astype(_F32), ab_t.astype(_F32)


def _make_linear_schedule(
    T: int, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    betas = jnp.linspace(beta_start, beta_end, T, dtype=_F32).clip(1e-8, 0.999)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0).astype(_F32)
    return betas.astype(_F32), alphas.astype(_F32), alpha_bars.astype(_F32)


# -----------------------------
# State container
# -----------------------------
@struct.dataclass
class DiffusionState:
    # non-pytrees (static / JIT-constant config)
    apply_fn: Callable = struct.field(pytree_node=False)  # unet.apply
    T: int = struct.field(pytree_node=False)  # number of steps
    v_prediction: bool = struct.field(
        pytree_node=False
    )  # True = v-pred, False = eps-pred
    t_embed_dim: int = struct.field(pytree_node=False)  # time embedding dim (e.g., 128)
    ema_decay: float = struct.field(pytree_node=False)  # EMA decay for params
    snr_gamma: float = struct.field(pytree_node=False)  # power for SNR weight
    charbonnier_eps: float = struct.field(pytree_node=False)  # epsilon for Charbonnier

    # pytree params / optimizer
    params: FrozenDict
    ema_params: FrozenDict
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # schedules (pytree)
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray

    # rng for internal use (optional)
    rng: jnp.ndarray

    def replace(self, **updates):
        return dc_replace(self, **updates)


def create_diffusion_state(
    *,
    rng: jnp.ndarray,
    apply_fn: Callable,
    init_params: FrozenDict,
    T: int = 1000,
    lr: float = 2e-4,
    v_prediction: bool = True,
    schedule: str = "cosine",
    ema_decay: float = 0.999,
    snr_gamma: float = 0.5,  # SNR weighting gamma
    charbonnier_eps: float = 1e-3,  # Charbonnier epsilon
) -> DiffusionState:
    if schedule == "cosine":
        betas, alphas, alpha_bars = _make_cosine_schedule(T)
    elif schedule == "linear":
        betas, alphas, alpha_bars = _make_linear_schedule(T)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    tx = optax.adamw(learning_rate=lr, weight_decay=0.0)

    return DiffusionState(
        apply_fn=apply_fn,
        T=int(T),
        v_prediction=bool(v_prediction),
        t_embed_dim=128,  # keep in sync with UNet / models.time_embed
        ema_decay=float(ema_decay),
        snr_gamma=float(snr_gamma),
        charbonnier_eps=float(charbonnier_eps),
        params=init_params,
        ema_params=init_params,
        opt_state=tx.init(init_params),
        tx=tx,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        rng=rng,
    )


# -----------------------------
# Loss helpers
# -----------------------------
def _snr_weight(alpha_bar_t: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    alpha_bar_t: (B,)
    w = (snr^gamma) / (snr^gamma + 1),  snr = alpha_bar / (1 - alpha_bar)
    """
    snr = (alpha_bar_t / jnp.maximum(1.0 - alpha_bar_t, 1e-7)).astype(_F32)
    w = (snr**gamma) / (snr**gamma + 1.0)
    return w.astype(_F32)


def _charbonnier(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return jnp.sqrt(x * x + jnp.array(eps, _F32) ** 2)


def _ema_update(ema_params: FrozenDict, params: FrozenDict, decay: float) -> FrozenDict:
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params
    )


# -----------------------------
# Training step
# -----------------------------
def diffusion_train_step(
    state: DiffusionState,
    x0: jnp.ndarray,  # (B,H,W,K,C)
    rng: jnp.ndarray,  # PRNGKey
    v_pred_flag: bool,  # kept for API compatibility; ignored (state.v_prediction)
    cond_vec: jnp.ndarray,  # (B, D)
) -> Tuple[DiffusionState, jnp.ndarray]:
    """
    One training step with Charbonnier + SNR-weighted loss.
    JIT-safe: all dynamic sizes avoided, config is static on the state.
    """
    del v_pred_flag  # always use state.v_prediction

    B = x0.shape[0]
    T = state.T

    def loss_fn(params):
        # sample t in [0, T-1]
        rng1, rng2 = jax.random.split(rng)
        t_idx = jax.random.randint(rng1, (B,), minval=0, maxval=T, dtype=_I32)  # (B,)
        a_bar = state.alpha_bars[t_idx]  # (B,)
        a = jnp.sqrt(a_bar)  # (B,)
        sig = jnp.sqrt(jnp.maximum(1.0 - a_bar, 1e-8))  # (B,)

        # noise and x_t
        eps = jax.random.normal(rng2, x0.shape, dtype=_F32)
        a_ = a.reshape((B,) + (1,) * (x0.ndim - 1))
        s_ = sig.reshape((B,) + (1,) * (x0.ndim - 1))
        xt = a_ * x0 + s_ * eps

        # time embedding
        t_frac = (t_idx.astype(_F32) + 0.5) / jnp.array(T, _F32)  # (B,)
        t_emb = _time_embed(t_frac, state.t_embed_dim)  # (B, t_dim)

        # model prediction
        pred = state.apply_fn({"params": params}, xt, t_emb, cond_vec).astype(_F32)

        # build training target and score conversion
        if state.v_prediction:
            # v = a * eps - sig * x0
            target = a_ * eps - s_ * x0
            diff = pred - target
        else:
            # eps-pred
            target = eps
            diff = pred - target

        # Charbonnier + SNR weighting
        per_voxel = _charbonnier(diff, state.charbonnier_eps)  # (B, H,W,K,C)
        per_ex = jnp.mean(per_voxel, axis=tuple(range(1, x0.ndim)))  # (B,)
        w = _snr_weight(a_bar, state.snr_gamma)  # (B,)
        loss = jnp.mean(w * per_ex)  # scalar

        return loss

    grads = jax.grad(loss_fn)(state.params)
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_ema = _ema_update(state.ema_params, new_params, state.ema_decay)  # type: ignore

    new_state = state.replace(
        params=new_params,
        ema_params=new_ema,
        opt_state=new_opt,
        rng=jax.random.fold_in(state.rng, 1),
    )

    # Recompute loss for logging (cheap)
    loss_val = loss_fn(new_params)
    return new_state, loss_val
