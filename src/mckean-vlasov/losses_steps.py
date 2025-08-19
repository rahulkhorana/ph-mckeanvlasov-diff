# losses_steps.py
from __future__ import annotations

from dataclasses import replace as dc_replace
from functools import partial
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
import optax
from flax import struct
from flax.core import FrozenDict

FloatScalar = Union[float, jnp.ndarray]

# ===================== Noise schedules =====================


def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = 0.008
    t = jnp.linspace(0, 1, T + 1, dtype=jnp.float32)
    f = jnp.cos(((t + s) / (1 + s)) * jnp.pi * 0.5) ** 2
    f = f / f[0]
    alpha_bars = f[1:]
    alphas = alpha_bars / jnp.concatenate(
        [jnp.array([1.0], jnp.float32), alpha_bars[:-1]]
    )
    betas = 1.0 - alphas
    return (
        betas.astype(jnp.float32),
        alphas.astype(jnp.float32),
        alpha_bars.astype(jnp.float32),
    )


def linear_beta_schedule(
    T: int, start=1e-4, end=2e-2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    betas = jnp.linspace(start, end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


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
    tau: float = 0.07
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
    schedule: str = "cosine",
    ema_decay: float = 0.999,
) -> DiffusionState:
    betas, alphas, alpha_bars = (
        cosine_beta_schedule(T) if schedule == "cosine" else linear_beta_schedule(T)
    )
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
    lr: float = 1e-3,
    tau: float = 0.07,
    gp_lambda: float = 1e-4,
) -> EnergyState:
    tx = optax.adam(lr)
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
    tx = optax.adam(lr)
    return EncoderState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
    )


# ===================== Helpers for diffusion =====================


def _sinusoidal_time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(
        jnp.linspace(jnp.log(1.0), jnp.log(10_000.0), half, dtype=jnp.float32)
    )
    angles = t_cont[:, None] * (1.0 / freqs[None, :])
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return jax.nn.relu(emb)


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


# ===================== Diffusion loss & step =====================


def _ddpm_loss(
    unet_apply: Callable,
    params: FrozenDict,
    rng,
    x0: jnp.ndarray,  # (B,H,W,KS,C) standardized
    alpha_bars: jnp.ndarray,  # (T,)
    alphas: jnp.ndarray,  # (T,)
    v_prediction: bool,
    cond_vec: jnp.ndarray,  # (B, D)
    t_dim: int = 128,
    lambda_x0: float = 0.1,
    x0_weight_pow: float = 1.5,
) -> jnp.ndarray:
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)

    t_int = jax.random.randint(
        key_t, shape=(B,), minval=0, maxval=alpha_bars.shape[0], dtype=jnp.int32
    )
    a_bar = alpha_bars[t_int].astype(jnp.float32)

    eps = jax.random.normal(key_eps, x0.shape, dtype=jnp.float32)
    sa = jnp.sqrt(a_bar)[..., None, None, None, None]
    sb = jnp.sqrt(1.0 - a_bar)[..., None, None, None, None]
    xt = sa * x0 + sb * eps

    t_cont = (t_int.astype(jnp.float32) + 0.5) / float(alpha_bars.shape[0])
    temb = _sinusoidal_time_embed(t_cont, dim=t_dim)

    pred = unet_apply({"params": params}, xt, temb, cond_vec)  # predict v or eps

    if v_prediction:
        target = sa * eps - sb * x0
    else:
        target = eps

    main_loss = jnp.mean(_charbonnier(pred - target))

    x0_hat, _ = _reconstruct_x0_eps_from_v(xt, pred, a_bar)
    w_x0 = jnp.power(1.0 - a_bar, x0_weight_pow)[..., None, None, None, None]
    aux_loss = jnp.mean(w_x0 * _charbonnier(x0_hat - x0))

    return main_loss + lambda_x0 * aux_loss


@partial(jax.jit, static_argnames=("v_prediction",))
def diffusion_train_step(
    state: DiffusionState,
    x0: jnp.ndarray,  # (B,H,W,KS,C) standardized
    rng,  # PRNGKey
    v_prediction: bool,  # bool
    cond_vec: jnp.ndarray,  # (B,D)
):
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


# ======================= ENERGY: subsampled, JAX-friendly =======================


def _row_loss_subsampled(
    E_apply, params, Li, cond_vec, tau: FloatScalar, margin: float, neg_k: int, i: int
):
    """
    Fixed-size ring sampling: J = neg_k+1 (static). We mask invalid negatives when B < J.
    Positive is at rel=0 (self), negatives are i+1..i+neg_k (mod B).
    """
    B = cond_vec.shape[0]  # Python int (static w.r.t. jit)
    J = int(neg_k) + 1  # static
    rel = jnp.arange(J, dtype=jnp.int32)  # (J,)
    idx = (i + rel) % max(B, 1)  # (J,)
    cond_sub = cond_vec[idx]  # (J,D)
    LiJ = jnp.broadcast_to(Li, (J,) + Li.shape)  # (J,H,W,KS,C)

    tau = jnp.asarray(tau, jnp.float32)
    e = E_apply({"params": params}, LiJ, cond_sub)  # (J,)
    logits = -e / jnp.maximum(tau, 1e-6)  # (J,)

    # mask: keep positive (rel=0) always; valid negatives only when rel < B
    valid = jnp.concatenate([jnp.array([True]), (rel[1:] < B)])  # shape (J,)
    logits = jnp.where(valid, logits, -jnp.inf)

    # CE with label 0 (positive)
    logits_b = jnp.expand_dims(logits, 0)  # type: ignore
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits_b, jnp.array([i], jnp.int32)
    ).squeeze()

    # hinge on energies for valid negatives
    E_row = -tau * logits
    pos = E_row[0]
    diff = E_row - pos
    neg_mask = jnp.concatenate(
        [jnp.array([0.0], jnp.float32), (rel[1:] < B).astype(jnp.float32)]
    )
    hinge = jnp.maximum(0.0, margin - diff) * neg_mask
    denom = jnp.maximum(jnp.sum(neg_mask), 1.0)
    hinge = jnp.sum(hinge) / denom
    return ce, hinge


def energy_total_loss_subsampled(
    E_apply,
    params,
    L,  # (B,H,W,KS,C)
    cond_vec,  # (B,D)
    tau: FloatScalar,
    gp_lambda: FloatScalar = 0.0,
    margin: float = 0.1,
    hinge_weight: float = 0.25,
    neg_k: int = 7,  # fixed #negatives per row
    gp_subset: int = 4,  # GP on first gp_subset rows
) -> jnp.ndarray:
    B = L.shape[0]  # Python int
    tau = jnp.asarray(tau, jnp.float32)
    gp_lambda = jnp.asarray(gp_lambda, jnp.float32)

    def scan_body(carry, i):
        Li = L[i]
        ce_i, hinge_i = _row_loss_subsampled(
            E_apply, params, Li, cond_vec, tau, margin, neg_k, i
        )
        return (carry[0] + ce_i, carry[1] + hinge_i), None

    (ce_sum, hinge_sum), _ = lax.scan(scan_body, (0.0, 0.0), jnp.arange(B))
    ce = ce_sum / jnp.maximum(B, 1)
    hinge = hinge_sum / jnp.maximum(B, 1)

    def gp_branch(_):
        M = min(B, gp_subset)
        L_sub = L[:M]
        C_sub = cond_vec[:M]

        def e_mean(L_, C_):
            return E_apply({"params": params}, L_, C_).mean()

        g = jax.grad(e_mean, argnums=0)(L_sub, C_sub)
        return jnp.mean(jnp.square(g))

    gp = lax.cond(
        gp_lambda > 0.0, gp_branch, lambda _: jnp.array(0.0, jnp.float32), operand=None
    )
    return ce + hinge_weight * hinge + gp_lambda * gp


@partial(jax.jit, static_argnames=("E_apply", "neg_k", "gp_subset"))
def energy_step_E(
    e_state: EnergyState,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    cond_vec: jnp.ndarray,  # (B,D)
    E_apply,
    neg_k: int = 7,
    gp_subset: int = 2,
):
    tau = jnp.asarray(e_state.tau, jnp.float32)
    gp_lambda = jnp.asarray(e_state.gp_lambda, jnp.float32)

    def loss_e(params):
        return energy_total_loss_subsampled(
            E_apply,
            params,
            L,
            cond_vec,
            tau=tau,
            gp_lambda=gp_lambda,
            margin=0.1,
            hinge_weight=0.25,
            neg_k=neg_k,
            gp_subset=gp_subset,
        )

    loss, grads = jax.value_and_grad(loss_e)(e_state.params)
    updates, new_opt = e_state.tx.update(grads, e_state.opt_state, e_state.params)
    new_params = optax.apply_updates(e_state.params, updates)
    new_state = e_state.replace(params=new_params, opt_state=new_opt)
    return new_state, loss


@partial(jax.jit, static_argnames=("E_apply", "neg_k"))
def energy_step_encoder(
    enc_state: EncoderState,
    y_emb: jnp.ndarray,  # (B, y_dim)
    feats_b: jnp.ndarray,  # (B, T_max, S_max, F)
    set_b: jnp.ndarray,  # (B, T_max, S_max, 1)
    time_b: jnp.ndarray,  # (B, T_max, 1)
    E_apply,
    eparams: FrozenDict,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    tau: FloatScalar,
    neg_k: int = 7,
):
    tau = jnp.asarray(tau, jnp.float32)

    def loss_enc(enc_params):
        m_emb = enc_state.apply_fn(
            {"params": enc_params}, feats_b, set_b, time_b
        )  # (B,d)
        cond_vec = jnp.concatenate([y_emb, m_emb], axis=-1)
        B = L.shape[0]

        def body(carry, i):
            Li = L[i]
            ce_i, _ = _row_loss_subsampled(
                E_apply, eparams, Li, cond_vec, tau, margin=0.0, neg_k=neg_k, i=i
            )
            return carry + ce_i, None

        ce_sum, _ = lax.scan(body, 0.0, jnp.arange(B))
        return ce_sum / jnp.maximum(B, 1)

    loss, grads = jax.value_and_grad(loss_enc)(enc_state.params)
    updates, new_opt = enc_state.tx.update(grads, enc_state.opt_state, enc_state.params)
    new_params = optax.apply_updates(enc_state.params, updates)
    new_state = enc_state.replace(params=new_params, opt_state=new_opt)
    return new_state, loss
