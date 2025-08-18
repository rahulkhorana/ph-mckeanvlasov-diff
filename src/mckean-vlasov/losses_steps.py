# losses_steps.py
from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from jax.tree_util import tree_map
from dataclasses import replace as dc_replace


# ----- beta schedules -----
def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = 0.008
    t = jnp.linspace(0, 1, T + 1)
    alphas_cum = jnp.cos(((t + s) / (1 + s)) * jnp.pi * 0.5) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    alpha_bars = alphas_cum[1:]
    alphas = alpha_bars / jnp.concatenate([jnp.array([1.0]), alpha_bars[:-1]])
    betas = 1.0 - alphas
    return (
        betas.astype(jnp.float32),
        alphas.astype(jnp.float32),
        alpha_bars.astype(jnp.float32),
    )


def linear_beta_schedule(T: int, start=1e-4, end=2e-2):
    betas = jnp.linspace(start, end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


# ----- states -----
@struct.dataclass
class DiffusionState:
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    params: FrozenDict
    opt_state: optax.OptState

    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray
    T: int
    v_prediction: bool
    ema_params: FrozenDict
    ema_decay: float

    def replace(self, **updates):
        return dc_replace(self, **updates)


@struct.dataclass
class EnergyState:
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    params: FrozenDict
    opt_state: optax.OptState

    tau: float
    gp_lambda: float

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


def create_diffusion_state(
    rng,
    apply_fn,
    init_params,
    T=1000,
    lr=2e-4,
    v_prediction=True,
    schedule="cosine",
    ema_decay=0.999,
):
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


def create_energy_state(apply_fn, init_params, lr=1e-3, tau=0.07, gp_lambda=1e-4):
    tx = optax.adam(lr)
    return EnergyState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
        tau=float(tau),
        gp_lambda=float(gp_lambda),
    )


def create_encoder_state(apply_fn, init_params, lr=1e-3):
    tx = optax.adam(lr)
    return EncoderState(
        apply_fn=apply_fn, tx=tx, params=init_params, opt_state=tx.init(init_params)
    )


# ----- diffusion loss -----
def _ddpm_loss(
    unet_apply: Callable,
    params: FrozenDict,
    rng,
    x0: jnp.ndarray,  # (B,H,W,KS,3)
    alpha_bars: jnp.ndarray,
    alphas: jnp.ndarray,
    v_prediction: bool,
    temb: jnp.ndarray,  # (B,t_dim)
    cond_vec: jnp.ndarray,  # (B,D)
) -> jnp.ndarray:
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)
    t_int = jax.random.randint(key_t, (B,), 0, alpha_bars.shape[0])
    a_bar = alpha_bars[t_int][:, None, None, None, None]
    eps = jax.random.normal(key_eps, x0.shape)

    xt = jnp.sqrt(a_bar) * x0 + jnp.sqrt(1.0 - a_bar) * eps
    pred = unet_apply({"params": params}, xt, temb, cond_vec)

    if v_prediction:
        sqrt_ab = jnp.sqrt(a_bar)
        sqrt_1ab = jnp.sqrt(1.0 - a_bar)
        v = sqrt_ab * eps - sqrt_1ab * x0
        loss = jnp.mean((pred - v) ** 2)
    else:
        loss = jnp.mean((pred - eps) ** 2)
    return loss


@partial(jax.jit, static_argnames=("v_prediction",))
def diffusion_train_step(
    state: DiffusionState,
    x0: jnp.ndarray,
    rng,
    v_prediction: bool,
    temb: jnp.ndarray,
    cond_vec: jnp.ndarray,
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
            temb,
            cond_vec,
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_ema = tree_map(
        lambda e, q: state.ema_decay * e + (1.0 - state.ema_decay) * q,
        state.ema_params,
        new_params,
    )
    return state.replace(params=new_params, opt_state=new_opt, ema_params=new_ema), loss


# ===== energy loss: K-negatives NT-Xent (O(B·K) instead of O(B²)) =====


def _energy_logits_row(E_apply, eparams, Li, cond_mat, neg_idx_i):
    """
    For one sample i:
      Li:          (H,W,KS,3)
      cond_mat:    (B,D)
      neg_idx_i:   (K,)
    Build candidates conds: [cond_i] + cond[neg_idx_i] -> (K+1,D),
    replicate Li -> (K+1,H,W,KS,3), evaluate energies -> logits_i = -E/tau elsewhere.
    """
    pos = cond_mat[0]  # we'll pass cond_mat so that index 0 is pos via a gather trick
    # But we want cond_i specifically, so we gather it outside and pass here as first row.

    # This function expects 'cond_mat' already stacked as (K+1, D) = [pos; negs]
    # So just broadcast Li to match that K+1.
    K1 = cond_mat.shape[0]
    Li_rep = jnp.broadcast_to(Li[None, ...], (K1,) + Li.shape)  # (K+1, H,W,KS,3)
    E = E_apply({"params": eparams}, Li_rep, cond_mat)  # (K+1,)
    return E  # caller does scaling / CE


def energy_ntxent_k(E_apply, eparams, L, cond_mat, neg_idx, tau: float):
    """
    K-negative NT-Xent:
      L:        (B,H,W,KS,3)
      cond_mat: (B,D)
      neg_idx:  (B,K) integer indices into cond_mat (negatives for each i)
    returns scalar loss
    """
    B = L.shape[0]
    K = neg_idx.shape[1]

    def per_i(i):
        # gather positive and negatives for this i
        pos = cond_mat[i]  # (D,)
        negs = jnp.take(cond_mat, neg_idx[i], axis=0)  # (K,D)
        cand = jnp.concatenate([pos[None, :], negs], axis=0)  # (K+1, D)

        Ei = _energy_logits_row(E_apply, eparams, L[i], cand, neg_idx[i])  # (K+1,)
        logits = -Ei / tau
        # label is 0 (the positive is first)
        loss_i = optax.softmax_cross_entropy_with_integer_labels(
            logits[None, :], jnp.array([0], dtype=jnp.int32)
        )
        return loss_i.squeeze()

    losses = jax.vmap(per_i)(jnp.arange(B, dtype=jnp.int32))
    return losses.mean()


def energy_grad_penalty_diag(E_apply, eparams, L, cond_mat):
    """
    Gradient penalty on the positive diagonal pairs only:
      mean_i E(L_i, cond_i)
    """

    def e_mean_diag(L_):
        return E_apply({"params": eparams}, L_, cond_mat).mean()

    g = jax.grad(e_mean_diag)(L)  # same shapes as L
    return (g * g).mean()


@jax.jit
def energy_step_E(
    e_state: EnergyState,
    L: jnp.ndarray,  # (B,H,W,KS,3)
    cond_mat: jnp.ndarray,  # (B,D)
    neg_idx: jnp.ndarray,  # (B,K) int32
):
    """JIT-safe update of the energy network using K-negatives NT-Xent."""

    def loss_e(eparams):
        ce = energy_ntxent_k(
            e_state.apply_fn, eparams, L, cond_mat, neg_idx, tau=e_state.tau
        )
        gp = e_state.gp_lambda * energy_grad_penalty_diag(
            e_state.apply_fn, eparams, L, cond_mat
        )
        return ce + gp

    lossE, gradsE = jax.value_and_grad(loss_e)(e_state.params)
    updatesE, new_optE = e_state.tx.update(gradsE, e_state.opt_state, e_state.params)
    new_paramsE = optax.apply_updates(e_state.params, updatesE)
    new_e = e_state.replace(params=new_paramsE, opt_state=new_optE)
    return new_e, lossE


def energy_step_encoder(
    enc_state: EncoderState,
    e_apply: Callable,
    eparams: FrozenDict,
    mods_embed_fn,  # (enc_params, mods_batch) -> (B, Dm)
    mods_batch,  # ragged list on host
    y_emb: jnp.ndarray,  # (B, Dy)
    L: jnp.ndarray,  # (B,H,W,KS,3)
    neg_idx: jnp.ndarray,  # (B,K)
    tau: float,
):
    """
    Host-side encoder update (ragged mods). Recomputes the conditioner with current enc_params,
    then uses the same K-negatives structure to define the loss.
    """

    def loss_enc(enc_params):
        m_emb2 = mods_embed_fn(enc_params, mods_batch)  # (B,Dm)
        cond2 = jnp.concatenate([y_emb, m_emb2], axis=-1)  # (B,Dy+Dm)
        ce = energy_ntxent_k(e_apply, eparams, L, cond2, neg_idx, tau=tau)
        return ce

    lossEnc, gradsEnc = jax.value_and_grad(loss_enc)(enc_state.params)
    updatesEnc, new_optEnc = enc_state.tx.update(
        gradsEnc, enc_state.opt_state, enc_state.params
    )
    new_paramsEnc = optax.apply_updates(enc_state.params, updatesEnc)
    new_enc = enc_state.replace(params=new_paramsEnc, opt_state=new_optEnc)
    return new_enc, lossEnc
