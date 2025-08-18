# ---------------------------------------------------------------------
# 3D-Volume aware, conditional diffusion + strong energy objective
# - Cosine/linear schedules
# - DiffusionState / EnergyState / EncoderState with .replace()
# - Cond-aware DDPM loss (templated for v-prediction)
# - Symmetric NT-Xent energy loss on E(L, cond) with in-batch negatives
# - Gradient penalty on E wrt volume for stability
# - All jitted with correct static_argnames usage
# - Ready to integrate McKean–Vlasov / bridge (pass temb, cond externally)
# ---------------------------------------------------------------------

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from jax.tree_util import tree_map
from dataclasses import replace as dc_replace


# -------------------------- beta schedules ---------------------------


def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cosine schedule used in improved DDPM. Returns (betas, alphas, alpha_bars)."""
    s = 0.008
    t = jnp.linspace(0.0, 1.0, T + 1)
    alphas_cum = jnp.cos(((t + s) / (1.0 + s)) * jnp.pi * 0.5) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    alpha_bars = alphas_cum[1:]  # (T,)
    alphas = alpha_bars / jnp.concatenate([jnp.array([1.0]), alpha_bars[:-1]])
    betas = 1.0 - alphas
    return (
        betas.astype(jnp.float32),
        alphas.astype(jnp.float32),
        alpha_bars.astype(jnp.float32),
    )


def linear_beta_schedule(
    T: int, start: float = 1e-4, end: float = 2e-2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    betas = jnp.linspace(start, end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


# ------------------------------ states -------------------------------


@struct.dataclass
class DiffusionState:
    apply_fn: Callable
    params: FrozenDict
    opt_state: optax.OptState
    tx: optax.GradientTransformation
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
    apply_fn: Callable
    params: FrozenDict
    opt_state: optax.OptState
    tx: optax.GradientTransformation
    tau: float  # NT-Xent temperature
    gp_lambda: float  # gradient penalty multiplier

    def replace(self, **updates):
        return dc_replace(self, **updates)


@struct.dataclass
class EncoderState:
    apply_fn: Callable
    params: FrozenDict
    opt_state: optax.OptState
    tx: optax.GradientTransformation

    def replace(self, **updates):
        return dc_replace(self, **updates)


# ------------------------- state factories ---------------------------


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
    if schedule == "cosine":
        betas, alphas, alpha_bars = cosine_beta_schedule(T)
    else:
        betas, alphas, alpha_bars = linear_beta_schedule(T)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )

    return DiffusionState(
        apply_fn=apply_fn,
        params=init_params,
        opt_state=tx.init(init_params),
        tx=tx,
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
        params=init_params,
        opt_state=tx.init(init_params),
        tx=tx,
        tau=float(tau),
        gp_lambda=float(gp_lambda),
    )


def create_encoder_state(
    apply_fn: Callable,
    init_params: FrozenDict,
    lr: float = 1e-3,
) -> EncoderState:
    tx = optax.adam(lr)
    return EncoderState(
        apply_fn=apply_fn,
        params=init_params,
        opt_state=tx.init(init_params),
        tx=tx,
    )


# ------------------------ diffusion (cond) ---------------------------


def _ddpm_loss(
    unet_apply: Callable,
    params: FrozenDict,
    rng,
    x0: jnp.ndarray,  # (B,H,W,K,C)  3D volume
    alpha_bars: jnp.ndarray,  # (T,)
    alphas: jnp.ndarray,  # (T,)
    v_prediction: bool,
    temb: jnp.ndarray,  # (B, D_t)      provided by caller (time embed or MV bridge features)
    cond_vec: jnp.ndarray,  # (B, D_cond)   labels+modules embedding
) -> jnp.ndarray:
    """Noise prediction (eps) or v-prediction, conditional on (temb, cond_vec)."""
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)
    t = jax.random.randint(key_t, (B,), 0, alpha_bars.shape[0])

    a_bar_t = alpha_bars[t][:, None, None, None, None]  # (B,1,1,1,1)
    eps = jax.random.normal(key_eps, x0.shape)
    xt = jnp.sqrt(a_bar_t) * x0 + jnp.sqrt(1.0 - a_bar_t) * eps

    # model predicts eps or v depending on training mode
    pred = unet_apply({"params": params}, xt, temb, cond_vec)

    if v_prediction:
        # v = sqrt(alpha_t) * eps - sqrt(1 - alpha_t) * x0
        a_t = alphas[t][:, None, None, None, None]
        v = jnp.sqrt(a_t) * eps - jnp.sqrt(1.0 - a_t) * x0
        loss = jnp.mean((pred - v) ** 2)
    else:
        loss = jnp.mean((pred - eps) ** 2)

    return loss


@partial(jax.jit, static_argnames=("v_prediction",))
def diffusion_train_step(
    state: DiffusionState,
    x0: jnp.ndarray,  # (B,H,W,K,C)
    temb: jnp.ndarray,  # (B,D_t)
    cond_vec: jnp.ndarray,  # (B,D_cond)
    rng,
    v_prediction: bool,
):
    """One SGD step for conditional diffusion with EMA update."""

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
        lambda e, p: state.ema_decay * e + (1.0 - state.ema_decay) * p,
        state.ema_params,
        new_params,
    )

    return state.replace(params=new_params, opt_state=new_opt, ema_params=new_ema), loss


# ---------------------- energy (NT-Xent + GP) -----------------------


def _energy_matrix(
    E_apply: Callable, eparams: FrozenDict, L: jnp.ndarray, cond: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute pairwise energies E_ij = E(L_i, cond_j).
      L:    (B, H, W, K, C)
      cond: (B, D)
    returns: (B, B) with rows i, cols j
    """
    B = L.shape[0]

    def col_energy(mj: jnp.ndarray) -> jnp.ndarray:  # mj: (D,)
        cond_full = jnp.broadcast_to(mj, (B, mj.shape[0]))  # (B,D)
        return E_apply({"params": eparams}, L, cond_full)  # (B,)

    # map over columns (j), then transpose to (B,B) with rows i
    E_cols = jax.vmap(col_energy, in_axes=0)(cond)  # (B, B) with axis0=j
    return E_cols.T


def _rowwise_xent_from_energy(Eij: jnp.ndarray, tau: float) -> jnp.ndarray:
    """
    NT-Xent row-wise: logits = -E/tau. Label for row i is column i.
    Uses stable log-sum-exp with per-row max subtraction.
    """
    logits = -Eij / tau  # (B,B)
    logits = logits - jnp.max(logits, axis=1, keepdims=True)
    labels = jnp.arange(Eij.shape[0], dtype=jnp.int32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss


def energy_ntxent_symmetric(E_apply, eparams, L, cond, tau: float) -> jnp.ndarray:
    """
    Symmetric NT-Xent on energies:
      loss = 0.5 * (row-wise NTXent(E) + col-wise NTXent(E^T))
    Improves signal and avoids collapse/saturation.
    """
    Eij = _energy_matrix(E_apply, eparams, L, cond)  # (B,B)
    loss_row = _rowwise_xent_from_energy(Eij, tau)
    loss_col = _rowwise_xent_from_energy(Eij.T, tau)
    return 0.5 * (loss_row + loss_col)


def energy_grad_penalty(E_apply, eparams, L, cond) -> jnp.ndarray:
    """||∇_L mean_j E(L, cond_j)||^2 averaged over batch."""

    def e_mean(L_):
        # share cond across batch
        return E_apply({"params": eparams}, L_, cond).mean()

    g = jax.grad(e_mean)(L)
    return (g * g).mean()


@jax.jit
def energy_train_step(
    e_state: EnergyState,
    enc_state: EncoderState,
    mods_embed_fn: Callable,  # (enc_params, mods_batch) -> (B,D_cond_mod)
    L: jnp.ndarray,  # (B,H,W,K,C) volumes
    mods_batch,  # python list of ragged module tensors (length B)
    lbl_embed: jnp.ndarray,  # (B,D_lbl) pass label embedding from main
):
    """
    Joint step:
      - Update E params with symmetric NT-Xent + gradient penalty
      - Update encoder params using NT-Xent (no GP on encoder to save compute)
    """
    # --- embed modules with current encoder params ---
    m_emb = mods_embed_fn(enc_state.params, mods_batch)  # (B,D_mod)
    cond = jnp.concatenate([lbl_embed, m_emb], axis=-1)  # (B,D_cond)

    # --- energy params update ---
    def loss_e(p):
        ntx = energy_ntxent_symmetric(e_state.apply_fn, p, L, cond, tau=e_state.tau)
        gp = e_state.gp_lambda * energy_grad_penalty(e_state.apply_fn, p, L, cond)
        return ntx + gp

    lossE, gradsE = jax.value_and_grad(loss_e)(e_state.params)
    updatesE, new_optE = e_state.tx.update(gradsE, e_state.opt_state, e_state.params)
    new_paramsE = optax.apply_updates(e_state.params, updatesE)
    new_e_state = e_state.replace(params=new_paramsE, opt_state=new_optE)

    # --- encoder params update (contrastive only) ---
    def loss_enc(enc_p):
        m_emb2 = mods_embed_fn(enc_p, mods_batch)
        cond2 = jnp.concatenate([lbl_embed, m_emb2], axis=-1)
        ntx2 = energy_ntxent_symmetric(
            e_state.apply_fn, new_paramsE, L, cond2, tau=e_state.tau
        )
        return ntx2

    lossEnc, gradsEnc = jax.value_and_grad(loss_enc)(enc_state.params)
    updatesEnc, new_optEnc = enc_state.tx.update(
        gradsEnc, enc_state.opt_state, enc_state.params
    )
    new_paramsEnc = optax.apply_updates(enc_state.params, updatesEnc)
    new_enc_state = enc_state.replace(params=new_paramsEnc, opt_state=new_optEnc)

    return new_e_state, new_enc_state, (lossE, lossEnc)


# ------------------------- helper embeddings ------------------------


def time_embed_scalar(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    """
    Param-free sinusoidal time embedding that accepts continuous t in [0,1].
    Useful when you move toward McKean–Vlasov bridges where t is continuous.
    """
    t = t_cont.astype(jnp.float32)
    half = dim // 2
    # geometric frequencies
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10000.0), half))
    ang = t[:, None] * (1.0 / freqs[None, :])
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
