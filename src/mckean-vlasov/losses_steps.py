# losses_steps.py
from __future__ import annotations

from functools import partial
from dataclasses import replace as dc_replace
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
import optax
from flax import struct
from flax.core import FrozenDict

FloatScalar = Union[float, jnp.ndarray]
_F32 = jnp.float32
_EPS = 1e-6

# ===================== Noise schedules =====================


def cosine_beta_schedule(T: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = 0.008
    t = jnp.linspace(0, 1, T + 1, dtype=_F32)
    f = jnp.cos(((t + s) / (1 + s)) * jnp.pi * 0.5) ** 2
    f = f / f[0]
    alpha_bars = f[1:]
    alphas = alpha_bars / jnp.concatenate([jnp.array([1.0], _F32), alpha_bars[:-1]])
    betas = 1.0 - alphas
    return betas.astype(_F32), alphas.astype(_F32), alpha_bars.astype(_F32)


def linear_beta_schedule(
    T: int, start=1e-4, end=2e-2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    betas = jnp.linspace(start, end, T, dtype=_F32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


# ===================== States =====================


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
    tau: float = 0.10
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


# ===================== Builders =====================


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
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
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
    lr: float = 3e-4,
    tau: float = 0.10,
    gp_lambda: float = 1e-4,
) -> EnergyState:
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
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
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    return EncoderState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
    )


# ===================== Diffusion: Min-SNR v-objective =====================


def _sinusoidal_time_embed(t_cont: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10_000.0), half, dtype=_F32))
    angles = t_cont[:, None] * (1.0 / freqs[None, :])
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def _charbonnier(x: jnp.ndarray, eps: float = 1e-6, alpha: float = 0.5) -> jnp.ndarray:
    return jnp.power(x * x + eps, alpha)


def _reconstruct_x0_eps_from_v(xt: jnp.ndarray, v: jnp.ndarray, a_bar: jnp.ndarray):
    sa = jnp.sqrt(a_bar)[..., None, None, None, None]
    sb = jnp.sqrt(1.0 - a_bar)[..., None, None, None, None]
    x0_hat = sa * xt - sb * v
    eps_hat = sb * xt + sa * v
    return x0_hat, eps_hat


def _minsnr_weight(a_bar: jnp.ndarray, gamma: float = 5.0) -> jnp.ndarray:
    snr = a_bar / jnp.clip(1.0 - a_bar, 1e-8, None)
    w = jnp.minimum(snr, jnp.asarray(gamma, _F32)) / (snr + 1.0)
    return w


def _ddpm_loss(
    unet_apply: Callable,
    params: FrozenDict,
    rng,
    x0: jnp.ndarray,
    alpha_bars: jnp.ndarray,
    v_prediction: bool,
    cond_vec: jnp.ndarray,
) -> jnp.ndarray:
    x0 = x0.astype(_F32)
    cond_vec = cond_vec.astype(_F32)
    B = x0.shape[0]
    key_t, key_eps = jax.random.split(rng)
    T = int(alpha_bars.shape[0])
    t_int = jax.random.randint(key_t, (B,), 0, T, dtype=jnp.int32)
    a_bar = alpha_bars[t_int].astype(_F32)
    eps = jax.random.normal(key_eps, x0.shape, dtype=_F32)
    sa = jnp.sqrt(a_bar)[..., None, None, None, None]
    sb = jnp.sqrt(1.0 - a_bar)[..., None, None, None, None]
    xt = sa * x0 + sb * eps
    t_cont = (t_int.astype(_F32) + 0.5) / float(T)
    temb = _sinusoidal_time_embed(t_cont, dim=128)
    pred = unet_apply({"params": params}, xt, temb, cond_vec).astype(_F32)
    target = sa * eps - sb * x0 if v_prediction else eps
    resid = pred - target
    per = _charbonnier(resid).reshape(B, -1).mean(-1)
    w = _minsnr_weight(a_bar, gamma=5.0)
    main_loss = (w * per).mean()
    x0_hat, _ = _reconstruct_x0_eps_from_v(xt, pred, a_bar)
    x0_resid = x0_hat - x0
    w_x0 = jnp.power(1.0 - a_bar, 1.25)
    aux_loss = (w_x0 * _charbonnier(x0_resid).reshape(B, -1).mean(-1)).mean()
    total_loss = main_loss + 0.1 * aux_loss
    # MODIFIED: Convert any NaN/inf loss to 0.0 to prevent gradient poisoning.
    return jnp.nan_to_num(total_loss, nan=-1)


@partial(jax.jit, static_argnames=("v_prediction",))
def diffusion_train_step(
    state: DiffusionState,
    x0: jnp.ndarray,
    rng,
    v_prediction: bool,
    cond_vec: jnp.ndarray,
):
    loss_fn = lambda p: _ddpm_loss(
        state.apply_fn, p, rng, x0, state.alpha_bars, v_prediction, cond_vec
    )
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state, loss, grads


def apply_diffusion_updates(state: DiffusionState, grads):
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_ema = tree_map(
        lambda e, q: state.ema_decay * e + (1.0 - state.ema_decay) * q,
        state.ema_params,
        new_params,
    )
    return state.replace(params=new_params, opt_state=new_opt, ema_params=new_ema)


# ======================= ENERGY: Debiased InfoNCE + Ranking =======================


def _row_pass_debiased(
    E_apply,
    params,
    Li,
    cond_vec,
    i_idx: int,
    tau: FloatScalar,
    margin: float,
    chunk: int,
):
    B, D = cond_vec.shape
    tau = jnp.asarray(tau, _F32)
    cpos = lax.dynamic_slice(cond_vec, (i_idx, 0), (1, D))
    E_pos = E_apply({"params": params}, Li[None, ...], cpos).astype(_F32)[0]
    nseg = (B + chunk - 1) // chunk
    B_pad, pad_n = nseg * chunk, nseg * chunk - B
    C_pad = jnp.pad(cond_vec, ((0, pad_n), (0, 0)))
    valid_pad = jnp.concatenate([jnp.ones((B,), bool), jnp.zeros((pad_n,), bool)])
    init_vals = (
        jnp.array(-jnp.inf, _F32),
        jnp.array(0.0, _F32),
        jnp.array(0.0, _F32),
        jnp.array(0.0, _F32),
        jnp.array(0.0, _F32),
    )

    def body(carry, s):
        lse_all, rank_sum, s_neg, s2_neg, c_neg = carry
        j0 = s * chunk
        seg = lax.dynamic_slice(C_pad, (j0, 0), (chunk, D))
        vmask = lax.dynamic_slice(valid_pad, (j0,), (chunk,))
        is_pos = (jnp.arange(chunk, dtype=jnp.int32) + j0 == i_idx) & vmask
        LiB = jnp.broadcast_to(Li, (chunk,) + Li.shape)
        e_seg = E_apply({"params": params}, LiB, seg).astype(_F32)
        dE = e_seg - E_pos
        logits = -dE / jnp.maximum(tau, _EPS)
        logits_masked = jnp.where(vmask, logits, -jnp.inf)
        lse_all = jnp.logaddexp(lse_all, jax.nn.logsumexp(logits_masked))  # type: ignore
        neg, wneg = vmask & (~is_pos), (vmask & (~is_pos)).astype(_F32)
        rank_seg = jax.nn.softplus((margin - dE) / jnp.maximum(tau, _EPS)) * wneg
        rank_sum += jnp.sum(rank_seg)
        s_neg += jnp.sum(dE * wneg)
        s2_neg += jnp.sum((dE**2) * wneg)
        c_neg += jnp.sum(wneg)
        return (lse_all, rank_sum, s_neg, s2_neg, c_neg), None

    (lse_all, rank_sum, s_neg, s2_neg, c_neg), _ = lax.scan(
        body, init_vals, jnp.arange(nseg)
    )

    neg_lse = lse_all + jnp.log1p(-jnp.exp(-jnp.clip(lse_all, -20.0, 20.0)) + _EPS)

    rank_mean = rank_sum / jnp.maximum(c_neg, 1.0)
    mean_neg = s_neg / jnp.maximum(c_neg, 1.0)
    var_neg = (s2_neg / jnp.maximum(c_neg, 1.0)) - mean_neg**2
    return neg_lse, rank_mean, var_neg


def _gp_subset(E_apply, params, L, cond_vec, gp_subset: int):
    M = int(min(L.shape[0], gp_subset))
    e_mean = lambda L_, C_: E_apply({"params": params}, L_, C_).mean()
    g = jax.grad(e_mean, argnums=0)(L[:M], cond_vec[:M])
    return jnp.mean(jnp.square(g))


def energy_total_loss_chunked(
    E_apply,
    params,
    L,
    cond_vec,
    tau: FloatScalar,
    gp_lambda: FloatScalar,
    chunk: int,
    gp_subset: int,
):
    L = L.astype(_F32)
    cond_vec = cond_vec.astype(_F32)  # Sanitize cond_vec
    tau, gp_lambda = jnp.asarray(tau, _F32), jnp.asarray(gp_lambda, _F32)
    B = L.shape[0]

    def row_body(carry, i):
        neg_lse, rank_mean, var_neg = _row_pass_debiased(
            E_apply, params, L[i], cond_vec, i, tau, 0.10, chunk
        )
        ce_i = jax.nn.softplus(neg_lse)
        var_pen = jnp.maximum(0.0, 1e-3 - var_neg)
        return carry + ce_i + 0.5 * rank_mean + var_pen, None

    loss_sum, _ = lax.scan(row_body, jnp.array(0.0, _F32), jnp.arange(B))
    base = loss_sum / jnp.maximum(jnp.array(B, _F32), 1.0)

    def mean_diag():
        nseg, B_pad, pad_n = (
            (B + chunk - 1) // chunk,
            ((B + chunk - 1) // chunk) * chunk,
            ((B + chunk - 1) // chunk) * chunk - B,
        )
        L_pad, C_pad = jnp.pad(
            L, ((0, pad_n), (0, 0), (0, 0), (0, 0), (0, 0))
        ), jnp.pad(cond_vec, ((0, pad_n), (0, 0)))
        mask = jnp.concatenate([jnp.ones((B,), _F32), jnp.zeros((pad_n,), _F32)])
        body = lambda c, s: (
            (
                c[0]
                + jnp.sum(
                    E_apply(
                        {"params": params},
                        lax.dynamic_slice(
                            L_pad, (s * chunk, 0, 0, 0, 0), (chunk,) + L.shape[1:]
                        ),
                        lax.dynamic_slice(
                            C_pad, (s * chunk, 0), (chunk, C_pad.shape[1])
                        ),
                    ).astype(_F32)
                    * lax.dynamic_slice(mask, (s * chunk,), (chunk,))
                ),
                c[1] + jnp.sum(lax.dynamic_slice(mask, (s * chunk,), (chunk,))),
            ),
            None,
        )
        (sum_e, sum_m), _ = lax.scan(
            body, (jnp.array(0.0, _F32), jnp.array(0.0, _F32)), jnp.arange(nseg)
        )
        return sum_e / jnp.maximum(sum_m, 1.0)

    center = 1e-4 * (mean_diag() ** 2)
    gp = lax.cond(
        gp_lambda > 0.0,
        lambda _: _gp_subset(E_apply, params, L, cond_vec, gp_subset),
        lambda _: jnp.array(0.0, _F32),
        None,
    )
    total_loss = base + center + gp_lambda * gp
    # MODIFIED: Convert any NaN/inf loss to 0.0 to prevent gradient poisoning.
    return jnp.nan_to_num(total_loss)


@partial(jax.jit, static_argnames=("E_apply", "chunk", "gp_subset"))
def energy_step_E(
    e_state: EnergyState,
    L: jnp.ndarray,
    cond_vec: jnp.ndarray,
    E_apply,
    chunk: int,
    gp_subset: int,
):
    loss_fn = lambda p: energy_total_loss_chunked(
        E_apply, p, L, cond_vec, e_state.tau, e_state.gp_lambda, chunk, gp_subset
    )
    loss, grads = jax.value_and_grad(loss_fn)(e_state.params)
    return e_state, loss, grads


def apply_energy_updates_E(e_state: EnergyState, grads):
    updates, new_opt = e_state.tx.update(grads, e_state.opt_state, e_state.params)
    new_params = optax.apply_updates(e_state.params, updates)
    return e_state.replace(params=new_params, opt_state=new_opt)


def _encoder_loss(
    enc_apply,
    enc_params,
    y_emb,
    feats_b,
    set_b,
    time_b,
    E_apply,
    eparams,
    L,
    tau,
    chunk,
):
    m_emb = enc_apply({"params": enc_params}, feats_b, set_b, time_b).astype(_F32)
    cond_all = jnp.concatenate(
        [y_emb.astype(_F32), jnp.nan_to_num(m_emb)], axis=-1
    )  # Sanitize m_emb
    B = L.shape[0]

    def body(sum_ce, i):
        neg_lse, _, _ = _row_pass_debiased(
            E_apply, eparams, L[i], cond_all, i, tau, 0.0, chunk
        )
        return sum_ce + jax.nn.softplus(neg_lse), None

    ce_sum, _ = lax.scan(body, jnp.array(0.0, _F32), jnp.arange(B))
    total_loss = ce_sum / jnp.maximum(jnp.array(B, _F32), 1.0)
    # MODIFIED: Convert any NaN/inf loss to 0.0 to prevent gradient poisoning.
    return jnp.nan_to_num(total_loss)


@partial(jax.jit, static_argnames=("E_apply", "chunk"))
def energy_step_encoder(
    enc_state: EncoderState,
    y_emb,
    feats_b,
    set_b,
    time_b,
    E_apply,
    eparams,
    L,
    tau,
    chunk,
):
    loss_fn = lambda p: _encoder_loss(
        enc_state.apply_fn,
        p,
        y_emb,
        feats_b,
        set_b,
        time_b,
        E_apply,
        eparams,
        L,
        tau,
        chunk,
    )
    loss, grads = jax.value_and_grad(loss_fn)(enc_state.params)
    return enc_state, loss, grads


def apply_encoder_updates(enc_state: EncoderState, grads):
    updates, new_opt = enc_state.tx.update(grads, enc_state.opt_state, enc_state.params)
    new_params = optax.apply_updates(enc_state.params, updates)
    return enc_state.replace(params=new_params, opt_state=new_opt)
