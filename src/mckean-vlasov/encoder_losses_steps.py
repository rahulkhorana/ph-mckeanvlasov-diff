# encoder_losses_steps.py
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from flax.core import FrozenDict
import optax

_F32 = jnp.float32
_I32 = jnp.int32


# ===================== Encoder state (Flax-compatible) =====================


@struct.dataclass
class EncoderState:
    # non-pytrees
    apply_fn: Callable = struct.field(
        pytree_node=False
    )  # enc_apply({"params": p}, feats, set_b, time_b) -> (B, d)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # pytrees
    params: FrozenDict
    opt_state: optax.OptState

    # telemetry (EMA of cond norm, optional)
    cond_norm_ema: jnp.ndarray = struct.field(pytree_node=True)  # () f32

    def replace(self, **updates):
        return dc_replace(self, **updates)


def create_encoder_state(
    apply_fn: Callable,
    init_params: FrozenDict,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> EncoderState:
    """Build encoder optimizer state."""
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    return EncoderState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
        cond_norm_ema=jnp.array(1.0, _F32),
    )


# ===================== Small JIT-safe helpers =====================


@jax.jit
def _stop_grad(x: jnp.ndarray) -> jnp.ndarray:
    return lax.stop_gradient(x)


@jax.jit
def _ema(prev: jnp.ndarray, value: jnp.ndarray, decay: float = 0.99) -> jnp.ndarray:
    return decay * prev + (1.0 - decay) * value


# DO NOT jit this (tx is a Python object)
def _adamw_update(tx, params, opt_state, grads):
    updates, new_opt = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt


@jax.jit
def _slice_Li(L: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    # L: (B,H,W,KS,C) -> Li: (H,W,KS,C)
    Li = lax.dynamic_slice(L, (i, 0, 0, 0, 0), (1,) + L.shape[1:])
    return Li[0]


@jax.jit
def _dynamic_slice_row(x: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    """
    Return x[i:i+1, :] using dynamic start index and static slice size.
    Avoids tracer->python int conversion entirely.
    """
    return lax.dynamic_slice_in_dim(x, start_index=i, slice_size=1, axis=0)


@jax.jit
def _safe_tau(tau: float) -> jnp.ndarray:
    return jnp.maximum(jnp.array(tau, _F32), jnp.array(1e-6, _F32))


@jax.jit
def _norm2(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(jnp.sum(x * x), jnp.array(1e-12, _F32)))


# ===================== Row-wise negatives from queue (no BxB) =====================


def _row_neg_logits_vs_queue(
    E_apply: Callable,  # energy apply
    eparams: FrozenDict,  # energy params (frozen)
    Li: jnp.ndarray,  # (H,W,KS,C)
    queue: jnp.ndarray,  # (Q,D)   negatives as cond vectors
    valid_count: jnp.ndarray,  # () i32  number of valid rows in queue
    tau: float,
    chunk: int,
    rng: jnp.ndarray | None,
    gumbel_scale: float,
) -> jnp.ndarray:
    """
    Returns (Q,) logits for negatives (masked beyond valid_count). Uses fixed-size chunking
    over Q; requires Q % chunk == 0.
    """
    Q, D = queue.shape
    assert Q % chunk == 0, "queue size Q must be a multiple of `chunk` (static)."
    nseg = Q // chunk

    logits0 = jnp.full((Q,), -jnp.inf, _F32)

    def body(curr, s):
        j0 = s * chunk
        cond_seg = lax.dynamic_slice(queue, (j0, 0), (chunk, D))  # (chunk, D)
        idxs = j0 + jnp.arange(chunk, dtype=_I32)
        mask_seg = idxs < valid_count  # (chunk,)

        LiB = jnp.broadcast_to(Li, (chunk,) + Li.shape)  # (chunk,H,W,KS,C)
        e_seg = E_apply({"params": eparams}, LiB, cond_seg).astype(_F32)  # (chunk,)
        logits_seg = -e_seg / _safe_tau(tau)

        # optional gumbel noise
        if rng is not None and gumbel_scale != 0.0:
            subkey = jax.random.fold_in(rng, s)  # tracer-safe
            u = jax.random.uniform(
                subkey, (chunk,), minval=1e-6, maxval=1.0, dtype=_F32
            )
            g = -jnp.log(-jnp.log(u))
            logits_seg = logits_seg + jnp.array(gumbel_scale, _F32) * g

        logits_seg = jnp.where(mask_seg, logits_seg, -jnp.inf)
        curr = lax.dynamic_update_slice_in_dim(curr, logits_seg, j0, axis=0)  # type: ignore
        return curr, None

    logits, _ = lax.scan(body, logits0, jnp.arange(nseg, dtype=_I32))
    return logits  # (Q,)


# ===================== Per-row non-saturating contrastive loss =====================


def _row_loss(
    E_apply: Callable,
    eparams: FrozenDict,
    Li: jnp.ndarray,  # (H,W,KS,C)
    cond_i: jnp.ndarray,  # (1,D)
    queue: jnp.ndarray,  # (Q,D)
    valid_count: jnp.ndarray,  # () i32
    *,
    tau: float,
    k_top: int,
    chunk: int,
    rng: jnp.ndarray | None,
    gumbel_scale: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Non-saturating top-k logistic loss:
      loss_i = softplus( logsumexp( topk(neg_logits) - pos_logit ) )
    Returns (loss_i, mean_neg_energy, std_neg_energy)
    """
    # positive
    e_pos = E_apply({"params": eparams}, Li[None, ...], cond_i).reshape(())  # ()
    logit_pos = -e_pos / _safe_tau(tau)

    # negatives (logits)
    neg_logits = _row_neg_logits_vs_queue(
        E_apply, eparams, Li, queue, valid_count, tau, chunk, rng, gumbel_scale
    )  # (Q,)

    # top-k selection (largest logits are hardest negatives)
    k = max(1, min(int(k_top), neg_logits.shape[0]))
    topk_vals, _ = lax.top_k(neg_logits, k)  # (k,)
    lse = jax.nn.logsumexp(topk_vals - logit_pos)  # scalar
    loss_i = jax.nn.softplus(lse)

    # energy stats for telemetry (map logits back to energy with tau)
    mask = jnp.isfinite(neg_logits)
    neg_e = -jnp.where(mask, neg_logits, -jnp.inf) * _safe_tau(tau)
    # replace -inf by e_pos to avoid NaN stats
    neg_e = jnp.where(jnp.isfinite(neg_e), neg_e, e_pos)
    mean_e = jnp.mean(neg_e)
    std_e = jnp.sqrt(jnp.maximum(jnp.var(neg_e), jnp.array(1e-12, _F32)))
    return loss_i, mean_e, std_e


# ===================== Public step: update ENCODER only =====================


def energy_step_encoder(
    enc_state: EncoderState,
    *,
    # frozen energy scorer
    E_apply: Callable,
    eparams: FrozenDict,
    tau: float = 0.07,
    k_top: int = 32,
    gumbel_scale: float = 0.2,
    # data
    L: jnp.ndarray,  # (B,H,W,KS,C)
    y_emb: jnp.ndarray,  # (B, d_y)
    feats_b: jnp.ndarray,  # (B, T, S, F)
    set_b: jnp.ndarray,  # (B, T, S, 1)
    time_b: jnp.ndarray,  # (B, T, 1)
    # negatives memory (from energy trainer)
    queue: jnp.ndarray,  # (Q, D)
    queue_count: jnp.ndarray,  # () i32
    # rng and chunking
    rng: jnp.ndarray,
    chunk: int = 64,  # must divide Q
) -> Tuple[EncoderState, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Train the encoder against a *frozen* energy model using a MoCo-style memory queue.
    - No in-batch negatives are formed.
    - VRAM-friendly: per-row chunked scoring over the queue.
    - The queue is NOT mutated here; it's owned by the energy trainer.

    Returns:
      new_encoder_state, loss (scalar), metrics dict
    """
    B = L.shape[0]
    Q, D = queue.shape
    assert Q % chunk == 0, "queue size Q must be a multiple of `chunk`."

    # Build current cond vectors from encoder
    m_emb = enc_state.apply_fn(
        {"params": enc_state.params}, feats_b, set_b, time_b
    )  # (B, d_m)
    cond_vec = jnp.concatenate([y_emb, m_emb], axis=-1).astype(_F32)  # (B, D)
    assert cond_vec.shape[1] == D, "cond_vec dim must match queue dim."

    def row_body(carry, i):
        row_rng = jax.random.fold_in(rng, i)
        Li = _slice_Li(L, jnp.array(i, _I32))  # (H,W,KS,C)
        cond_i = _dynamic_slice_row(cond_vec, jnp.array(i, _I32))  # (1, D)

        li, mean_e, std_e = _row_loss(
            E_apply,
            eparams,
            Li,
            cond_i,
            queue,
            queue_count,
            tau=tau,
            k_top=k_top,
            chunk=chunk,
            rng=row_rng,
            gumbel_scale=gumbel_scale,
        )
        carry = (
            carry[0] + li,
            carry[1] + mean_e,
            carry[2] + std_e,
        )
        return carry, None

    (sum_loss, sum_mean_e, sum_std_e), _ = lax.scan(
        row_body,
        (jnp.array(0.0, _F32), jnp.array(0.0, _F32), jnp.array(0.0, _F32)),
        jnp.arange(B, dtype=_I32),
    )

    loss = sum_loss / jnp.array(B, _F32)
    mean_e = sum_mean_e / jnp.array(B, _F32)
    std_e = sum_std_e / jnp.array(B, _F32)

    # grads wrt encoder params only
    def loss_only(p):
        m_emb_local = enc_state.apply_fn({"params": p}, feats_b, set_b, time_b)
        cond_local = jnp.concatenate([y_emb, m_emb_local], axis=-1).astype(
            _F32
        )  # (B,D)

        def row_body2(acc, i):
            row_rng = jax.random.fold_in(rng, i)
            Li = _slice_Li(L, jnp.array(i, _I32))
            cond_i = _dynamic_slice_row(cond_local, jnp.array(i, _I32))
            li, _, _ = _row_loss(
                E_apply,
                eparams,
                Li,
                cond_i,
                queue,
                queue_count,
                tau=tau,
                k_top=k_top,
                chunk=chunk,
                rng=row_rng,
                gumbel_scale=gumbel_scale,
            )
            return acc + li, None

        sum_l, _ = lax.scan(row_body2, jnp.array(0.0, _F32), jnp.arange(B, dtype=_I32))
        return sum_l / jnp.array(B, _F32)

    grads = jax.grad(loss_only)(enc_state.params)
    new_params, new_opt = _adamw_update(
        enc_state.tx, enc_state.params, enc_state.opt_state, grads
    )

    # telemetry: EMA of cond-norm
    cond_norm = _norm2(jnp.mean(cond_vec, axis=0))
    new_cema = _ema(enc_state.cond_norm_ema, cond_norm, 0.99)

    new_state = enc_state.replace(
        params=new_params, opt_state=new_opt, cond_norm_ema=new_cema
    )
    metrics = {
        "enc/loss": loss,
        "enc/mean_e": mean_e,
        "enc/std_e": std_e,
        "enc/cond_norm_ema": new_cema,
    }
    return new_state, loss, metrics
