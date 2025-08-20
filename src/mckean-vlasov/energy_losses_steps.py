# energy_losses_steps.py
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


# ===================== Energy trainer state (Flax-compatible) =====================


@struct.dataclass
class EnergyState:
    # non-pytrees
    apply_fn: Callable = struct.field(
        pytree_node=False
    )  # E_apply({"params": p}, L, cond_vec) -> (B,) energy
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # pytrees
    params: FrozenDict
    opt_state: optax.OptState

    # small telemetry / stabilizer
    scale_ema: jnp.ndarray = struct.field(pytree_node=True)  # () f32

    # MoCo-style queue living on device (negatives as cond vectors)
    queue: jnp.ndarray = struct.field(pytree_node=True)  # (Q, D)
    queue_head: jnp.ndarray = struct.field(pytree_node=True)  # () i32
    queue_count: jnp.ndarray = struct.field(pytree_node=True)  # () i32
    queue_size: int = 1024  # python int (static)

    # contrastive controls
    tau: float = 0.07
    gumbel_scale: float = 0.2
    k_top: int = 32
    label_temp: float = 1.0  # reserved for future multi-label weighting

    def replace(self, **updates):
        return dc_replace(self, **updates)


def create_energy_state(
    apply_fn: Callable,
    init_params: FrozenDict,
    *,
    D_cond: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    tau: float = 0.07,
    Q: int = 4096,
    k_top: int = 32,
    gumbel: float = 0.2,
    label_temp: float = 1.0,
) -> EnergyState:
    """Initialize a VRAM-friendly MoCo-style energy trainer."""
    queue = jnp.zeros((int(Q), int(D_cond)), dtype=_F32)
    head0 = jnp.array(0, _I32)
    cnt0 = jnp.array(0, _I32)

    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)

    return EnergyState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
        scale_ema=jnp.array(1.0, _F32),
        queue=queue,
        queue_head=head0,
        queue_count=cnt0,
        queue_size=int(Q),
        tau=float(tau),
        gumbel_scale=float(gumbel),
        k_top=int(k_top),
        label_temp=float(label_temp),
    )


# ===================== Small helpers (JIT-safe) =====================


@jax.jit
def _safe_tau(tau: float) -> jnp.ndarray:
    return jnp.maximum(jnp.array(tau, _F32), jnp.array(1e-6, _F32))


def _dynamic_slice_row(x: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    """Return x[i:i+1, :] with static shapes."""
    D = x.shape[1]  # static for jit
    return lax.dynamic_slice(x, (i, 0), (1, D))


def _slice_Li(L: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    """L: (B,H,W,KS,C) -> Li: (H,W,KS,C)"""
    return lax.dynamic_slice(L, (i, 0, 0, 0, 0), (1,) + L.shape[1:])[0]


@jax.jit
def _stop_grad(x: jnp.ndarray) -> jnp.ndarray:
    return lax.stop_gradient(x)


@jax.jit
def _ema(prev: jnp.ndarray, value: jnp.ndarray, decay: float = 0.99) -> jnp.ndarray:
    return jnp.array(decay, _F32) * prev + (1.0 - jnp.array(decay, _F32)) * value


def _adamw_update(tx, params, opt_state, grads):
    updates, new_opt = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt


# ===================== Queue ops (static sizes → no dynamic shapes) =====================


def _enqueue_queue(
    queue: jnp.ndarray,  # (Q, D)
    queue_head: jnp.ndarray,  # () i32
    queue_count: jnp.ndarray,  # () i32
    new_rows: jnp.ndarray,  # (B, D)
    Q: int,  # python int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Circular enqueue B rows into queue with wrap-around."""
    B, D = new_rows.shape
    Qi = jnp.array(Q, _I32)

    head = queue_head % Qi
    tail_space = Qi - head
    first = jnp.minimum(jnp.array(B, _I32), tail_space)
    second = jnp.array(B, _I32) - first

    q = queue
    # [head:head+first]
    q = lax.dynamic_update_slice_in_dim(q, new_rows[:first, :], head, axis=0)

    # wrap [0:second]
    def do_wrap(q_):
        return lax.dynamic_update_slice_in_dim(
            q_, new_rows[first:, :], jnp.array(0, _I32), axis=0
        )

    q = lax.cond(second > 0, do_wrap, lambda x: x, q)

    new_head = (head + jnp.array(B, _I32)) % Qi
    new_cnt = jnp.minimum(queue_count + jnp.array(B, _I32), Qi)
    return q, new_head, new_cnt


# ===================== Memory-safe energy forward =====================


def _remat_energy_forward(E_apply: Callable):
    """
    Returns a rematerialized forward: (params, Lb, cond) -> energy
    Does compute in bf16 to cut activation RAM, casts back to f32.
    """

    def forward(params, Lb, cond):
        out = E_apply(
            {"params": params}, Lb.astype(jnp.bfloat16), cond.astype(jnp.bfloat16)
        )
        return out.astype(_F32)

    # rematerialize to avoid storing all intermediate activations
    return jax.checkpoint(forward)  # type:ignore


# ===================== Row-wise negatives from queue (no BxB) =====================


def _row_neg_logits_vs_queue(
    E_apply: Callable,
    params: FrozenDict,
    Li: jnp.ndarray,  # (H,W,KS,C)
    queue: jnp.ndarray,  # (Q,D)
    valid_count: jnp.ndarray,  # () i32
    tau: float,
    chunk: int,
    rng: jnp.ndarray | None,
    gumbel_scale: float,
) -> jnp.ndarray:
    """
    Returns (Q,) logits for negatives (masked beyond valid_count).
    Chunked over Q; **we keep chunk tiny (1–4)** to cap VRAM.
    """
    Q, D = queue.shape
    assert Q % chunk == 0, "Set queue size Q to a multiple of `chunk`."
    nseg = Q // chunk

    logits0 = jnp.full((Q,), -jnp.inf, _F32)
    tau_f = _safe_tau(tau)
    remat_E = _remat_energy_forward(E_apply)

    def body(curr, s):
        j0 = s * chunk
        cond_seg = lax.dynamic_slice(queue, (j0, 0), (chunk, D))  # (chunk, D)
        idxs = j0 + jnp.arange(chunk, dtype=_I32)  # (chunk,)
        mask_seg = idxs < valid_count  # (chunk,)

        # microbatch energy: (chunk,H,W,KS,C) vs (chunk,D)
        LiB = jnp.broadcast_to(Li, (chunk,) + Li.shape)
        e_seg = remat_E(params, LiB, cond_seg).astype(_F32)  # (chunk,)
        logits_seg = -e_seg / tau_f

        # optional gumbel; no Python int() on tracer, pass tracer to fold_in
        if rng is not None and gumbel_scale != 0.0:
            subkey = jax.random.fold_in(rng, s)
            u = jax.random.uniform(
                subkey, (chunk,), minval=1e-6, maxval=1.0, dtype=_F32
            )
            g = -jnp.log(-jnp.log(u))
            logits_seg = logits_seg + jnp.array(gumbel_scale, _F32) * g

        logits_seg = jnp.where(mask_seg, logits_seg, -jnp.inf)
        curr = lax.dynamic_update_slice_in_dim(
            curr, logits_seg, j0, axis=0  # type:ignore
        )
        return curr, None

    logits, _ = lax.scan(body, logits0, jnp.arange(nseg, dtype=_I32))
    return logits  # (Q,)


# ===================== Per-row non-saturating contrastive loss =====================


def _row_loss(
    E_apply: Callable,
    params: FrozenDict,
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
    Non-saturating top-k logistic:
      loss_i = softplus( logsumexp( topk(neg_logits) - pos_logit ) )
    Returns (loss_i, mean_neg_energy, std_neg_energy).
    """
    tau_f = _safe_tau(tau)
    remat_E = _remat_energy_forward(E_apply)

    # positive (micro-batch 1)
    e_pos = remat_E(params, Li[None, ...], cond_i).reshape(())  # ()
    logit_pos = -e_pos / tau_f

    # negatives (logits)
    neg_logits = _row_neg_logits_vs_queue(
        E_apply, params, Li, queue, valid_count, tau, chunk, rng, gumbel_scale
    )  # (Q,)

    # top-k (use lax.top_k to avoid dynamic slices)
    k = max(1, min(int(k_top), int(neg_logits.shape[0])))
    topk_vals, _ = lax.top_k(neg_logits, k)  # (k,)

    lse = jax.nn.logsumexp(topk_vals - logit_pos)
    loss_i = jax.nn.softplus(lse)

    # telemetry (convert logits back to energies with tau)
    mask = jnp.isfinite(neg_logits)
    neg_e = -jnp.where(mask, neg_logits, -jnp.inf) * tau_f
    neg_e = jnp.where(jnp.isfinite(neg_e), neg_e, e_pos)
    mean_e = jnp.mean(neg_e)
    std_e = jnp.sqrt(jnp.maximum(jnp.var(neg_e), jnp.array(1e-12, _F32)))
    return loss_i, mean_e, std_e


# ===================== Batch loss (scan over rows) =====================


def _batch_energy_loss(
    state: EnergyState,
    params: FrozenDict,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    cond_vec: jnp.ndarray,  # (B,D)
    rng: jnp.ndarray,
    *,
    chunk: int,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure loss for current batch (no in-batch negatives, only queue).
    Returns: loss, metrics, new_queue, new_head, new_count
    """
    B = L.shape[0]

    def row_body(carry, i):
        row_rng = jax.random.fold_in(rng, i)
        Li = _slice_Li(L, jnp.array(i, _I32))  # (H,W,KS,C)
        cond_i = _dynamic_slice_row(cond_vec, jnp.array(i, _I32))  # (1,D)

        li, mean_e, std_e = _row_loss(
            state.apply_fn,
            params,
            Li,
            cond_i,
            state.queue,
            state.queue_count,
            tau=state.tau,
            k_top=state.k_top,
            chunk=chunk,
            rng=row_rng,
            gumbel_scale=state.gumbel_scale,
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

    # enqueue current conds (stop-grad)
    cond_sg = _stop_grad(cond_vec.astype(_F32))
    new_q, new_head, new_cnt = _enqueue_queue(
        state.queue, state.queue_head, state.queue_count, cond_sg, state.queue_size
    )

    metrics = {
        "energy/mean_e": mean_e,
        "energy/std_e": std_e,
        "energy/queue_count": state.queue_count.astype(_F32),
    }
    return loss, metrics, new_q, new_head, new_cnt


# ===================== Loss+grads wrapper (correct aux ordering) =====================


def _loss_and_grads(
    state: EnergyState,
    params: FrozenDict,
    L: jnp.ndarray,
    cond_vec: jnp.ndarray,
    rng: jnp.ndarray,
    chunk: int,
):
    """
    Return ((loss, aux), grads) with aux=(metrics, new_q, new_head, new_cnt).
    """

    def loss_fn(p):
        loss, metrics, new_q, new_head, new_cnt = _batch_energy_loss(
            state, p, L, cond_vec, rng, chunk=chunk
        )
        aux = (metrics, new_q, new_head, new_cnt)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, aux, grads


# ===================== Public training step =====================


def _choose_safe_chunk(Q: int, requested: int) -> int:
    """Pick the largest divisor of Q not exceeding requested; at least 1."""
    c = max(1, min(int(requested), int(Q)))
    while Q % c != 0 and c > 1:
        c //= 2
    return max(1, c)


def energy_step_E_bank(
    state: EnergyState,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    cond_vec: jnp.ndarray,  # (B,D)
    rng: jnp.ndarray,  # PRNGKey
    *,
    chunk: int = 2,  # **tiny** micro-batch for queue scoring
    k_top_override: int | None = None,
) -> Tuple[EnergyState, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    One VRAM-friendly contrastive step for the energy model using a device-side queue.
    """
    # optional runtime k_top override
    if k_top_override is not None and k_top_override != state.k_top:
        state = state.replace(k_top=int(k_top_override))

    # auto-shrink chunk to a safe divisor of Q to prevent unrolling bloat
    chunk_eff = _choose_safe_chunk(state.queue_size, chunk)

    # compute loss+grads
    loss, aux, grads = _loss_and_grads(
        state, state.params, L, cond_vec, rng, chunk=chunk_eff
    )
    metrics, new_q, new_head, new_cnt = aux

    # optimizer update
    new_params, new_opt = _adamw_update(state.tx, state.params, state.opt_state, grads)

    # EMA scale telemetry
    new_scale = _ema(state.scale_ema, metrics["energy/std_e"], 0.99)

    # pack state
    new_state = state.replace(
        params=new_params,
        opt_state=new_opt,
        queue=new_q,
        queue_head=new_head,
        queue_count=new_cnt,
        scale_ema=new_scale,
    )
    return new_state, loss, metrics
