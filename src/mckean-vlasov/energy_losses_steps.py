from __future__ import annotations
import jax
import optax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Callable, Dict
from dataclasses import field as dc_field
from dataclasses import replace as dc_replace

from flax.core import FrozenDict
from jax import lax


# dc_field(default_factory=lambda:

_F32 = jnp.float32
_I32 = jnp.int32


# --------------------------- State container ---------------------------


@struct.dataclass
class EnergyState:
    # non-pytrees
    apply_fn: Callable = struct.field(
        pytree_node=False
    )  # E_apply(params, L, cond) -> (B,) energy
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # pytrees
    params: FrozenDict
    opt_state: optax.OptState

    # tiny EMA of per-row neg std (stabilization / telemetry)
    scale_ema: jnp.ndarray = struct.field(pytree_node=True)  # () float32

    # memory queue (negatives): shape (Q, D)
    queue: jnp.ndarray = struct.field(pytree_node=True)
    queue_head: jnp.ndarray = struct.field(pytree_node=True)  # () int32
    queue_count: jnp.ndarray = struct.field(pytree_node=True)  # () int32
    queue_size: int = 4096  # python int (static)

    # contrastive / scale
    tau: float = 0.07  # temperature (python float, static)
    gumbel_scale: float = 0.2  # scale of Gumbel noise (python float, static)
    k_top: int = 32  # top-k negatives to focus on (python int, static)
    label_temp: float = 1.0  # kept for API completeness (not used directly here)

    def replace(self, **updates):
        return dc_replace(self, **updates)


def create_energy_state(
    apply_fn: Callable,
    init_params: FrozenDict,
    D_cond: int,
    *,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    tau: float = 0.07,
    Q: int = 4096,
    k_top: int = 32,
    gumbel: float = 0.2,
    label_temp: float = 1.0,
) -> EnergyState:
    """
    Build a MoCo-style energy trainer state.

    Args:
        apply_fn: E_apply callable
        init_params: flax params for the energy model
        D_cond: dimension of cond_vec
        lr, weight_decay: optimizer
        tau: temperature
        Q: queue size (should be multiple of `chunk` you’ll use at step)
        k_top: focus on hardest negatives
        gumbel: gumbel noise scale (0 disables)
        label_temp: reserved for multi-label weighting (not applied here)

    Returns:
        EnergyMoCoState
    """
    # Small, fixed-size memory bank of negatives (lives on device, tiny)
    queue = jnp.zeros((int(Q), int(D_cond)), dtype=_F32)
    head0 = jnp.array(0, _I32)
    cnt0 = jnp.array(0, _I32)

    # AdamW keeps weights in check
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)

    return EnergyState(
        apply_fn=apply_fn,
        tx=tx,
        params=init_params,
        opt_state=tx.init(init_params),
        tau=float(tau),
        gumbel_scale=float(gumbel),
        k_top=int(k_top),
        label_temp=float(label_temp),
        queue=queue,
        queue_head=head0,
        queue_count=cnt0,
        queue_size=int(Q),
        scale_ema=jnp.array(1.0, _F32),
    )


# --------------------------- Queue helpers ---------------------------


def _enqueue_queue(
    queue: jnp.ndarray,  # (Q, D)
    queue_head: jnp.ndarray,  # () i32
    queue_count: jnp.ndarray,  # () i32
    new_rows: jnp.ndarray,  # (B, D)
    Q: int,  # python int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Circular enqueue B rows into queue with wrap-around. All slice sizes static.
    """
    B, D = new_rows.shape
    head = queue_head % jnp.array(Q, _I32)  # () i32
    tail_space = jnp.array(Q, _I32) - head  # () i32
    # First contiguous chunk size = min(B, tail_space)
    first = jnp.minimum(jnp.array(B, _I32), tail_space)
    second = jnp.array(B, _I32) - first

    q = queue
    # write first [0:first] -> [head:head+first]
    q = lax.dynamic_update_slice_in_dim(q, new_rows[:first, :], head, axis=0)

    # write wrapped [first:B] -> [0:second] if needed
    def do_wrap(q_):
        return lax.dynamic_update_slice_in_dim(
            q_, new_rows[first:, :], jnp.array(0, _I32), axis=0
        )

    q = lax.cond(second > 0, do_wrap, lambda x: x, q)

    new_head = (head + jnp.array(B, _I32)) % jnp.array(Q, _I32)
    new_cnt = jnp.minimum(queue_count + jnp.array(B, _I32), jnp.array(Q, _I32))
    return q, new_head, new_cnt


# --------------------------- Row-wise logits over queue ---------------------------


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
    Compute logits (negatives only) for a single Li against all valid entries in queue.
    Uses fixed-size chunking over Q; masks entries >= valid_count.
    Returns shape (Q,) with -inf on invalid slots.
    """
    Q, D = queue.shape
    nseg = Q // chunk  # require Q % chunk == 0 for pure static scan
    assert Q % chunk == 0, "Set queue size Q to a multiple of `chunk`."

    logits0 = jnp.full((Q,), -jnp.inf, _F32)

    # Prepare Li replicated per-chunk once we know chunk size in the loop body
    def body(carry, s):
        j0 = s * chunk
        cond_seg = lax.dynamic_slice(queue, (j0, 0), (chunk, D))  # (chunk, D)
        # mask for valid rows in this segment
        idxs = j0 + jnp.arange(chunk, dtype=_I32)  # (chunk,)
        mask_seg = idxs < valid_count  # (chunk,)

        LiB = jnp.broadcast_to(Li, (chunk,) + Li.shape)  # (chunk, H,W,KS,C)
        e_seg = E_apply({"params": params}, LiB, cond_seg).astype(_F32)  # (chunk,)
        logits_seg = -e_seg / jnp.maximum(jnp.array(tau, _F32), 1e-6)

        # optional Gumbel noise (stochastic hard negatives)
        if rng is not None and gumbel_scale != 0.0:
            subkey = jax.random.fold_in(rng, int(s))
            u = jax.random.uniform(
                subkey, (chunk,), minval=1e-6, maxval=1.0, dtype=_F32
            )
            g = -jnp.log(-jnp.log(u))
            logits_seg = logits_seg + jnp.array(gumbel_scale, _F32) * g

        # mask out invalid queue slots
        logits_seg = jnp.where(mask_seg, logits_seg, -jnp.inf)

        carry = lax.dynamic_update_slice_in_dim(carry, logits_seg, j0, axis=0)  # type: ignore
        return carry, None

    logits, _ = lax.scan(body, logits0, jnp.arange(nseg))
    return logits  # (Q,)


# --------------------------- Loss per-row (no BxB) ---------------------------


def _row_loss(
    E_apply: Callable,
    params: FrozenDict,
    Li: jnp.ndarray,  # (H,W,KS,C)
    cond_i: jnp.ndarray,  # (1,D)
    queue: jnp.ndarray,  # (Q,D)
    valid_count: jnp.ndarray,  # () i32
    tau: float,
    k_top: int,
    chunk: int,
    rng: jnp.ndarray | None,
    gumbel_scale: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns: (loss_i, mean_neg_e, std_neg_e)
      loss_i = softplus(logsumexp(topk_neg - pos_logit))
    """
    # positive energy/logit
    e_pos = E_apply({"params": params}, Li[None, ...], cond_i).reshape(())  # ()
    logit_pos = -e_pos / jnp.maximum(jnp.array(tau, _F32), 1e-6)

    # negatives from queue
    neg_logits = _row_neg_logits_vs_queue(
        E_apply, params, Li, queue, valid_count, tau, chunk, rng, gumbel_scale
    )  # (Q,)

    # top-k focus (largest logits are hardest negatives)
    # Note: jnp.sort ascending → take last k
    k = int(k_top)
    k = max(1, min(k, neg_logits.shape[0]))
    sorted_neg = jnp.sort(neg_logits)  # ascending
    topk = sorted_neg[-k:]  # (k,)

    # Non-saturating logistic: log(1 + sum_j exp(neg - pos))
    lse = jax.nn.logsumexp(topk - logit_pos)
    loss_i = jax.nn.softplus(lse)  # scalar

    # stats in energy space for EMA scale (convert back)
    # mask invalid (-inf) before mapping back; safe convert using tau
    valid_mask = jnp.isfinite(neg_logits)
    neg_e = -jnp.where(valid_mask, neg_logits, -jnp.inf) * jnp.array(tau, _F32)
    # replace -inf by pos energy to avoid contaminating stats
    neg_e = jnp.where(jnp.isfinite(neg_e), neg_e, e_pos)
    mean_e = jnp.mean(neg_e)  # type: ignore
    std_e = jnp.std(neg_e)  # type: ignore

    return loss_i, mean_e, std_e


# --------------------------- Public step (JIT) ---------------------------


@jax.jit
def _ceiling_div(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return (a + b - 1) // b


@jax.jit
def _safe_mean(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(x.astype(_F32))


@jax.jit
def _safe_std(x: jnp.ndarray) -> jnp.ndarray:
    x = x.astype(_F32)
    return jnp.sqrt(jnp.maximum(jnp.var(x), jnp.array(1e-12, _F32)))


@jax.jit
def _adamw_update(
    tx: optax.GradientTransformation,
    params: FrozenDict,
    opt_state: optax.OptState,
    grads,
):
    updates, new_opt = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt


@jax.jit
def _ema_update(
    prev: jnp.ndarray, value: jnp.ndarray, decay: float = 0.99
) -> jnp.ndarray:
    return jnp.array(decay, _F32) * prev + (1.0 - jnp.array(decay, _F32)) * value


@jax.jit
def _stop_grad(x: jnp.ndarray) -> jnp.ndarray:
    return lax.stop_gradient(x)


@jax.jit
def _concat(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([a, b], axis=0)


@jax.jit
def _dynamic_slice_row(x: jnp.ndarray, i: jnp.ndarray, D: int) -> jnp.ndarray:
    return lax.dynamic_slice(x, (i, 0), (1, D))


@jax.jit
def _broadcast_L(Li: jnp.ndarray, n: int) -> jnp.ndarray:
    return jnp.broadcast_to(Li, (n,) + Li.shape)


@jax.jit
def _int32(x: int) -> jnp.ndarray:
    return jnp.array(x, _I32)


@jax.jit
def _zero_like_scalar(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x, dtype=_F32)


@jax.jit
def _one_like_scalar(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones_like(x, dtype=_F32)


@jax.jit
def _sum(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x)


@jax.jit
def _stack(xs):
    return jnp.stack(xs, axis=0)


def _batch_energy_loss(
    state: EnergyState,
    params: FrozenDict,
    L: jnp.ndarray,  # (B, H,W,KS,C)
    cond_vec: jnp.ndarray,  # (B, D)
    rng: jnp.ndarray,
    *,
    chunk: int,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure loss for current batch (no in-batch negatives, only queue).
    Returns:
      loss (scalar),
      metrics dict,
      new_queue, new_head, new_count (for enqueue later)
    """
    B, D = cond_vec.shape
    Q, Dq = state.queue.shape
    assert D == Dq, "cond_vec dim must equal queue dim."

    # Per-row loss in a scan (static shapes).
    def row_body(carry, i):
        # deterministic row RNG
        row_rng = jax.random.fold_in(rng, i)

        Li = lax.dynamic_slice(L, (i, 0, 0, 0, 0), (1,) + L.shape[1:])
        Li = Li[0]  # (H,W,KS,C)
        cond_i = _dynamic_slice_row(cond_vec, jnp.array(i, _I32), D)  # (1, D)

        li, mean_e, std_e = _row_loss(
            state.apply_fn,
            params,
            Li,
            cond_i,
            state.queue,
            state.queue_count,
            state.tau,
            state.k_top,
            chunk,
            row_rng,
            state.gumbel_scale,
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

    # enqueue stop-grad version of conds
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


@jax.jit
def _update_scale(prev_scale: jnp.ndarray, batch_std: jnp.ndarray) -> jnp.ndarray:
    return _ema_update(prev_scale, batch_std, decay=0.99)


@jax.jit
def _apply_grads(
    tx: optax.GradientTransformation,
    params: FrozenDict,
    opt_state: optax.OptState,
    grads,
):
    return _adamw_update(tx, params, opt_state, grads)


@jax.jit
def _replace_state_core(
    state: EnergyState,
    params: FrozenDict,
    opt_state: optax.OptState,
    queue: jnp.ndarray,
    head: jnp.ndarray,
    count: jnp.ndarray,
    scale_ema: jnp.ndarray,
) -> EnergyState:
    # flax.struct dataclasses are immutable: use .replace
    return state.replace(
        params=params,
        opt_state=opt_state,
        queue=queue,
        queue_head=head,
        queue_count=count,
        scale_ema=scale_ema,
    )


@jax.jit
def _pack_aux(loss: jnp.ndarray, metrics: Dict[str, jnp.ndarray]):
    # pack metrics as a tuple of scalars to keep jit+pytree simple
    return (
        loss,
        metrics["energy/mean_e"],
        metrics["energy/std_e"],
        metrics["energy/queue_count"],
    )


@jax.jit
def _unpack_aux(aux):
    loss, mean_e, std_e, qcnt = aux
    return loss, {
        "energy/mean_e": mean_e,
        "energy/std_e": std_e,
        "energy/queue_count": qcnt,
    }


@jax.jit
def _loss_and_grads(
    state: EnergyState,
    params: FrozenDict,
    L: jnp.ndarray,
    cond_vec: jnp.ndarray,
    rng: jnp.ndarray,
    chunk: int,
):
    def loss_fn(p):
        loss, metrics, new_q, new_head, new_cnt = _batch_energy_loss(
            state, p, L, cond_vec, rng, chunk=chunk
        )
        return _pack_aux(loss, metrics), (new_q, new_head, new_cnt)

    (aux, (new_q, new_head, new_cnt)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)
    return aux, grads, new_q, new_head, new_cnt


def energy_step_E_bank(
    state: EnergyState,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    cond_vec: jnp.ndarray,  # (B,D)
    rng: jnp.ndarray,  # PRNGKey
    *,
    chunk: int = 64,  # must divide queue_size
) -> Tuple[EnergyState, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    One MoCo contrastive step for the energy model.

    Notes:
      - No in-batch negatives (only queue) → stable VRAM.
      - Queue size Q **must be a multiple of `chunk`**.
      - All slice sizes are static; no dynamic index sizes.

    Returns:
      new_state, loss (scalar), metrics dict
    """
    # Compute loss & grads (and the updated queue content)
    aux, grads, new_q, new_head, new_cnt = _loss_and_grads(
        state, state.params, L, cond_vec, rng, jnp.array(chunk, _I32)
    )
    loss, metrics = _unpack_aux(aux)

    # Optimizer update
    new_params, new_opt = _apply_grads(state.tx, state.params, state.opt_state, grads)

    # EMA scale update
    new_scale = _update_scale(state.scale_ema, metrics["energy/std_e"])

    # Pack new state
    new_state = _replace_state_core(
        state,
        params=new_params,
        opt_state=new_opt,
        queue=new_q,
        head=new_head,
        count=new_cnt,
        scale_ema=new_scale,
    )
    return new_state, loss, metrics
