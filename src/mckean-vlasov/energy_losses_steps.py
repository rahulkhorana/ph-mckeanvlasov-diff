# energy_losses_steps.py
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
import optax

_F32 = jnp.float32
_I32 = jnp.int32


# ===================== State =====================


@struct.dataclass
class EnergyState:
    # non-pytrees
    apply_fn: Callable = struct.field(
        pytree_node=False
    )  # E({"params":p}, L, cond)->(B,)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # pytrees
    params: FrozenDict
    opt_state: optax.OptState

    # telemetry
    scale_ema: jnp.ndarray = struct.field(pytree_node=True)  # () f32

    # MoCo queue on device (small)
    queue: jnp.ndarray = struct.field(pytree_node=True)  # (Q, D)
    queue_head: jnp.ndarray = struct.field(pytree_node=True)  # () i32
    queue_count: jnp.ndarray = struct.field(pytree_node=True)  # () i32
    queue_size: int = 4096  # static

    # contrastive hparams (python scalars)
    tau: float = 0.07
    gumbel_scale: float = 0.2
    k_top: int = 32
    label_temp: float = 1.0  # reserved

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
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    queue = jnp.zeros((int(Q), int(D_cond)), dtype=_F32)
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
        queue_head=jnp.array(0, _I32),
        queue_count=jnp.array(0, _I32),
        queue_size=int(Q),
        scale_ema=jnp.array(1.0, _F32),
    )


# ===================== small helpers (no global jit) =====================


def _safe_tau(tau: float) -> jnp.ndarray:
    return jnp.maximum(jnp.array(tau, _F32), jnp.array(1e-6, _F32))


def _nan_to_finite(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return jnp.clip(x, -1e6, 1e6)


def _avg_pool_hw_2x(x: jnp.ndarray) -> jnp.ndarray:
    # x: (B,H,W,K,C)
    B, H, W, K, C = x.shape
    x = x.reshape(B, H // 2, 2, W // 2, 2, K, C).mean(axis=(2, 4))
    return x


def _avg_pool_hw(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    if factor <= 1:
        return x
    y = x
    f = int(factor)
    while f >= 2:
        y = _avg_pool_hw_2x(y)
        f //= 2
    return y


def _enqueue_queue(
    queue: jnp.ndarray,  # (Q,D)
    queue_head: jnp.ndarray,  # () i32
    queue_count: jnp.ndarray,  # () i32
    new_rows: jnp.ndarray,  # (B,D)
    Q: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B, D = new_rows.shape
    head = int(queue_head.item())
    cnt = int(queue_count.item())
    # write (possibly wrap)
    q = queue
    first = min(B, Q - head)
    if first > 0:
        q = q.at[head : head + first, :].set(new_rows[:first, :])
    second = B - first
    if second > 0:
        q = q.at[0:second, :].set(new_rows[first:, :])
    new_head = (head + B) % Q
    new_cnt = min(cnt + B, Q)
    return q, jnp.array(new_head, _I32), jnp.array(new_cnt, _I32)


# ===================== E apply wrappers =====================


def _make_E_call(apply_fn: Callable, backend: str):
    """
    Returns a callable E_call(params, L, cond)->(B,) possibly jitted to CPU.
    """

    def _call(p, Lb, cond):
        return apply_fn({"params": p}, Lb, cond).astype(_F32)

    if backend == "cpu":
        return jax.jit(_call, backend="cpu")
    elif backend == "gpu":
        # leave non-jitted to avoid large fused graphs; we already micro-batch
        return _call
    else:
        return _call


def _safe_E_apply(E_call, params, Lb, cond, pool_hw: int) -> jnp.ndarray:
    Ls = _avg_pool_hw(Lb, pool_hw) if pool_hw > 1 else Lb
    e = E_call(params, Ls, cond)
    return _nan_to_finite(e)


# ===================== Row losses (Python loops, micro-batch=1) =====================


def _row_neg_logits_vs_queue_py(
    E_call: Callable,  # params,L,cond -> (1,)
    params: FrozenDict,
    Li: jnp.ndarray,  # (H,W,KS,C)
    queue: jnp.ndarray,  # (Q,D)
    valid_count: jnp.ndarray,  # () i32
    tau_f: jnp.ndarray,  # scalar f32
    gumbel_scale: float,
    rng: jnp.ndarray | None,
    pool_hw: int,
) -> jnp.ndarray:
    Q, D = queue.shape
    vcnt = int(jax.device_get(valid_count))
    vals = []
    for j in range(Q):
        if j < vcnt:
            cond_j = queue[j : j + 1, :]  # (1,D)
            e_j = _safe_E_apply(E_call, params, Li[None, ...], cond_j, pool_hw=pool_hw)[
                0
            ]
            logit_j = -e_j / tau_f
            # optional gumbel (stochastic hard negs)
            if rng is not None and gumbel_scale != 0.0:
                subkey = jax.random.fold_in(rng, j)
                u = jax.random.uniform(
                    subkey, (1,), minval=1e-6, maxval=1.0, dtype=_F32
                )[0]
                g = -jnp.log(-jnp.log(u))
                logit_j = logit_j + jnp.array(gumbel_scale, _F32) * g
            vals.append(logit_j)
        else:
            vals.append(jnp.array(-jnp.inf, _F32))
    return jnp.stack(vals, axis=0)  # (Q,)


def _row_loss_py(
    E_call: Callable,
    params: FrozenDict,
    Li: jnp.ndarray,  # (H,W,KS,C)
    cond_i: jnp.ndarray,  # (1,D)
    queue: jnp.ndarray,  # (Q,D)
    valid_count: jnp.ndarray,  # () i32
    *,
    tau: float,
    k_top: int,
    rng: jnp.ndarray | None,
    gumbel_scale: float,
    pool_hw: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    tau_f = _safe_tau(tau)
    # positive
    e_pos = _safe_E_apply(E_call, params, Li[None, ...], cond_i, pool_hw=pool_hw)[0]
    logit_pos = -e_pos / tau_f

    if int(jax.device_get(valid_count)) == 0:
        return jnp.array(0.0, _F32), e_pos, jnp.array(1e-6, _F32)

    # negatives
    neg_logits = _row_neg_logits_vs_queue_py(
        E_call, params, Li, queue, valid_count, tau_f, gumbel_scale, rng, pool_hw
    )  # (Q,)

    k = max(1, min(int(k_top), neg_logits.shape[0]))
    topk_vals, _ = jax.lax.top_k(neg_logits, k)
    lse = jax.nn.logsumexp(topk_vals - logit_pos)
    loss_i = jax.nn.softplus(jnp.where(jnp.isfinite(lse), lse, -jnp.inf))

    mask = jnp.isfinite(neg_logits)
    neg_e = -jnp.where(mask, neg_logits, -jnp.inf) * tau_f
    neg_e = jnp.where(jnp.isfinite(neg_e), neg_e, e_pos)
    mean_e = jnp.mean(neg_e)
    std_e = jnp.sqrt(jnp.maximum(jnp.var(neg_e), jnp.array(1e-12, _F32)))
    return loss_i, mean_e, std_e


# ===================== Batch loss (Python loop over B) =====================


def _batch_energy_loss_py(
    E_call: Callable,
    state: EnergyState,
    params: FrozenDict,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    cond_vec: jnp.ndarray,  # (B,D)
    rng: jnp.ndarray,
    *,
    pool_hw: int,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B = int(L.shape[0])
    total = jnp.array(0.0, _F32)
    mean_e_sum = jnp.array(0.0, _F32)
    std_e_sum = jnp.array(0.0, _F32)

    for i in range(B):
        row_rng = jax.random.fold_in(rng, i)
        Li = L[i]  # (H,W,KS,C)
        cond_i = cond_vec[i : i + 1, :]  # (1,D)

        li, me, se = _row_loss_py(
            E_call,
            params,
            Li,
            cond_i,
            state.queue,
            state.queue_count,
            tau=state.tau,
            k_top=state.k_top,
            rng=row_rng,
            gumbel_scale=state.gumbel_scale,
            pool_hw=pool_hw,
        )
        total = total + li
        mean_e_sum = mean_e_sum + me
        std_e_sum = std_e_sum + se

    loss = total / jnp.array(B, _F32)
    mean_e = jnp.nan_to_num(
        mean_e_sum / jnp.array(B, _F32), nan=0.0, posinf=0.0, neginf=0.0
    )
    std_e = jnp.nan_to_num(
        std_e_sum / jnp.array(B, _F32), nan=1e-6, posinf=1e6, neginf=1e-6
    )

    # enqueue stop-grad conds
    cond_sg = jax.lax.stop_gradient(cond_vec.astype(_F32))
    new_q, new_head, new_cnt = _enqueue_queue(
        state.queue, state.queue_head, state.queue_count, cond_sg, state.queue_size
    )

    metrics = {
        "energy/mean_e": mean_e,
        "energy/std_e": std_e,
        "energy/q_cnt": state.queue_count.astype(_F32),
        "energy/loss": loss,
    }
    return loss, metrics, new_q, new_head, new_cnt


# ===================== One training step =====================


def energy_step_E_bank(
    state: EnergyState,
    L: jnp.ndarray,  # (B,H,W,KS,C)
    cond_vec: jnp.ndarray,  # (B,D)
    rng: jnp.ndarray,  # PRNGKey
    *,
    pool_hw: int = 4,  # 4x downsample in energy scorer
    score_on: str = "cpu",  # "cpu" avoids GPU workspaces; set "gpu" if you want
    chunk=0,
) -> Tuple[EnergyState, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    MoCo-style energy step with hard VRAM clamp:
      - per-sample, per-negative micro-batch = 1
      - CPU backend by default (no GPU workspace allocations)
      - HxW pooling reduces activations 16× by default
    """
    # build scorer on chosen backend
    backend = "cpu" if score_on.lower() == "cpu" else "gpu"
    E_call = _make_E_call(state.apply_fn, backend=backend)

    # define pure loss for autodiff
    def loss_fn(p):
        loss, metrics, new_q, new_head, new_cnt = _batch_energy_loss_py(
            E_call, state, p, L, cond_vec, rng, pool_hw=pool_hw
        )
        aux = (metrics, new_q, new_head, new_cnt)
        return loss, aux

    (loss, (metrics, new_q, new_head, new_cnt)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    # optimizer update (on whatever device grads live on)
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # EMA telemetry
    new_scale = 0.99 * state.scale_ema + 0.01 * metrics["energy/std_e"]

    # pack new state
    new_state = state.replace(
        params=new_params,
        opt_state=new_opt,
        queue=new_q,
        queue_head=new_head,
        queue_count=new_cnt,
        scale_ema=new_scale,
    )
    return new_state, loss, metrics
