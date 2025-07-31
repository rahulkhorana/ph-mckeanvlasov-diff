import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


class ScoreNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, dim, cond_dim=0, width=128, depth=3, *, key):
        # input size = manifold-dim  + 1 time-dim + optional condition dims
        in_size = dim + 1 + cond_dim
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, cond: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        # x: (batch, dim), t: (batch,) or (batch,1)
        if t.ndim == 1:
            t = t[:, None]
        if cond is not None:
            input_vec = jnp.concatenate([x, t, cond], axis=-1)
        else:
            input_vec = jnp.concatenate([x, t], axis=-1)
        # input_vec: (batch, in_size)

        # Hand-vmap a single-call into self.mlp, so each example is mlp(iv)
        def apply_one(iv):
            return self.mlp(iv)  # iv: (in_size,), returns (dim,)

        # Now map over axis 0
        return jax.vmap(apply_one)(input_vec)
