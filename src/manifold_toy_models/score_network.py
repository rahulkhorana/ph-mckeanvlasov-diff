import jax
import jax.numpy as jnp
import equinox as eqx
from equinox.nn import MLP, Linear
from typing import Callable, Any
from manifold_utils import ManifoldWrapper


def sinusoidal_embed(t, dim):
    """Standard positional encoding for time inputs"""
    half_dim = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000), half_dim))
    angles = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class FiLMLayer(eqx.Module):
    scale: Linear
    shift: Linear

    def __init__(self, width: int, t_dim: int, key):
        k1, k2 = jax.random.split(key)
        self.scale = Linear(1, 1, key=k1)
        self.shift = Linear(1, 1, key=k2)

    def __call__(self, x: jnp.ndarray, t_embed: jnp.ndarray):
        return x * self.scale(t_embed) + self.shift(t_embed)


class ScoreNet(eqx.Module):
    mlp: eqx.nn.MLP
    t_dim: int
    x_dim: int

    def __init__(
        self,
        dim: int,
        width: int = 128,
        depth: int = 3,
        t_dim: int = 16,
        *,
        key: "jax.Array",
    ):
        self.t_dim = t_dim
        self.x_dim = dim

        mlp_input_dim = dim + t_dim
        (mlp_key,) = jax.random.split(key, 1)

        self.mlp = eqx.nn.MLP(
            in_size=mlp_input_dim,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=mlp_key,
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # x: (D,), t: scalar or (B,)
        if t.ndim == 0:
            t = t[None]
        if x.ndim == 1:
            x = x[None, :]  # (1, D)

        # Normalize x to prevent unstable magnitudes
        x_normed = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)  # (B, D)

        t_embed = sinusoidal_embed(t, self.t_dim)  # (B, t_dim)
        if t_embed.ndim == 1:
            t_embed = t_embed[None, :]
        inp = jnp.concatenate([x_normed, t_embed], axis=-1)  # (B, D + t_dim)

        out = jax.vmap(self.mlp)(inp)  # (B, D)
        if out.shape[0] == 1:
            return out[0]
        return out
