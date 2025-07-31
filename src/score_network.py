import jax, jax.numpy as jnp, equinox as eqx
from typing import Optional, Sequence


def sinusoidal_embed(t, dim):
    freqs = jnp.exp(jnp.linspace(jnp.log(1e-4), jnp.log(1.0), dim // 2))
    t = jnp.atleast_1d(t)  # ensure shape (B,) or (1,)
    angles = t[:, None] * freqs[None, :]  # (B, dim//2)
    embed = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (B, dim)
    return embed.squeeze(0) if embed.shape[0] == 1 else embed


class ScoreNet(eqx.Module):
    t_dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    embed_mlp: eqx.nn.MLP

    def __init__(
        self,
        dim: int,
        *,
        t_dim: int = 32,
        cond_dim: int = 0,
        width: int = 128,
        depth: int = 3,
        key,
    ):
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        in_size = dim + t_dim + cond_dim

        self.embed_mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            use_bias=True,
            key=key,
        )

    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, cond: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        if t.ndim == 2:
            assert t.shape[1] == 1, f"t shape {t.shape} unexpected"
            t = t[:, 0]
        t_embed = sinusoidal_embed(t, self.t_dim)  # (B, t_dim)
        if cond is None:
            inp = jnp.concatenate([x, t_embed], axis=-1)  # (B, in_dim)
        else:
            inp = jnp.concatenate([x, t_embed, cond], axis=-1)
        return self.embed_mlp(inp)  # (B, dim)
