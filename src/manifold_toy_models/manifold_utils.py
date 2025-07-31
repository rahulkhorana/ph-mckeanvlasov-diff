from typing import Literal
import jax
import jax.numpy as jnp


class ManifoldWrapper:
    def __init__(self, kind: Literal["sphere", "torus", "klein"], dim: int):
        if dim < 2:
            raise ValueError("dim >= 2 required")

        self.kind = kind
        self.dim = dim
        self.embedded_dim = dim + 1 if kind == "sphere" else dim

    def project(self, x):
        """Project onto the manifold."""
        single_point = False
        if x.ndim == 1:
            x = x[None, :]
            single_point = True

        if self.kind == "sphere":
            norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
            x_proj = x / jnp.where(norm < 1e-6, 1.0, norm)
        elif self.kind == "torus":
            x_proj = jnp.mod(x + jnp.pi, 2 * jnp.pi) - jnp.pi
        elif self.kind == "klein":
            x_proj = self._klein_mod(x)
        else:
            raise NotImplementedError(f"Unknown kind: {self.kind}")

        return x_proj[0] if single_point else x_proj

    def _klein_mod(self, x):
        """Implements Klein bottle wrapping."""
        x0 = jnp.mod(x[..., 0], 1.0)
        y_raw = x[..., 1]
        flip_x = jnp.where(jnp.floor(y_raw) % 2 == 0, x0, -x0)
        y_mod = jnp.mod(y_raw, 1.0)
        return jnp.stack([flip_x, y_mod], axis=-1)

    def project_to_tangent(self, x, v):
        if self.kind == "sphere":
            inner = jnp.sum(v * x, axis=-1, keepdims=True)
            return v - inner * x
        else:
            return v  # flat space: no constraint

    def exp_map(self, x, v):
        if self.kind == "sphere":
            norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
            norm_safe = jnp.where(norm < 1e-6, 1e-6, norm)
            direction = v / norm_safe
            moved = jnp.cos(norm) * x + jnp.sin(norm) * direction
            return self.project(moved)
        else:
            return self.project(x + v)

    def euler_maruyama_step(self, x, drift, noise, dt):
        update = drift * dt + noise * jnp.sqrt(dt)
        update = self.project_to_tangent(x, update)
        return self.exp_map(x, update)

    def normalize(self, x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / jnp.where(norm < 1e-6, 1.0, norm)

    def sample_wrapped_normal(self, n, key, mu=None, std=1.0):
        key_noise, _ = jax.random.split(key)

        if mu is None:
            mu = jnp.zeros((self.embedded_dim,))
            if self.kind == "sphere":
                mu = mu.at[-1].set(1.0)

        mu = self.project(mu)
        z = jax.random.normal(key_noise, shape=(n, self.embedded_dim)) * std
        mu_batched = jnp.broadcast_to(mu, (n, self.embedded_dim))
        v_proj = jax.vmap(self.project_to_tangent)(mu_batched, z)

        def exp_map_single(m, v):
            norm = jnp.linalg.norm(v)
            norm = jnp.where(norm < 1e-6, 1e-6, norm)
            dir = v / norm
            return self.project(jnp.cos(norm) * m + jnp.sin(norm) * dir)  # type: ignore

        return jax.vmap(exp_map_single)(mu_batched, v_proj)
