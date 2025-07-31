import jax
import jax.numpy as jnp
from typing import Literal


class ManifoldWrapper:
    def __init__(self, kind: Literal["sphere", "torus", "klein"], dim: int):
        self.kind = kind
        self.dim = dim
        assert dim >= 2, "dim >= 2 required"
        if kind == "sphere":
            self.embedded_dim = dim + 1
        elif kind in ["torus", "klein"]:
            self.embedded_dim = dim
        else:
            raise ValueError(f"Unsupported manifold kind: {kind}")

    def project(self, x):
        single_point = False
        if x.ndim == 1:
            x = x[None, :]  # (1, d)
            single_point = True

        assert x.ndim == 2, f"xt wrong shape: {x.shape}"

        if self.kind == "sphere":
            norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
            projected = x / jnp.where(norm < 1e-6, 1.0, norm)
        elif self.kind == "torus":
            projected = jnp.mod(x + jnp.pi, 2 * jnp.pi) - jnp.pi
        elif self.kind == "klein":
            projected = self._klein_mod(x)
        else:
            raise NotImplementedError
        # If input was a single point, return squeezed version
        if single_point:
            return projected[0]
        return projected

    def to_tangent(self, x, v):
        if self.kind == "sphere":
            inner = jnp.sum(v * x, axis=-1, keepdims=True)
            return v - inner * x
        elif self.kind in ["torus", "klein"]:
            return v
        else:
            raise NotImplementedError

    def exp_map(self, x, v):
        if self.kind == "sphere":
            norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
            norm_safe = jnp.where(norm < 1e-6, 1e-6, norm)
            direction = v / norm_safe
            moved = jnp.cos(norm) * x + jnp.sin(norm) * direction
            return self.project(moved)
        elif self.kind in ["torus", "klein"]:
            return self.project(x + v)
        else:
            raise NotImplementedError

    def euler_maruyama_step(self, x, drift, noise, dt):
        tangent_update = drift * dt + noise * jnp.sqrt(dt)
        tangent_update = self.to_tangent(x, tangent_update)
        return self.exp_map(x, tangent_update)

    def wrapped_gaussian(self, n, key, std=1.0, mu=None):
        return self.sample_wrapped_normal(n, key, mu=mu, std=std)

    def _klein_mod(self, x):
        x0 = jnp.mod(x[..., 0], 1.0)
        y_raw = x[..., 1]
        flip_x = jnp.where(jnp.floor(y_raw) % 2 == 0, x0, -x0)
        y_mod = jnp.mod(y_raw, 1.0)
        return jnp.stack([flip_x, y_mod], axis=-1)

    def normalize(self, x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / jnp.where(norm < 1e-6, 1.0, norm)

    def project_to_tangent(self, x, v):
        inner = jnp.sum(x * v, axis=-1, keepdims=True)
        return v - inner * x

    def euler_step(self, x, drift, noise, dt):
        update = drift * dt + noise * jnp.sqrt(dt)
        update = self.project_to_tangent(x, update)
        return self.exp_map(x, update)

    def sample_wrapped_normal(self, n, key, mu=None, std=1.0):
        key_noise, _ = jax.random.split(key)

        if mu is None:
            mu = jnp.zeros((self.embedded_dim,))
            if self.kind == "sphere":
                mu = mu.at[-1].set(1.0)

        mu = self.project(mu)  # (embedded_dim,)
        z = jax.random.normal(key_noise, shape=(n, self.embedded_dim)) * std
        mu_batched = jnp.broadcast_to(mu, (n, self.embedded_dim))

        z_proj = jax.vmap(self.project_to_tangent)(mu_batched, z)

        def exp_map_single(mu_i, v_i):
            norm = jnp.linalg.norm(v_i)
            norm = jnp.where(norm < 1e-6, 1e-6, norm)
            direction = v_i / norm
            point = jnp.cos(norm) * mu_i + jnp.sin(norm) * direction  # type: ignore
            return self.normalize(point)

        return jax.vmap(exp_map_single)(mu_batched, z_proj)
