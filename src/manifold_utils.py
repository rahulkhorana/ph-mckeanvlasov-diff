import jax
import jax.numpy as jnp
from typing import Literal


class ManifoldWrapper:
    def __init__(self, kind: Literal["sphere", "torus", "klein"], dim: int):
        self.kind = kind
        self.dim = dim
        assert dim >= 2, "dim ≥ 2 required"
        if kind == "sphere":
            self.embedded_dim = dim + 1
        elif kind in ["torus", "klein"]:
            self.embedded_dim = dim
        else:
            raise ValueError(f"Unsupported manifold kind {kind}")

    # ---------- geometry helpers ------------------------------------------------
    def _klein_mod(self, x):
        x0 = jnp.mod(x[..., 0], 1.0)
        y = jnp.mod(x[..., 1], 1.0)
        flip = jnp.where(jnp.floor(x[..., 1]) % 2 == 0, x0, -x0)
        return jnp.stack([flip, y], -1)

    def project(self, x):
        single = False
        if x.ndim == 1:
            x, single = x[None, :], True

        if self.kind == "sphere":
            nrm = jnp.linalg.norm(x, -1, keepdims=True)
            y = x / jnp.where(nrm < 1e-6, 1.0, nrm)
        elif self.kind == "torus":
            y = jnp.mod(x + jnp.pi, 2 * jnp.pi) - jnp.pi
        elif self.kind == "klein":
            y = self._klein_mod(x)
        else:
            raise NotImplementedError

        return y[0] if single else y

    def project_to_tangent(self, x, v):
        inner = jnp.sum(x * v, -1, keepdims=True)
        return v - inner * x

    def exp_map(self, x, v):
        if self.kind == "sphere":
            nrm = jnp.linalg.norm(v, -1, keepdims=True)
            nrm = jnp.where(nrm < 1e-6, 1e-6, nrm)
            dir_ = v / nrm
            return self.project(jnp.cos(nrm) * x + jnp.sin(nrm) * dir_)  # type: ignore
        else:  # flat manifolds
            return self.project(x + v)

    # ---------- sampling ---------------------------------------------------------
    def sample_wrapped_normal(self, n, key, *, mu=None, std=1.0):
        (k1,) = jax.random.split(key, 1)
        if mu is None:
            mu = jnp.zeros((self.embedded_dim,))
            if self.kind == "sphere":
                mu = mu.at[-1].set(1.0)
        mu = self.project(mu)

        z = jax.random.normal(k1, (n, self.embedded_dim)) * std
        mu_b = jnp.broadcast_to(mu, (n, self.embedded_dim))
        z = jax.vmap(self.project_to_tangent)(mu_b, z)

        def _one(m, v):
            nrm = jnp.linalg.norm(v)
            nrm = jnp.where(nrm < 1e-6, 1e-6, nrm)
            dir_ = v / nrm
            pt = jnp.cos(nrm) * m + jnp.sin(nrm) * dir_  # type: ignore
            return self.project(pt)

        return jax.vmap(_one)(mu_b, z)
