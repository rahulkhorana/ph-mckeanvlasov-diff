import jax
import jax.numpy as jnp
from typing import Literal, Optional


class ManifoldWrapper:
    """
    A robust, geometrically correct wrapper for manifold operations.
    This version provides the correct target distributions:
    - Sphere: A concentrated wrapped normal distribution.
    - Torus: A uniform distribution covering the entire surface.
    """

    def __init__(self, kind: Literal["sphere", "torus"], dim: int):
        self.kind = kind
        self.dim = dim

        if kind == "sphere":
            if dim < 2:
                raise ValueError("Sphere dim must be >= 2")
            self.embedded_dim = dim + 1
        elif kind == "torus":
            if dim != 2:
                raise ValueError("Only T^2 (dim=2) is supported for 3D donut.")
            self.embedded_dim = 3  # T^2 is embedded in R^3
            self.R = 2.0  # Major radius
            self.r = 1.0  # Minor radius
        else:
            raise NotImplementedError(f"Unknown kind: {self.kind}")

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Projects a point (or batch of points) onto the manifold."""
        if x.ndim == 1:
            return self._project_single(x)
        return jax.vmap(self._project_single)(x)

    def _project_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """Projects a single point onto the manifold."""
        if self.kind == "sphere":
            norm = jnp.linalg.norm(x)
            return x / jnp.maximum(norm, 1e-8)

        elif self.kind == "torus":
            xy_norm = jnp.linalg.norm(x[:2])
            p_on_maj_circ = x[:2] / jnp.maximum(xy_norm, 1e-8) * self.R
            v = x - jnp.pad(p_on_maj_circ, (0, 1))
            v_scaled = v / jnp.maximum(jnp.linalg.norm(v), 1e-8) * self.r
            return jnp.pad(p_on_maj_circ, (0, 1)) + v_scaled

        else:
            norm = jnp.linalg.norm(x)
            return x / jnp.maximum(norm, 1e-8)

    def project_to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Projects a vector v onto the tangent space at point x."""
        x_proj = self._project_single(x)

        if self.kind == "sphere":
            normal = x_proj / jnp.maximum(jnp.linalg.norm(x_proj), 1e-8)

        elif self.kind == "torus":
            xy_norm = jnp.linalg.norm(x_proj[:2])
            p_on_maj_circ = x_proj[:2] / jnp.maximum(xy_norm, 1e-8) * self.R
            center_of_tube = jnp.pad(p_on_maj_circ, (0, 1))
            normal = x_proj - center_of_tube
            normal = normal / jnp.maximum(jnp.linalg.norm(normal), 1e-8)

        v_on_normal = jnp.dot(v, normal) * normal  # type: ignore
        return v - v_on_normal

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Approximates the exponential map by moving along the tangent vector v
        and then projecting back onto the manifold."""
        return self.project(x + v)

    def sample_wrapped_normal(
        self, n: int, key, std: float = 0.5
    ) -> jnp.ndarray | None:
        """
        Generates the target distribution for the model to learn.
        """
        if self.kind == "sphere":
            # DEFINITIVE SPHERE FIX: Generate points in a Gaussian cloud
            # around a point on the sphere, then project them. This is the
            # simplest and most robust way to get a wrapped normal.
            mu = jnp.zeros((self.embedded_dim,)).at[-1].set(1.0)  # North pole

            # Generate noise around the mean point in the ambient space
            noise = jax.random.normal(key, shape=(n, self.embedded_dim)) * std
            points = mu + noise

            # Project the noisy points onto the sphere surface.
            return self.project(points)

        elif self.kind == "torus":
            # CORRECT TORUS LOGIC (DO NOT TOUCH): Sample uniformly from the
            # torus by sampling the underlying angles uniformly.
            phi_key, theta_key = jax.random.split(key)
            phi = jax.random.uniform(phi_key, shape=(n,), minval=0, maxval=2 * jnp.pi)
            theta = jax.random.uniform(
                theta_key, shape=(n,), minval=0, maxval=2 * jnp.pi
            )

            x = (self.R + self.r * jnp.cos(theta)) * jnp.cos(phi)
            y = (self.R + self.r * jnp.cos(theta)) * jnp.sin(phi)
            z = self.r * jnp.sin(theta)
            return jnp.stack([x, y, z], axis=-1)
