import jax
import jax.numpy as jnp
from typing import Literal


class ManifoldWrapper:
    """
    A robust, geometrically correct wrapper for manifold operations.
    This version uses a definitive, calculus-based tangent space calculation
    for the torus to fix all prior geometric failures.
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

    def project_to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Projects a vector v onto the tangent space at point x."""
        if self.kind == "sphere":
            # For a sphere, the normal is the normalized position vector.
            normal = x / jnp.maximum(jnp.linalg.norm(x), 1e-8)
            v_on_normal = jnp.dot(v, normal) * normal
            return v - v_on_normal

        elif self.kind == "torus":
            # DEFINITIVE FIX: First, project the point x for which we are
            # calculating the tangent plane onto the torus. This ensures that
            # the geometric calculations for the angles are valid.
            x_proj = self._project_single(x)

            # Now, calculate the tangent plane at the projected point, x_proj.
            phi = jnp.arctan2(x_proj[1], x_proj[0])
            xy_dist_from_origin = jnp.linalg.norm(x_proj[:2])

            # We need cos(theta) and sin(theta) for the derivative formulas.
            # Clip values to avoid numerical instability from projection errors.
            cos_theta = jnp.clip((xy_dist_from_origin - self.R) / self.r, -1.0, 1.0)
            sin_theta = jnp.clip(x_proj[2] / self.r, -1.0, 1.0)

            # The two orthogonal (but not yet unit) tangent vectors derived
            # from the partial derivatives of the torus's parametric equations.
            t_phi = jnp.array(
                [
                    -(self.R + self.r * cos_theta) * jnp.sin(phi),
                    (self.R + self.r * cos_theta) * jnp.cos(phi),
                    0.0,
                ]
            )
            t_theta = jnp.array(
                [
                    -self.r * sin_theta * jnp.cos(phi),
                    -self.r * sin_theta * jnp.sin(phi),
                    self.r * cos_theta,
                ]
            )

            # Normalize them to get an orthonormal basis {e1, e2} for the tangent plane.
            e1 = t_phi / jnp.maximum(jnp.linalg.norm(t_phi), 1e-8)
            # Use Gram-Schmidt to ensure orthogonality, making it more robust.
            t_theta_ortho = t_theta - jnp.dot(t_theta, e1) * e1
            e2 = t_theta_ortho / jnp.maximum(jnp.linalg.norm(t_theta_ortho), 1e-8)

            # Project the ambient vector v onto this basis to find the component
            # that lies in the tangent plane: v_tangent = <v, e1>e1 + <v, e2>e2
            v_on_tangent = jnp.dot(v, e1) * e1 + jnp.dot(v, e2) * e2
            return v_on_tangent

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Approximates the exponential map by moving along the tangent vector v
        and then projecting back onto the manifold."""
        return self.project(x + v)

    def sample_wrapped_normal(self, n: int, key, mu=None, std=0.4) -> jnp.ndarray:
        """
        Samples points from the manifold.
        - For a sphere, this samples from a wrapped normal distribution.
        - For a torus, this samples uniformly to cover the entire surface.
        """
        if self.kind == "sphere":
            # This is the standard, working logic for the sphere.
            if mu is None:
                mu = jnp.zeros((self.embedded_dim,)).at[-1].set(1.0)  # North pole
            mu = self.project(mu)
            z = jax.random.normal(key, shape=(n, self.embedded_dim)) * std
            mu_batched = jnp.broadcast_to(mu, (n, self.embedded_dim))
            v_proj = jax.vmap(self.project_to_tangent)(mu_batched, z)
            return jax.vmap(self.exp_map)(mu_batched, v_proj)

        elif self.kind == "torus":
            # Sample uniformly from the torus by sampling the underlying angles.
            phi_key, theta_key = jax.random.split(key)
            phi = jax.random.uniform(phi_key, shape=(n,), minval=0, maxval=2 * jnp.pi)
            theta = jax.random.uniform(
                theta_key, shape=(n,), minval=0, maxval=2 * jnp.pi
            )

            x_coords = (self.R + self.r * jnp.cos(theta)) * jnp.cos(phi)
            y_coords = (self.R + self.r * jnp.cos(theta)) * jnp.sin(phi)
            z_coords = self.r * jnp.sin(theta)
            return jnp.stack([x_coords, y_coords, z_coords], axis=-1)
