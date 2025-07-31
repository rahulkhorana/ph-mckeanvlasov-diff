# Riemannian Score-Based Generative Model (SGM) with Itô SDEs on Manifolds

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Callable, Any
from tqdm import trange


# ----------- Manifold Utilities -----------
class ManifoldWrapper:
    def __init__(self, kind: str, geom_dim: int):
        self.kind = kind
        self.geom_dim = geom_dim
        self.embedded_dim = self._get_embedded_dim()

    def _get_embedded_dim(self):
        if self.kind == "sphere":
            return self.geom_dim + 1
        elif self.kind == "torus":
            return 2 * self.geom_dim
        else:
            raise NotImplementedError

    def sample_wrapped_normal(self, B: int, key):
        z = jax.random.normal(key, (B, self.embedded_dim))
        return self.project(z)

    def project(self, x):
        if self.kind == "sphere":
            return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        elif self.kind == "torus":
            # Map to unit circle per pair
            return jnp.concatenate(
                [
                    x[..., 2 * i : 2 * i + 2]
                    / (
                        jnp.linalg.norm(
                            x[..., 2 * i : 2 * i + 2], axis=-1, keepdims=True
                        )
                        + 1e-6
                    )
                    for i in range(self.geom_dim)
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError

    def project_to_tangent(self, x, v):
        if self.kind == "sphere":
            return v - jnp.sum(x * v, axis=-1, keepdims=True) * x
        elif self.kind == "torus":
            return v  # Euclidean tangent bundle (flat)
        else:
            raise NotImplementedError


# ----------- Score Network -----------
def sinusoidal_embed(t, dim):
    half_dim = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000), half_dim))
    angles = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class ScoreNet(eqx.Module):
    mlp: eqx.nn.MLP
    t_dim: int

    def __init__(self, dim, t_dim=16, width=128, depth=3, *, key):
        self.t_dim = t_dim
        t = jax.random.split(key, 1)[0]
        self.mlp = eqx.nn.MLP(
            in_size=dim + t_dim,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=t,
        )

    def __call__(self, x, t):
        if x.ndim == 1:
            x = x[None, :]
        if t.ndim == 0:
            t = t[None]

        t_embed = sinusoidal_embed(t, self.t_dim)
        if t_embed.ndim == 1:
            t_embed = t_embed[None, :]

        # inp = jnp.concatenate([x, t_embed], axis=-1)
        inp = jnp.concatenate(
            [x, jnp.broadcast_to(t_embed, (*x.shape[:-1], t_embed.shape[-1]))], axis=-1
        )
        out = jax.vmap(self.mlp)(inp)
        out = jnp.clip(out, -10.0, 10.0)  # or jnp.tanh(out) * scale
        return out


# ----------- Loss Function -----------
def make_loss_fn(model: ScoreNet, manifold: ManifoldWrapper, sigma_fn: Callable):
    def loss_fn(params, x0, key):
        B = x0.shape[0]
        key_t, key_noise = jax.random.split(key)
        t = jax.random.uniform(key_t, (B,), minval=1e-3, maxval=1.0)
        z = jax.random.normal(key_noise, x0.shape)
        ξ = jax.vmap(manifold.project_to_tangent)(x0, z)

        xt = x0 + sigma_fn(t)[:, None] * ξ
        xt = manifold.project(xt)
        target = -ξ / sigma_fn(t)[:, None]

        score_pred = jax.vmap(lambda x_i, t_i: model(x_i, t_i))(xt, t)
        return jnp.mean(jnp.sum((score_pred - target) ** 2, axis=-1))

    return loss_fn


# ----------- Euler-Maruyama Sampler -----------
def euler_maruyama_step(x, t, dt, model, manifold, key, sigma_fn):
    noise = jax.random.normal(key, x.shape)
    noise_proj = jax.vmap(manifold.project_to_tangent)(x, noise)
    g = sigma_fn(t)[:, None]

    drift = jax.vmap(lambda x_, t_: model(x_, t_))(x, t)
    drift = jax.vmap(manifold.project_to_tangent)(x, drift)

    x_next = x + drift * dt + g * noise_proj * jnp.sqrt(dt)
    return manifold.project(x_next)


def sample_sde(model, xT, ts, manifold, key, sigma_fn):
    x = xT
    xs = []
    keys = jax.random.split(key, len(ts) - 1)
    for i in range(len(ts) - 1):
        t = jnp.full((x.shape[0],), ts[i])
        # dt = ts[i + 1] - ts[i]
        dt = jnp.clip(ts[i + 1] - ts[i], 1e-5, 1e-2)
        x = euler_maruyama_step(x, t, dt, model, manifold, keys[i], sigma_fn)
        xs.append(jnp.copy(x))
        # print(
        #   f"Step {i}: shape = {x.shape}, dtype = {x.dtype}, x[0].shape = {x[0].shape}"
        # )
    for i, x in enumerate(xs):
        if x.shape != xs[0].shape:
            print(f"Mismatch at step {i}: xs0shp: {xs[0].shape} shape = {x.shape}")
    return jnp.stack(xs)


# ----------- Training -----------
def train(kind="sphere", geom_dim=2, steps=3000, B=64):
    key = jax.random.PRNGKey(0)
    manifold = ManifoldWrapper(kind, geom_dim)
    D = manifold.embedded_dim

    key, subkey = jax.random.split(key)
    model = ScoreNet(dim=D, key=subkey)
    params = eqx.filter(model, eqx.is_array)
    static = eqx.filter(model, lambda x: not eqx.is_array(x))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    sigma_fn = lambda t: jnp.clip(0.2 + 1.5 * t, 0.2, 2.0)
    loss_fn = make_loss_fn(eqx.combine(params, static), manifold, sigma_fn)

    @jax.jit
    def train_step(params, opt_state, x0, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, x0, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in trange(steps):
        key, subkey = jax.random.split(key)
        x0 = manifold.sample_wrapped_normal(B, subkey)
        params, opt_state, loss = train_step(params, opt_state, x0, subkey)
        if i % 100 == 0:
            print(f"step {i:4d} | loss {float(loss):.4f}")

    return eqx.combine(params, static), manifold


if __name__ == "__main__":
    model, manifold = train(kind="torus", geom_dim=2)
    ts = jnp.linspace(1.0, 1e-3, 100)
    key = jax.random.PRNGKey(42)
    xT = manifold.sample_wrapped_normal(64, key)
    samples = sample_sde(
        model,
        xT,
        ts,
        manifold,
        key,
        sigma_fn=lambda t: jnp.clip(0.2 + 1.5 * t, 0.2, 2.0),
    )

    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    def plot_pca_3d(samples_4d):
        # samples_4d: Array[T, B, 4]
        final_samples = np.asarray(samples_4d)  # shape: (B, 4)

        # Project to 3D using PCA
        final_samples = samples_4d  # shape (B, D) → expected (num_samples, 4)
        # Convert from JAX array if needed
        if isinstance(samples_4d, jax.Array):
            samples_4d = np.array(samples_4d)

        # Use final timestep and flatten (B, N, D) → (B*N, D)
        final_samples = samples_4d[-1].reshape(-1, samples_4d.shape[-1])  # (4096, 4)

        # PCA to 3D
        pca = PCA(n_components=3)
        projected = pca.fit_transform(final_samples)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            projected[:, 0], projected[:, 1], projected[:, 2], c="blue", alpha=0.6
        )
        ax.set_title("PCA of Final Sample Distribution (Flattened to 3D)")
        plt.tight_layout()
        plt.show()

    plot_pca_3d(samples)

    def plot_3d_projection(samples):
        """
        Plot the final samples projected to 3D via axis selection.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if isinstance(samples, jax.Array):
            samples = np.array(samples)

        final_samples = samples[-1]  # shape: (B, N, 4)
        flattened = final_samples.reshape(-1, 4)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Just plot first 3 dims: (x, y, z)
        ax.scatter(
            flattened[:, 0],
            flattened[:, 1],
            flattened[:, 2],
            c="purple",
            alpha=0.6,
        )
        ax.set_title("3D Projection of Final Samples (dims 0-1-2)")
        plt.show()

    plot_3d_projection(samples)

    def plot_tsne(samples):
        """
        Flatten and project final samples with t-SNE into 2D.
        """
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        if isinstance(samples, jax.Array):
            samples = np.array(samples)

        final_samples = samples[-1].reshape(-1, samples.shape[-1])  # (B*N, 4)

        tsne = TSNE(n_components=2, perplexity=30, learning_rate=100)
        embedded = tsne.fit_transform(final_samples)

        plt.scatter(embedded[:, 0], embedded[:, 1], c="green", s=3, alpha=0.7)
        plt.title("t-SNE of Final Sample Distribution (2D)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    plot_tsne(samples)
