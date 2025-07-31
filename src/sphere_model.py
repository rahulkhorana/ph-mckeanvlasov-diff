import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geomstats.geometry.hypersphere import Hypersphere
from jax.random import PRNGKey

# Manifold setup
sphere = Hypersphere(dim=2)


# Project to tangent space at x
def project_to_tangent(x, v):
    return v - jnp.sum(x * v, axis=-1, keepdims=True) * x


# Score network
def get_score_model(key):
    return ScoreNet(dim=3, key=key)


class ScoreNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, dim, width=128, depth=3, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=dim + 1,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, x, t):
        xt = jnp.concatenate([x, jnp.expand_dims(t, -1)], axis=-1)
        score = self.mlp(xt)
        return project_to_tangent(x, score)


# von Mises-Fisher sampling on sphere
mean = jnp.array([1.0, 0.0, 0.0])


def sample_data(n, key):
    return sphere.random_von_mises_fisher(mu=mean, kappa=20, n_samples=n)


# Score matching loss generator
def make_loss_fn(static):
    def loss_fn(params, x0, key, sigma=1.0):
        model = eqx.combine(params, static)
        batch_size = x0.shape[0]
        t_key, noise_key = jax.random.split(key)
        t = jax.random.uniform(t_key, (batch_size,), minval=1e-3, maxval=1.0)
        noise = jax.random.normal(noise_key, x0.shape)
        noise = jax.vmap(project_to_tangent)(x0, noise)
        xt = x0 + jnp.expand_dims(jnp.sqrt(t) * sigma, -1) * noise
        xt = xt / jnp.linalg.norm(xt, axis=-1, keepdims=True)
        score_pred = jax.vmap(model)(xt, t)
        target = -noise / jnp.expand_dims(sigma * jnp.sqrt(t), -1)
        return jnp.mean((score_pred - target) ** 2)

    return loss_fn


# Reverse SDE sampling
def sample(model, shape, timesteps, key, sigma=1.0, dt=1e-3):
    x = jax.random.normal(key, shape)
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    for t in reversed(jnp.linspace(1.0, 1e-3, timesteps)):
        t_val = jnp.full((x.shape[0],), t)
        score = jax.vmap(model)(x, t_val)
        noise = jax.random.normal(key, x.shape)
        noise = jax.vmap(project_to_tangent)(x, noise)
        drift = -0.5 * sigma**2 * score
        diffusion = sigma * noise * jnp.sqrt(dt)
        x = x + drift * dt + diffusion
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x


# Training loop
def train():
    key = PRNGKey(42)
    model = get_score_model(key)
    params = eqx.filter(model, eqx.is_array)  # only arrays
    static = eqx.filter(model, lambda x: not eqx.is_array(x))  # only non-array parts
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def train_step(params, static, opt_state, x0, key):
        loss_fn = make_loss_fn(static)
        loss, grads = jax.value_and_grad(loss_fn)(params, x0, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for step in tqdm.trange(3000):
        key, subkey = jax.random.split(key)
        x0 = sample_data(64, subkey)
        params, opt_state, loss = train_step(params, static, opt_state, x0, key)
        if step % 250 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    # Final sampling
    model = eqx.combine(params, static)
    sample_key = PRNGKey(999)
    samples = sample(model, shape=(1000, 3), timesteps=1000, key=sample_key)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))  # type: ignore
    ax.set_title("Samples from Learned von Mises-Fisher Distribution on S^2")
    plt.show()


if __name__ == "__main__":
    train()
