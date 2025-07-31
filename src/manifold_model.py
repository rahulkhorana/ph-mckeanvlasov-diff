# manifold_model.py
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Callable

from score_network import ScoreNet
from manifold_utils import ManifoldWrapper


def make_loss_fn(score_model, manifold, sigma_fn):
    assert isinstance(manifold, ManifoldWrapper)

    def loss_fn(params, x0, key):
        key_t, key_noise = jax.random.split(key)
        batch_size = x0.shape[0]
        t = jax.random.uniform(key_t, (batch_size,), minval=1e-3, maxval=1.0)
        z = jax.random.normal(key_noise, shape=x0.shape)
        noise = jax.vmap(lambda x, n: manifold.project_to_tangent(x, n))(x0, z)
        xt = x0 + sigma_fn(t)[:, None] * noise
        xt = jax.vmap(manifold.project)(xt)
        score = score_model(xt, t)
        target = -noise / sigma_fn(t)[:, None]
        loss = jnp.mean(jnp.sum((score - target) ** 2, axis=-1))
        return loss

    return loss_fn


def train():
    key = jax.random.PRNGKey(0)
    batch_size = 64
    dim = 3
    manifold = ManifoldWrapper("sphere", dim=dim - 1)  # S^2 in R^3
    model_key, data_key = jax.random.split(key)
    model = ScoreNet(dim=dim, key=model_key)
    sigma_fn = lambda t: 0.2 + 1.5 * t

    optimizer = optax.adam(learning_rate=1e-3)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def train_step(params, static_model, opt_state, x0, key):
        def loss_fn_wrap(p, x, k):
            model_p = eqx.combine(static_model, p)
            loss = make_loss_fn(model_p, manifold, sigma_fn)(p, x, k)
            return loss

        loss, grads = jax.value_and_grad(loss_fn_wrap)(params, x0, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for step in trange(3000):
        data_key, sample_key, noise_key = jax.random.split(data_key, 3)
        mu = jnp.zeros((dim,))
        mu = manifold.normalize(mu + jnp.array([0.0] * (dim - 1) + [1.0]))
        x0 = manifold.sample_wrapped_normal(
            n=batch_size, key=sample_key, mu=mu, std=0.25
        )
        model_static = eqx.filter(model, eqx.is_inexact_array)
        params, opt_state, loss = train_step(
            params, model_static, opt_state, x0, noise_key
        )

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")


if __name__ == "__main__":
    train()
