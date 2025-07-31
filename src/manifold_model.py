# manifold_model.py
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from tqdm import trange

from manifold_utils import ManifoldWrapper
from score_network import ScoreNet


def project_to_tangent(x, v):
    return v - jnp.sum(x * v, axis=-1, keepdims=True) * x


def make_loss_fn(static_model, manifold, sigma_fn):
    def loss_fn(params, x0, key):
        model = eqx.combine(params, static_model)
        B = x0.shape[0]

        key_t, key_noise = jax.random.split(key)
        t = jax.random.uniform(key_t, (B,), minval=1e-3, maxval=1.0)
        z = jax.random.normal(key_noise, x0.shape)
        ξ = jax.vmap(manifold.project_to_tangent)(x0, z)

        xt = x0 + sigma_fn(t)[:, None] * ξ
        xt = manifold.project(xt)

        target = -ξ / sigma_fn(t)[:, None]
        score_pred = jax.vmap(lambda x_i, t_i: model(x_i, t_i))(xt, t)  # <- FIXED
        loss = jnp.mean(jnp.sum((score_pred - target) ** 2, axis=-1))
        loss = jnp.where(jnp.isnan(loss), 1e6, loss)
        return loss

    return loss_fn


def train(kind="sphere", geom_dim=2, steps=3000, B=64):
    key = jax.random.PRNGKey(0)

    # ---- Geometry ----
    manifold = ManifoldWrapper(kind, geom_dim)  # type: ignore
    D = manifold.embedded_dim

    # ---- Model and Optim ----
    key, subkey = jax.random.split(key)
    model = ScoreNet(dim=D, key=subkey)
    params = eqx.filter(model, eqx.is_array)
    static = eqx.filter(model, lambda x: not eqx.is_array(x))
    opt = optax.adam(1e-4)
    opt_state = opt.init(params)

    sigma_fn = lambda t: 0.2 + 1.5 * t
    loss_fn = make_loss_fn(static, manifold, sigma_fn)

    @eqx.filter_jit
    def train_step(params, opt_state, x0, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, x0, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # ---- Training Loop ----
    for i in trange(steps):
        key, subkey = jax.random.split(key)
        x0 = manifold.sample_wrapped_normal(B, subkey)
        params, opt_state, loss = train_step(params, opt_state, x0, subkey)

        if i % 100 == 0:
            print(f"step {i:4d}  loss {float(loss):.4f}")


if __name__ == "__main__":
    train(kind="torus", geom_dim=2)
