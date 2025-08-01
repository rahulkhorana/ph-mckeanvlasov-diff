import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import trange
from typing import Callable

from manifold_toy_models.manifold_utils import ManifoldWrapper


# --- SDE and Model Definition ---


def sinusoidal_embed(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Standard sinusoidal time embedding."""
    half_dim = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000.0), half_dim))
    angles = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class ScoreNet(eqx.Module):
    """A neural network to approximate the score function."""

    mlp: eqx.nn.MLP
    time_embed_dim: int

    def __init__(
        self,
        dim: int,
        time_embed_dim: int = 32,
        width: int = 256,
        depth: int = 4,
        *,
        key,
    ):
        self.time_embed_dim = time_embed_dim
        mlp_key, _ = jax.random.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=dim + time_embed_dim,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=mlp_key,
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # This function is designed to be vmapped.
        # It takes a single data point x of shape (dim,)
        # and a single time t of shape () or (1,).

        # Ensure t is array-like for sinusoidal_embed
        if hasattr(t, "shape") and t.ndim == 0:
            t = t[None]  # Shape becomes (1,)

        # t_embed will have shape (1, time_embed_dim)
        t_embed = sinusoidal_embed(t, self.time_embed_dim)

        # Squeeze out the batch dimension from the time embedding
        t_embed_squeezed = jnp.squeeze(
            t_embed, axis=0
        )  # Shape becomes (time_embed_dim,)

        # Concatenate the single data point with its time embedding
        # x has shape (dim,), t_embed_squeezed has shape (time_embed_dim,)
        inp = jnp.concatenate(
            [x, t_embed_squeezed]
        )  # Shape becomes (dim + time_embed_dim,)

        # The MLP expects an input of shape (in_size,) for a single example
        return self.mlp(inp)


# --- VP SDE Formulation (Variance Preserving) ---

BETA_MIN = 0.1
BETA_MAX = 20.0


def sde_beta(t: jnp.ndarray) -> jnp.ndarray:
    """The beta(t) schedule for the VP SDE."""
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def sde_integral_beta(t: jnp.ndarray) -> jnp.ndarray:
    """The analytical integral of beta(s) from 0 to t."""
    return BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t**2


def vp_sde_perturbation(
    x0: jnp.ndarray, t: jnp.ndarray, z: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perturbs data x0 at time t according to the VP SDE."""
    integral_beta_t = sde_integral_beta(t)
    alpha_t = jnp.exp(-0.5 * integral_beta_t)
    sigma_t = jnp.sqrt(1.0 - jnp.exp(-integral_beta_t))

    xt = alpha_t[:, None] * x0 + sigma_t[:, None] * z
    return xt, sigma_t[:, None]


# --- Training ---


def loss_fn(model: ScoreNet, x0: jnp.ndarray, key) -> jnp.ndarray:
    """Denoising score matching loss for the VP SDE."""
    t_key, noise_key = jax.random.split(key)
    t = jax.random.uniform(t_key, (x0.shape[0],), minval=1e-5, maxval=1.0)
    z = jax.random.normal(noise_key, x0.shape)

    xt, std = vp_sde_perturbation(x0, t, z)
    score_pred = jax.vmap(model)(xt, t)

    # The loss is E[lambda(t) * ||s_theta(xt, t) - score_true||^2]
    # With weighting lambda(t) = std^2, this simplifies to E[||s_theta*std + z||^2]
    loss = jnp.mean(jnp.sum((score_pred * std + z) ** 2, axis=-1))
    return loss


def train(
    model: ScoreNet,
    manifold: ManifoldWrapper,
    *,
    steps: int = 5000,
    lr: float = 3e-4,
    batch_size: int = 1024,
    key,
) -> ScoreNet:
    """Training loop for the score model."""
    opt = optax.adamw(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x0_batch, key):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x0_batch, key)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    data_key, train_key = jax.random.split(key)
    pbar = trange(steps)
    for step in pbar:
        step_data_key = jax.random.fold_in(data_key, step)
        step_train_key = jax.random.fold_in(train_key, step)
        x0_batch = manifold.sample_wrapped_normal(batch_size, key=step_data_key)

        model, opt_state, loss = train_step(model, opt_state, x0_batch, step_train_key)

        if step % 100 == 0:
            pbar.set_description(f"Step {step}, Loss = {loss:.4f}")

    return model


# --- Sampling ---


def sample_sde(
    model: ScoreNet,
    manifold: ManifoldWrapper,
    *,
    shape: tuple,
    n_steps: int = 1000,
    key,
) -> jnp.ndarray:
    """Samples from the model using the reverse-time SDE."""

    @eqx.filter_jit
    def sampling_loop(x_init, ts, keys):
        def loop_body(i, x):
            t = ts[i]
            vec_t = jnp.full((x.shape[0],), t)

            score = jax.vmap(model)(x, vec_t)

            # Ensure beta_t is a vector with the correct batch dimension for broadcasting.
            beta_t = jnp.full((x.shape[0],), sde_beta(t))

            # Corrected reverse SDE drift with explicit broadcasting
            drift = 0.5 * beta_t[:, None] * x + beta_t[:, None] * score

            # Corrected diffusion with explicit broadcasting
            diffusion = jnp.sqrt(beta_t[:, None])

            dt = ts[i] - ts[i + 1]  # Timestep is positive

            noise = jax.random.normal(keys[i], x.shape)
            x_next = x + drift * dt + diffusion * jnp.sqrt(dt) * noise

            # Project back to the manifold at each step
            x_next = manifold.project(x_next)

            return x_next

        x = jax.lax.fori_loop(0, n_steps - 1, loop_body, x_init)
        return x

    t_init_key, loop_keys_key = jax.random.split(key)
    x_init = jax.random.normal(t_init_key, shape)
    x_init = manifold.project(x_init)

    ts = jnp.linspace(1.0, 1e-5, n_steps)
    keys = jax.random.split(loop_keys_key, n_steps - 1)

    samples = sampling_loop(x_init, ts, keys)
    return samples
