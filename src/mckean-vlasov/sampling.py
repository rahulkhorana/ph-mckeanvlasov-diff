# sampling.py
import jax, jax.numpy as jnp
from models import time_embed


def make_module_bridge(E_apply, eparams, cond_vec, T: int):
    """
    cond_vec is *fixed* for sampling (B, cond_dim).
    Returns bridge_fn(x_t, t_cont, x0_hat) = ∇_{x0} E(x0, cond_vec).
    """

    def energy_mean(x0):
        e = E_apply({"params": eparams}, x0, cond_vec)  # (B,)
        return jnp.mean(e)

    grad_fn = jax.grad(energy_mean)

    def bridge_fn(x_t, t_cont, x0_hat):
        return grad_fn(x0_hat)

    return bridge_fn


def ddim_sample(
    unet_apply,
    params,
    shape,
    betas,
    alphas,
    alpha_bars,
    cond_vec,  # (B,cond_dim), fixed during sampling
    steps: int = 50,
    rng=None,
    v_prediction: bool = True,
    bridge_fn=None,
    bridge_scale: float = 1.0,  # module-bridge
    guidance_scale: float = 0.0,
    guidance_fn=None,  # (optional) extra guidance at x_t
    return_all: bool = False,
):
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).round().astype(jnp.int32)
    key = jax.random.PRNGKey(0) if rng is None else rng
    x = jax.random.normal(key, shape)
    traj = []

    for t in ts:
        a_t = alpha_bars[t]
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)]
        sqrt_ab = jnp.sqrt(a_t)
        sqrt_1ab = jnp.sqrt(1.0 - a_t)
        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)
        temb = time_embed(jnp.full((shape[0],), t_cont), dim=128)

        pred = unet_apply({"params": params}, x, temb, cond_vec)  # eps or v

        if v_prediction:
            eps = sqrt_ab * pred + sqrt_1ab * x
        else:
            eps = pred

        # optional x_t guidance (rarely needed now)
        if guidance_fn is not None and guidance_scale != 0.0:
            g = guidance_fn(x, t_cont)
            eps = eps - guidance_scale * sqrt_1ab * g

        # module bridge at x0_hat
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)
        if bridge_fn is not None and bridge_scale != 0.0:
            g_b = bridge_fn(x, t_cont, x0_hat)
            # larger near data manifold
            w_t = (1.0 - a_t) ** 0.5
            eps = eps - bridge_scale * w_t * g_b

        # update (DDIM, eta=0)
        x0_hat = (x - sqrt_1ab * eps) / jnp.clip(sqrt_ab, 1e-8)
        x = jnp.sqrt(a_tm1) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x
