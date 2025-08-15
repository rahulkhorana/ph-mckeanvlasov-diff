from typing import Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp
from models import time_embed


def ddim_sample(
    unet_apply,
    params,
    shape,
    betas,
    alphas,
    alpha_bars,
    guidance_fn=None,
    guidance_scale=1.0,
    steps=50,
    rng=None,
    v_prediction=True,
    return_all=False,
    m_emb=None,
):
    """
    unet_apply(x, t_emb, m_emb) -> eps or v
    m_emb: (B, Dm) fixed during sampling (from chosen modules)
    """
    T = alpha_bars.shape[0]
    ts = np.linspace(T - 1, 1, steps, dtype=int)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, shape)

    traj = []
    for t in ts:
        a_t = alpha_bars[t]
        a_prev = alpha_bars[max(t - 1, 0)]
        a_t_b = jnp.broadcast_to(a_t, (shape[0],)).astype(jnp.float32)
        t_cont = (jnp.full((shape[0],), t, jnp.float32) + 0.5) / float(T)
        temb = time_embed(t_cont, dim=128)

        pred = unet_apply({"params": params}, x, temb, m_emb)  # eps or v

        if v_prediction:
            sqrt_ab = jnp.sqrt(a_t)
            sqrt_1ab = jnp.sqrt(1.0 - a_t)
            eps = (pred + sqrt_1ab * x) / jnp.clip(sqrt_ab, 1e-8)
        else:
            eps = pred

        if guidance_fn is not None and guidance_scale != 0.0:
            g = guidance_fn(x, t_cont)  # ∇_x (mean energy)
            eps = eps - guidance_scale * g * jnp.sqrt(1.0 - a_t)

        x0_hat = (x - jnp.sqrt(1.0 - a_t) * eps) / jnp.sqrt(a_t)
        x = jnp.sqrt(a_prev) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, axis=1) if return_all else x


def make_energy_guidance(E_apply, eparams, m_emb_fixed):
    def energy_mean(x, _t):
        e = E_apply({"params": eparams}, x, m_emb_fixed)  # (B,)
        return jnp.mean(e)

    return jax.grad(energy_mean)
