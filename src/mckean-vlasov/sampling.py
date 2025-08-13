import jax
import jax.numpy as jnp
from losses_steps import time_embed


def make_energy_guidance(E_apply, eparams, embed_fn, modules_batch, T: int):
    """
    Precompute m_emb once; returns guidance(x, t_cont) → ∇_x E normalized+scheduled.
    """
    m_emb = embed_fn(modules_batch)  # (B,d)

    def energy_mean(x):
        e = E_apply({"params": eparams}, x, m_emb)  # (B,)
        return jnp.mean(e)

    grad_fn = jax.grad(energy_mean)

    def guidance(x, t_cont):
        g = grad_fn(x)  # (B,H,W,C)
        # per-sample L2 norm
        B = x.shape[0]
        g_flat = g.reshape(B, -1)
        nrm = jnp.linalg.norm(g_flat, axis=1, keepdims=True)
        g = (g_flat / (nrm + 1e-8)).reshape(*x.shape)
        # cosine schedule (small early, stronger late)
        w_t = jnp.cos(0.5 * jnp.pi * t_cont)  # (B,)
        g = g * w_t[:, None, None, None]
        # clip
        g = jnp.clip(g, -3.0, 3.0)
        return g

    return guidance


def ddim_sample(
    unet_apply,
    params,
    shape,
    betas,
    alphas,
    alpha_bars,
    guidance_fn=None,
    guidance_scale: float = 1.0,
    steps: int = 50,
    rng=None,
    v_prediction: bool = True,
    return_all: bool = False,
):
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).astype(jnp.int32)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, shape)

    traj = []
    for t in ts:
        a_t = alpha_bars[t]
        a_prev = alpha_bars[jnp.maximum(t - 1, 0)]
        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)

        temb = time_embed(jnp.full((shape[0],), t_cont), dim=128)
        pred = unet_apply({"params": params}, x, temb)  # eps or v

        if v_prediction:
            sqrt_ab = jnp.sqrt(a_t)
            sqrt_1ab = jnp.sqrt(1.0 - a_t)
            eps = (pred + sqrt_1ab * x) / jnp.clip(sqrt_ab, 1e-8)
        else:
            eps = pred

        if guidance_fn is not None and guidance_scale != 0.0:
            g = guidance_fn(x, jnp.full((shape[0],), t_cont))
            eps = eps - guidance_scale * g * jnp.sqrt(1.0 - a_t)

        # DDIM (eta=0)
        x0_hat = (x - jnp.sqrt(1.0 - a_t) * eps) / jnp.sqrt(a_t)
        x = jnp.sqrt(a_prev) * x0_hat

        if return_all:
            traj.append(x)

    if return_all:
        return jnp.stack(traj, axis=1)  # (B, steps, H, W, C)
    return x  # (B,H,W,C)
