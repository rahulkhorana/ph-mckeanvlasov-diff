import jax
import jax.numpy as jnp
from losses_steps import time_embed


def ddim_sample(
    unet_apply,
    params,
    shape,  # (B,H,W,C)
    betas,
    alphas,
    alpha_bars,
    guidance_fn=None,  # callable: grad_xE(x, t_cont) or None
    guidance_scale: float = 0.0,
    steps: int = 50,
    rng=None,
    v_prediction: bool = False,  # must match how the UNet was trained
):
    """
    Deterministic DDIM (eta=0) with optional energy guidance.
    guidance_fn must return ∇_x E(x, t_cont) with same shape as x.
    """
    T = int(alpha_bars.shape[0])

    if rng is None:
        rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, shape)

    # even subsampling of timesteps [T-1 ... 1]
    ts = jnp.linspace(T - 1, 1, steps, dtype=jnp.int32)

    for t in ts:
        # scalars for this t
        a_t = alpha_bars[t]  # ()
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)]  # ()

        # time embedding expects continuous in [0,1]
        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)  # center-of-bin
        temb = time_embed(jnp.full((shape[0],), t_cont, dtype=jnp.float32))

        pred = unet_apply({"params": params}, x, temb)  # eps or v

        if v_prediction:
            # convert v to eps (Salimans & Ho, 2022)
            sqrt_at = jnp.sqrt(a_t)
            sqrt_1mat = jnp.sqrt(1.0 - a_t)
            eps = (pred + sqrt_1mat * x) / jnp.clip(sqrt_at, 1e-8)
        else:
            eps = pred

        # energy guidance: eps_guided = eps - λ * sqrt(1-a_t) * ∇_x E
        if (guidance_fn is not None) and (guidance_scale != 0.0):
            g = guidance_fn(x, t_cont)  # ∇_x E
            eps = eps - guidance_scale * jnp.sqrt(1.0 - a_t) * g

        # DDIM deterministic update (eta=0)
        x0_hat = (x - jnp.sqrt(1.0 - a_t) * eps) / jnp.sqrt(jnp.clip(a_t, 1e-8))
        x = jnp.sqrt(a_tm1) * x0_hat + jnp.sqrt(jnp.clip(1.0 - a_tm1, 0.0)) * 0.0

    return x


def make_energy_guidance(E_apply, eparams, embed_fn, mods_batch):
    """
    embed_fn(mods_batch) -> m_emb (B, d) JAX array (precompute once)
    returns grad_fn(x, t_cont) that outputs ∇_x E_mean(x, M).
    """
    m_emb = embed_fn(mods_batch)  # (B,d), cached for the whole trajectory

    def energy_mean(x):
        # x: (B,H,W,C)
        e = E_apply({"params": eparams}, x, m_emb)  # (B,)
        return jnp.mean(e)

    grad_fn = jax.grad(energy_mean)  # ∇_x E_mean

    def guidance(x, t_cont, clip=1.0):
        g = grad_fn(x)
        n = jnp.linalg.norm(g.reshape(g.shape[0], -1), axis=1, keepdims=True) + 1e-8
        g = g / jnp.maximum(n, clip)[:, None, None, None]
        return g

    return guidance
