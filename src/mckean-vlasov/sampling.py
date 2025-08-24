# sampling.py — MV-SDE with CFG, v-pred, and SOTA Denoised Guidance
import jax, jax.numpy as jnp
from models import time_embed
from functools import partial


# ---------- helpers ----------
def _cfg_pred(apply_fn, params, x, temb, cond, cfg_scale: float, uncond):
    if uncond is None or cfg_scale == 0.0:
        return apply_fn({"params": params}, x, temb, cond)
    pred_u = apply_fn({"params": params}, x, temb, uncond)
    pred_c = apply_fn({"params": params}, x, temb, cond)
    return pred_u + cfg_scale * (pred_c - pred_u)


def _mf_none(*args, **kw):
    return 0.0


def _mf_rbf_drift(x0_hat, bandwidth: float = 0.5, kernel_size: int = 3):
    k = jnp.array([0.25, 0.5, 0.25], jnp.float32)

    def blur2d(img):
        pad_h = jnp.pad(img, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)), "edge")
        yh = k[0] * pad_h[:, :-2] + k[1] * pad_h[:, 1:-1] + k[2] * pad_h[:, 2:]
        pad_w = jnp.pad(yh, ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)), "edge")
        yw = k[0] * pad_w[:, :, :-2] + k[1] * pad_w[:, :, 1:-1] + k[2] * pad_w[:, :, 2:]
        return yw

    return (blur2d(x0_hat) - x0_hat) * float(bandwidth)


def _choose_mf(mode: str):
    return _mf_rbf_drift if mode == "rbf" else _mf_none


# ---------- Sampler with Denoised Guidance ----------
@partial(
    jax.jit,
    static_argnames=(
        "unet_apply",
        "guidance_apply",
        "steps",
        "v_pred",
        "mf_mode",
        "return_all",
    ),
)
def mv_sde_sample_guided(
    unet_apply,
    unet_params,
    guidance_apply,
    guidance_params,
    shape,
    betas,
    alphas,
    alpha_bars,
    cond_vec,
    cond_uncond_vec,
    rng,
    steps: int,
    cfg_scale: float,
    v_pred: bool,
    use_guidance: bool,
    guidance_scale: float,
    mf_mode: str,
    mf_lambda: float,
    mf_bandwidth: float,
    return_all: bool = False,
):
    T = alpha_bars.shape[0]
    ts = jnp.linspace(T - 1, 1, steps).round().astype(jnp.int32)
    x = jax.random.normal(rng, shape)
    traj = []
    mf_fn = _choose_mf(mf_mode)

    for t_idx in ts:
        t = jnp.full((shape[0],), t_idx, dtype=jnp.int32)
        a_t = alpha_bars[t][:, None, None, None, None]
        a_tm1 = alpha_bars[jnp.maximum(t - 1, 0)][:, None, None, None, None]
        sqrt_ab, sqrt_1ab = jnp.sqrt(a_t), jnp.sqrt(1.0 - a_t)

        t_cont = (t.astype(jnp.float32) + 0.5) / float(T)
        temb = time_embed(jnp.squeeze(t_cont), dim=128)

        prediction = _cfg_pred(
            unet_apply, unet_params, x, temb, cond_vec, cfg_scale, cond_uncond_vec
        )
        x0_hat_unet = (
            (sqrt_ab * x - sqrt_1ab * prediction)
            if v_pred
            else ((x - sqrt_1ab * prediction) / jnp.clip(sqrt_ab, 1e-8))
        )

        if use_guidance:
            x0_hat_guidance = guidance_apply(
                {"params": guidance_params}, x, temb, cond_vec
            )
            x0_hat = x0_hat_unet + guidance_scale * (x0_hat_guidance - x0_hat_unet)
        else:
            x0_hat = x0_hat_unet

        if mf_mode != "none" and mf_lambda != 0.0:
            drift = mf_fn(x0_hat, bandwidth=mf_bandwidth)
            x0_hat = x0_hat + mf_lambda * drift

        x = jnp.sqrt(a_tm1) * x0_hat

        if return_all:
            traj.append(x)

    return jnp.stack(traj, 1) if return_all else x
