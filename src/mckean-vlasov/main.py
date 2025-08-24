# main.py — Stable training loop with SOTA Denoised Guidance
import argparse, json, time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import jax, jax.numpy as jnp
from dataloader import (
    load_packed_pt,
    train_val_split,
    iterate_batches,
    describe,
    invert_norm,
)
from models import UNet3D_FiLM, GuidanceNet, build_modules_embedder, time_embed
from losses_steps import create_full_train_state, train_step
from sampling import mv_sde_sample_guided
from functools import partial


def one_hot(y: np.ndarray, num_classes: int) -> jnp.ndarray:
    return jax.nn.one_hot(jnp.array(y), num_classes)


def main():
    ap = argparse.ArgumentParser()
    # I/O
    ap.add_argument("--data_pt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="runs/denoised_guidance")
    # Training
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="cosine")
    ap.add_argument(
        "--v_pred", action="store_true", help="Train with v-prediction objective."
    )
    ap.add_argument("--cfg_drop", type=float, default=0.1)
    # Denoised Guidance
    ap.add_argument(
        "--use_guidance", action="store_true", help="Enable denoised guidance."
    )
    ap.add_argument("--lr_guidance", type=float, default=5e-5)
    ap.add_argument("--guidance_loss_weight", type=float, default=0.05)
    ap.add_argument(
        "--guidance_scale",
        type=float,
        default=0.2,
        help="Guidance scale during sampling.",
    )
    # Sampling
    ap.add_argument("--sampler", type=str, default="mv_sde")
    ap.add_argument("--sample_steps", type=int, default=200)
    ap.add_argument("--sample_count", type=int, default=256)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    # Mean-field
    ap.add_argument("--mf_mode", type=str, default="rbf")
    ap.add_argument("--mf_lambda", type=float, default=0.01)
    ap.add_argument("--mf_bandwidth", type=float, default=0.5)
    # Misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S_max", type=int, default=16)
    args = ap.parse_args()

    outdir = Path(args.outdir) / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "chunks").mkdir(exist_ok=True)
    with open(outdir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    pack = load_packed_pt(args.data_pt, require_modules=True)
    describe(pack)
    train_ds, val_ds, norm_meta = train_val_split(
        pack, val_frac=0.1, seed=42, normalize=True
    )
    H, W, K, C = train_ds.vol.shape[1:]
    num_classes = int(pack.get("labels", np.array([-1])).max() + 1)

    rng, k_unet, k_guidance, k_enc = jax.random.split(rng, 4)
    embed_fn = build_modules_embedder(k_enc, out_dim=256, S_max=args.S_max)
    cond_dim = 256 + num_classes

    unet = UNet3D_FiLM(ch=64)
    guidance_net = GuidanceNet(ch=16)

    dummy_x = jnp.zeros((args.batch, H, W, K, C))
    dummy_t = time_embed(jnp.zeros((args.batch,)), 128)
    dummy_c = jnp.zeros((args.batch, cond_dim))

    unet_params = unet.init(k_unet, dummy_x, dummy_t, dummy_c)["params"]
    guidance_params = (
        guidance_net.init(k_guidance, dummy_x, dummy_t, dummy_c)["params"]
        if args.use_guidance
        else None
    )

    full_state = create_full_train_state(
        rng,
        unet.apply,
        unet_params,
        guidance_net.apply,
        guidance_params,
        total_steps=args.steps,
        T=args.T,
        lr=args.lr,
        lr_guidance=args.lr_guidance,
        schedule=args.schedule,
        ema_decay=args.ema_decay,
    )

    train_it = iterate_batches(train_ds, args.batch, shuffle=True, seed=args.seed)

    # JIT the training step function
    jitted_train_step = jax.jit(
        partial(
            train_step,
            v_pred=args.v_pred,
            guidance_loss_weight=args.guidance_loss_weight,
        )
    )

    print(
        f"[train] Starting {args.steps} steps... (v_pred: {args.v_pred}, guidance: {args.use_guidance})"
    )

    for step in range(1, args.steps + 1):
        batch = next(train_it)
        batch_jnp = {
            "vol": jnp.array(batch["vol"]),
            "cond": jnp.concatenate(
                [embed_fn(batch["modules"]), one_hot(batch["labels"], num_classes)],  # type: ignore
                axis=-1,
            ),
        }

        rng, step_rng = jax.random.split(rng)
        full_state, metrics = jitted_train_step(full_state, batch_jnp, step_rng)

        if step == 1 or step % 100 == 0 or step == args.steps:
            # block_until_ready() is crucial for accurate timing and memory management in a loop
            metrics["total_loss"].block_until_ready()
            loss = float(metrics["total_loss"])
            diff_loss = float(metrics["diff_loss"])
            guide_loss = float(metrics["guidance_loss"])
            print(
                f"step {step:05d}/{args.steps} | loss={loss:.4f} (diff={diff_loss:.4f}, guide={guide_loss:.4f})"
            )

    print("[sample] Starting generation...")
    val_it = iterate_batches(val_ds, args.batch, shuffle=False, seed=123, epochs=1)
    samples_all, labels_all = [], []
    total_target = args.sample_count
    produced = 0

    while produced < total_target:
        try:
            val_batch = next(val_it)
        except StopIteration:
            break

        take = min(len(val_batch["labels"]), total_target - produced)
        y_ref = np.array(val_batch["labels"][:take])
        cond_ref = jnp.concatenate(
            [embed_fn(val_batch["modules"][:take]), one_hot(y_ref, num_classes)],  # type: ignore
            axis=-1,
        )
        cond_uncond = jnp.zeros_like(cond_ref) if args.cfg_scale > 0 else None

        rng, subkey = jax.random.split(rng)
        shape = (take, H, W, K, C)

        xgen = mv_sde_sample_guided(
            unet_apply=unet.apply,
            unet_params=full_state.unet_state.ema_params,
            guidance_apply=guidance_net.apply,
            guidance_params=full_state.guidance_state.ema_params,
            shape=shape,
            betas=full_state.betas,
            alphas=full_state.alphas,
            alpha_bars=full_state.alpha_bars,
            cond_vec=cond_ref,
            cond_uncond_vec=cond_uncond,
            rng=subkey,
            steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
            v_pred=args.v_pred,
            use_guidance=args.use_guidance,
            guidance_scale=args.guidance_scale,
            mf_mode=args.mf_mode,
            mf_lambda=args.mf_lambda,
            mf_bandwidth=args.mf_bandwidth,
        )
        xgen.block_until_ready()

        x_np = np.array(xgen)
        if "lo" in norm_meta and "hi" in norm_meta:
            lo, hi = np.asarray(norm_meta["lo"]), np.asarray(norm_meta["hi"])
            x_np = invert_norm(x_np, lo, hi)

        samples_all.append(x_np)
        labels_all.append(y_ref)
        produced += take
        print(f"  Generated {produced}/{total_target} samples...")

    samples_np = np.concatenate(samples_all, 0)
    labels_np = np.concatenate(labels_all, 0)
    out_npz = outdir / "samples_landscapes.npz"
    np.savez_compressed(out_npz, samples=samples_np, labels=labels_np)
    print(f"[done] Saved {samples_np.shape[0]} samples to {out_npz}")


if __name__ == "__main__":
    main()
