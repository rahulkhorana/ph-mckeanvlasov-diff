# main.py — MV-SDE first; ε-pred, CFG, dataset percentile norm, save un-normalized samples
import argparse, json, time
from pathlib import Path
from typing import List

import numpy as np
import jax, jax.numpy as jnp

from dataloader import (
    load_packed_pt,
    train_val_split,
    iterate_batches,
    describe,
    invert_norm,
)
from models import UNet3D_FiLM, build_modules_embedder, time_embed
from losses_steps import create_diffusion_state, diffusion_train_step
from sampling import ddim_sample, mv_sde_sample


def one_hot(y: np.ndarray, num_classes: int) -> jnp.ndarray:
    return jax.nn.one_hot(jnp.array(y), num_classes)


def main():
    ap = argparse.ArgumentParser()
    # data/io
    ap.add_argument(
        "--data_pt",
        type=str,
        default="../../datasets/unified_topological_data_v6_semifast.pt",
    )
    ap.add_argument("--outdir", type=str, default="runs/mv_sde")
    ap.add_argument("--save_tag", type=str, default="")
    # train
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_energy", type=float, default=0.0)  # not used here
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    ap.add_argument("--v_pred", action="store_true")  # ignored; kept for compat
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--cfg_drop", type=float, default=0.10)
    # sampling
    ap.add_argument("--sampler", type=str, default="mv_sde", choices=["ddim", "mv_sde"])
    ap.add_argument("--sample_steps", type=int, default=400)
    ap.add_argument("--sample_count", type=int, default=128)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument(
        "--prob_flow_ode", action="store_true"
    )  # for mv_sde (deterministic)
    ap.add_argument(
        "--mf_mode", type=str, default="none", choices=["none", "voxel", "rbf"]
    )
    ap.add_argument("--mf_lambda", type=float, default=0.0)
    ap.add_argument("--mf_bandwidth", type=float, default=0.5)
    ap.add_argument("--mf_kernel", type=int, default=3)  # kept for compat (unused now)
    # modules encoder
    ap.add_argument("--S_max", type=int, default=16)
    args = ap.parse_args()

    # outdir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sub = stamp if args.save_tag == "" else f"{stamp}_{args.save_tag}"
    outdir = Path(args.outdir) / sub
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "ckpts").mkdir(parents=True, exist_ok=True)

    # seeding
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # data
    pack = load_packed_pt(args.data_pt, require_modules=True)
    describe(pack)
    train_ds, val_ds, norm_meta = train_val_split(
        pack, val_frac=0.1, seed=42, normalize=True
    )

    H, W, K, C = train_ds.vol.shape[1:]
    labels_full = pack.get("labels", None)
    num_classes = int(labels_full.max() + 1) if labels_full is not None else 1

    # embedder
    rng, k_unet, k_enc = jax.random.split(rng, 3)
    embed_fn = build_modules_embedder(k_enc, out_dim=256, T_max=1, S_max=args.S_max)
    cond_dim = 256 + num_classes

    # UNet init
    unet = UNet3D_FiLM(ch=64)
    x_d = jnp.zeros((args.batch, H, W, K, C), jnp.float32)
    t_d = jnp.zeros((args.batch,), jnp.float32)
    temb_d = time_embed(t_d, 128)
    c_d = jnp.zeros((args.batch, cond_dim), jnp.float32)
    unet_params = unet.init(k_unet, x_d, temb_d, c_d)["params"]

    diff_state = create_diffusion_state(
        rng,
        unet.apply,
        unet_params,
        T=args.T,
        lr=args.lr,
        schedule=args.schedule,
        ema_decay=args.ema_decay,
    )

    # iterator
    train_it = iterate_batches(
        train_ds, batch_size=args.batch, shuffle=True, seed=args.seed, epochs=None
    )

    print("[train] starting…")
    for step in range(1, args.steps + 1):
        batch = next(train_it)
        vol = jnp.array(batch["vol"])  # normalized
        m_emb = embed_fn(batch["modules"])  # type: ignore (B,256)
        y_oh = one_hot(batch["labels"], num_classes)  # (B,C)
        cond = jnp.concatenate([m_emb, y_oh], axis=-1).astype(jnp.float32)

        rng, key_drop, key_diff = jax.random.split(rng, 3)
        drop_mask = jax.random.bernoulli(
            key_drop, p=args.cfg_drop, shape=(vol.shape[0], 1)
        )
        cond_train = cond * (1.0 - drop_mask)

        diff_state, dloss = diffusion_train_step(diff_state, vol, cond_train, key_diff)

        if step % 20 == 0 or step == 1 or step == args.steps:
            print(f"step {step:05d} | diff_loss={float(dloss):.5f}")

    # -------- sampling --------
    print("[sample] preparing validation cond…")
    val_it = iterate_batches(
        val_ds, batch_size=args.batch, shuffle=False, seed=123, epochs=1
    )

    rng_sample = jax.random.PRNGKey(args.seed ^ 0xC0FFEE)
    samples_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    total = 0
    while total < args.sample_count:
        try:
            val_batch = next(val_it)
        except StopIteration:
            # restart one pass (deterministic order)
            val_it = iterate_batches(
                val_ds, batch_size=args.batch, shuffle=False, seed=123, epochs=1
            )
            val_batch = next(val_it)

        m_emb_ref = embed_fn(val_batch["modules"])  # type: ignore
        y_ref = np.array(val_batch["labels"])
        y_oh_ref = one_hot(y_ref, num_classes)
        cond_ref = jnp.concatenate([m_emb_ref, y_oh_ref], axis=-1).astype(jnp.float32)
        cond_uncond = jnp.zeros_like(cond_ref) if args.cfg_scale != 0.0 else None

        shape = (cond_ref.shape[0], H, W, K, C)

        if args.sampler == "ddim":
            xgen = ddim_sample(
                unet_apply=diff_state.train.apply_fn,
                params=diff_state.ema_params,
                shape=shape,
                betas=diff_state.betas,
                alphas=diff_state.alphas,
                alpha_bars=diff_state.alpha_bars,
                cond_vec=cond_ref,
                steps=args.sample_steps,
                rng=rng_sample,
                cfg_scale=float(args.cfg_scale),
                cond_uncond_vec=cond_uncond,
                return_all=False,
            )
        else:
            xgen = mv_sde_sample(
                unet_apply=diff_state.train.apply_fn,
                params=diff_state.ema_params,
                shape=shape,
                betas=diff_state.betas,
                alphas=diff_state.alphas,
                alpha_bars=diff_state.alpha_bars,
                cond_vec=cond_ref,
                steps=args.sample_steps,
                rng=rng_sample,
                cfg_scale=float(args.cfg_scale),
                cond_uncond_vec=cond_uncond,
                mf_mode=args.mf_mode,
                mf_lambda=float(args.mf_lambda),
                mf_bandwidth=float(args.mf_bandwidth),
                prob_flow_ode=True,
                return_all=False,
            )

        x_np = np.array(xgen)  # normalized [-~1,1]
        # invert normalization before saving so eval sees original scale
        lo = np.asarray(norm_meta.get("lo"), np.float32)
        hi = np.asarray(norm_meta.get("hi"), np.float32)
        x_np = invert_norm(x_np, lo, hi)

        samples_all.append(x_np)
        labels_all.append(y_ref)
        total += x_np.shape[0]
        if total >= args.sample_count:
            break

    samples_np = np.concatenate(samples_all, 0)[: args.sample_count]
    labels_np = np.concatenate(labels_all, 0)[: args.sample_count]

    meta = {
        "seed": int(args.seed),
        "sampler": args.sampler,
        "sample_steps": int(args.sample_steps),
        "sample_count": int(args.sample_count),
        "cfg_scale": float(args.cfg_scale),
        "mf": {
            "mode": args.mf_mode,
            "lambda": float(args.mf_lambda),
            "bandwidth": float(args.mf_bandwidth),
        },
        "train": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "T": int(args.T),
            "schedule": args.schedule,
            "ema_decay": float(args.ema_decay),
            "cfg_drop": float(args.cfg_drop),
        },
        "data": {
            "H": int(H),
            "W": int(W),
            "K": int(K),
            "C": int(C),
            "num_classes": int(num_classes),
            "data_pt": str(args.data_pt),
        },
        "norm": norm_meta,  # for diagnostics
    }

    out_npz = outdir / "samples_landscapes.npz"
    np.savez_compressed(
        out_npz,
        samples=samples_np,  # (N,H,W,K,C) in original scale
        labels=np.array(labels_np, np.int64),
        meta=json.dumps(meta),
    )
    with open(outdir / "sampling_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] saved: {out_npz}")
    print(f"[done] labels shape={labels_np.shape}  samples shape={samples_np.shape}")


if __name__ == "__main__":
    main()
