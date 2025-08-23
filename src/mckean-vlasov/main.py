# main.py — module-bridge conditioned 3D diffusion on MPL volumes (P0–P4)
import argparse
from pathlib import Path
import json
import time

import numpy as np
import jax, jax.numpy as jnp

from dataloader import load_packed_pt, train_val_split, iterate_batches, describe
from models import UNet3D_FiLM, EnergyNetwork, build_modules_embedder, time_embed
from losses_steps import (
    create_diffusion_state,
    diffusion_train_step,
    create_energy_state,
    energy_train_step,
)
from sampling import (
    ddim_sample,
    mv_sde_sample,
    make_module_bridge,
)


def one_hot(y: np.ndarray, num_classes: int) -> jnp.ndarray:
    return jax.nn.one_hot(jnp.array(y), num_classes)


def main():
    ap = argparse.ArgumentParser()
    # ---- data & io ----
    ap.add_argument(
        "--data_pt",
        type=str,
        default="../../datasets/unified_topological_data_v6_semifast.pt",
    )
    ap.add_argument("--outdir", type=str, default="runs/mv_sde")
    ap.add_argument(
        "--ckpt_every", type=int, default=0
    )  # (not used but kept for compat)
    ap.add_argument(
        "--sample_every", type=int, default=0
    )  # (not used; sampling done at end)

    # ---- train ----
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_energy", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    ap.add_argument("--v_pred", action="store_true", help="v-prediction head")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument(
        "--cfg_drop", type=float, default=0.10, help="cond dropout prob (train)"
    )

    # ---- sampling ----
    ap.add_argument("--sampler", type=str, default="mv_sde", choices=["ddim", "mv_sde"])
    ap.add_argument("--sample_steps", type=int, default=500)
    ap.add_argument(
        "--cfg_scale", type=float, default=0.0, help="CFG scale at sample time"
    )
    ap.add_argument(
        "--bridge_scale", type=float, default=1.0, help="energy bridge weight on x0_hat"
    )
    ap.add_argument(
        "--use_energy",
        type=bool,
        default=True,
        help="train energy net & enable bridge",
    )
    ap.add_argument(
        "--prob_flow_ode",
        action="store_true",
        help="deterministic branch for MV-SDE; recommended",
    )
    # mean-field (MV-SDE) knobs
    ap.add_argument(
        "--mf_mode", type=str, default="rbf", choices=["none", "rbf", "voxel"]
    )
    ap.add_argument("--mf_lambda", type=float, default=0.05)
    ap.add_argument("--mf_bandwidth", type=float, default=0.5)
    ap.add_argument("--mf_kernel", type=int, default=3)

    # ---- modules encoder ----
    ap.add_argument("--S_max", type=int, default=16, help="max elements per module set")
    args = ap.parse_args()

    # ---- outdir & config dump ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / stamp
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "ckpts").mkdir(parents=True, exist_ok=True)

    # ---- seeding (P1) ----
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # ---- load data ----
    pack = load_packed_pt(args.data_pt, require_modules=True)
    describe(pack)
    train_ds, val_ds = train_val_split(pack, val_frac=0.1, seed=42)  # stable split (P2)

    H, W, K, C = train_ds.vol.shape[1:]
    if "labels" in pack:
        num_classes = int(pack["labels"].max() + 1)
    else:
        num_classes = int(np.max(train_ds.labels) + 1)

    # ---- build modules embedder (frozen) ----
    rng, k_unet, k_energy, k_enc = jax.random.split(rng, 4)
    embed_fn = build_modules_embedder(k_enc, out_dim=256, T_max=1, S_max=args.S_max)

    # fused cond is (modules 256 + label one-hot)
    cond_dim = 256 + num_classes

    # ---- init UNet ----
    unet = UNet3D_FiLM(ch=64)
    x_d = jnp.zeros((args.batch, H, W, K, C), jnp.float32)
    t_d = jnp.zeros((args.batch,), jnp.float32)
    temb_d = time_embed(t_d, 128)
    c_d = jnp.zeros((args.batch, cond_dim), jnp.float32)  # we pass fused cond here
    unet_params = unet.init(k_unet, x_d, temb_d, c_d)["params"]

    diff_state = create_diffusion_state(
        rng,
        unet.apply,
        unet_params,
        T=args.T,
        lr=args.lr,
        v_prediction=args.v_pred,
        schedule=args.schedule,
        ema_decay=args.ema_decay,
    )

    # ---- energy net (optional) ----
    if args.use_energy:
        E = EnergyNetwork(ch=64, cond_dim=cond_dim)
        E_params = E.init(k_energy, x_d, c_d)["params"]
        e_state = create_energy_state(E.apply, E_params, lr=args.lr_energy)
    else:
        E, e_state = None, None

    # ---- data iterator ----
    train_it = iterate_batches(
        train_ds, batch_size=args.batch, shuffle=True, seed=args.seed, epochs=None
    )

    # ---- training loop ----
    print("[train] starting…")
    for step in range(1, args.steps + 1):
        batch = next(train_it)
        vol = jnp.array(batch["vol"])  # (B,H,W,K,C)
        m_emb = embed_fn(batch["modules"])  # (B,256)
        y_oh = one_hot(batch["labels"], num_classes)  # (B,num_classes)
        cond = jnp.concatenate([m_emb, y_oh], axis=-1).astype(
            jnp.float32
        )  # (B,cond_dim)

        # classifier-free dropout on cond for diffusion (P3-compat)
        rng, key_drop, key_diff = jax.random.split(rng, 3)
        drop_mask = jax.random.bernoulli(
            key_drop, p=args.cfg_drop, shape=(vol.shape[0], 1)
        )
        cond_train = cond * (1.0 - drop_mask)

        # diffusion step
        diff_state, dloss = diffusion_train_step(
            diff_state,
            vol,
            cond_train,
            key_diff,
            v_prediction=bool(diff_state.v_prediction),
        )

        # energy step (optional, no dropout)
        if args.use_energy:
            e_state, eloss = energy_train_step(e_state, vol, cond)
        else:
            eloss = jnp.array(0.0, dtype=jnp.float32)

        if step % 20 == 0 or step == 1 or step == args.steps:
            print(
                f"step {step:05d} | diff_loss={float(dloss):.5f}"
                + (f" | energy_loss={float(eloss):.5f}" if args.use_energy else "")
            )

    # ---- sampling on a fixed validation batch (P2/P3) ----
    print("[sample] preparing validation cond…")
    val_it = iterate_batches(
        val_ds, batch_size=args.batch, shuffle=False, seed=123, epochs=1
    )
    val_batch = next(val_it)

    # reference labels and cond for sampling
    vol_ref = jnp.array(val_batch["vol"])  # shape only
    m_emb_ref = embed_fn(val_batch["modules"])  # (B,256)
    y_ref = np.array(val_batch["labels"])  # (B,)
    y_oh_ref = one_hot(y_ref, num_classes)  # (B,num_classes)
    cond_ref = jnp.concatenate([m_emb_ref, y_oh_ref], axis=-1).astype(
        jnp.float32
    )  # (B,cond_dim)

    # unconditional cond for CFG (zeros, matches training dropout) (P3)
    cond_uncond = jnp.zeros_like(cond_ref)

    # bridge (optional): energy on x0_hat
    if args.use_energy:
        bridge = make_module_bridge(
            E_apply=e_state.train.apply_fn,  # type: ignore
            eparams=e_state.train.params,  # type: ignore
            cond_vec=cond_ref,
            T=diff_state.T,
        )
    else:
        bridge = None

    rng_sample = jax.random.PRNGKey(args.seed ^ 0xC0FFEE)

    print(
        f"[sample] sampler={args.sampler}  steps={args.sample_steps} "
        f"cfg_scale={args.cfg_scale}  prob_flow_ode={args.prob_flow_ode}  "
        f"mf=({args.mf_mode},{args.mf_lambda},{args.mf_bandwidth},{args.mf_kernel})  "
        f"bridge_scale={args.bridge_scale if args.use_energy else 0.0}"
    )

    if args.sampler == "ddim":
        samples = ddim_sample(
            unet_apply=diff_state.train.apply_fn,
            params=diff_state.ema_params,  # EMA weights for sampling
            shape=(cond_ref.shape[0], H, W, K, C),
            betas=diff_state.betas,
            alphas=diff_state.alphas,
            alpha_bars=diff_state.alpha_bars,
            cond_vec=cond_ref,  # <- pass fused cond (P3)
            steps=args.sample_steps,
            rng=rng_sample,
            v_prediction=bool(diff_state.v_prediction),
            bridge_fn=bridge,
            bridge_scale=args.bridge_scale if args.use_energy else 0.0,
            cfg_scale=args.cfg_scale,
            cond_uncond_vec=cond_uncond if args.cfg_scale != 0.0 else None,
            return_all=False,
        )
    else:
        samples = mv_sde_sample(
            unet_apply=diff_state.train.apply_fn,
            params=diff_state.ema_params,
            shape=(cond_ref.shape[0], H, W, K, C),
            betas=diff_state.betas,
            alphas=diff_state.alphas,
            alpha_bars=diff_state.alpha_bars,
            cond_vec=cond_ref,  # <- pass fused cond (P3)
            steps=args.sample_steps,
            rng=rng_sample,
            v_prediction=bool(diff_state.v_prediction),
            mf_mode=args.mf_mode,
            mf_lambda=float(args.mf_lambda),
            mf_bandwidth=float(args.mf_bandwidth),
            mf_kernel_size=int(args.mf_kernel),
            bridge_fn=bridge,
            bridge_scale=args.bridge_scale if args.use_energy else 0.0,
            cfg_scale=args.cfg_scale,
            cond_uncond_vec=cond_uncond if args.cfg_scale != 0.0 else None,
            prob_flow_ode=bool(args.prob_flow_ode),  # default deterministic (P3)
            return_all=False,
        )

    # ---- save (P0): samples + labels + cond + meta ----
    samples_np = np.array(samples)  # (B,H,W,K,C)
    out_npz = outdir / "samples_landscapes.npz"
    meta = {
        "seed": int(args.seed),
        "sampler": args.sampler,
        "sample_steps": int(args.sample_steps),
        "cfg_scale": float(args.cfg_scale),
        "use_energy": bool(args.use_energy),
        "bridge_scale": float(args.bridge_scale if args.use_energy else 0.0),
        "prob_flow_ode": bool(args.prob_flow_ode),
        "mf": {
            "mode": args.mf_mode,
            "lambda": float(args.mf_lambda),
            "bandwidth": float(args.mf_bandwidth),
            "kernel": int(args.mf_kernel),
        },
        "train": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "lr_energy": float(args.lr_energy),
            "T": int(args.T),
            "schedule": args.schedule,
            "v_pred": bool(args.v_pred),
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
    }
    np.savez_compressed(
        out_npz,
        samples=samples_np,  # (B,H,W,K,C)
        labels=np.array(y_ref, dtype=np.int64),  # (B,)  <-- for sane eval (P0)
        cond=np.array(np.array(cond_ref), dtype=np.float32),  # (B,256+classes)
        meta=json.dumps(meta),
    )
    with open(outdir / "sampling_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] saved: {out_npz}")
    print(
        f"[done] labels shape={np.array(y_ref).shape}  cond shape={np.array(cond_ref).shape}"
    )


if __name__ == "__main__":
    main()
