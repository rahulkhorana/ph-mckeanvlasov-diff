# main.py — conditional 3D diffusion on MPL volumes with label+modules conditioning + MV-SDE sampling
import argparse
import os
import json
from pathlib import Path
from typing import Tuple, List, Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax

from dataloader import load_packed_pt, train_val_split, iterate_batches, describe
from models import (
    UNet3D_FiLM,
    ModulesTrajectoryEncoder,
    featurize_modules_trajectory,
    time_embed as time_embed_fn,
    EnergyNetwork,
)
from losses_steps import (
    create_diffusion_state,
    create_energy_state,
    create_encoder_state,
    diffusion_train_step,
    energy_step_E,
    energy_step_encoder,
)
from sampling import mv_sde_sample, make_energy_guidance


# -------------------------- utils --------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    return jax_key


def standardize_fit(x: np.ndarray) -> Tuple[float, float]:
    mu = float(x.mean())
    sigma = float(x.std() + 1e-6)
    return mu, sigma


def standardize_apply(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    return (x - mu) / sigma


def standardize_invert(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    return x * sigma + mu


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


# Build a *learnable* modules trajectory encoder wrapper with the signature
# mods_embed_fn(enc_params, mods_batch) -> (B, out_dim)
def build_mods_embedder(enc: ModulesTrajectoryEncoder, enc_params):
    def mods_embed_fn(params, mods_batch: List[List[Any]]):
        feats_list, set_list, time_list = [], [], []
        # Wrap each flat list -> trajectory of length 1
        for mods in mods_batch:
            f, s, t = featurize_modules_trajectory([mods], T_max=1, S_max=16)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        Fb = jnp.stack(feats_list, 0)
        Sb = jnp.stack(set_list, 0)
        Tb = jnp.stack(time_list, 0)
        return enc.apply({"params": params}, Fb, Sb, Tb)  # (B, out_dim)

    return mods_embed_fn


# Simple fixed (non-trainable) label projection to y_dim
def init_label_proj(rng, num_classes: int, y_dim: int) -> jnp.ndarray:
    k1, _ = jax.random.split(rng)
    table = jax.random.normal(k1, (num_classes, y_dim), dtype=jnp.float32) * (
        1.0 / np.sqrt(y_dim)
    )
    return table


def y_embed_from_table(y_table: jnp.ndarray, y_indices: np.ndarray) -> jnp.ndarray:
    # y_indices: (B,)
    return y_table[jnp.array(y_indices, dtype=jnp.int32)]  # (B, y_dim)


def infer_num_classes(pack: Dict[str, Any]) -> int:
    if "labels" in pack:
        return int(np.asarray(pack["labels"]).max() + 1)
    lm = pack.get("label_map", None)
    if isinstance(lm, dict) and len(lm) > 0:
        try:
            return int(max(lm.keys()) + 1)
        except Exception:
            return int(len(lm))
    raise ValueError("Could not infer number of classes from pack.")


# -------------------------- checkpointing --------------------------
def save_ckpt(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = serialization.to_bytes(obj)
    path.write_bytes(data)


def save_numpy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def save_json(d: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2))


# -------------------------- main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_pt",
        type=str,
        default="../../datasets/unified_topological_data_v6_semifast.pt",
    )
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_energy", type=float, default=1e-3)
    ap.add_argument("--lr_enc", type=float, default=1e-3)

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    ap.add_argument(
        "--v_pred", action="store_true", help="Use v-prediction target", default=True
    )
    ap.add_argument("--ema_decay", type=float, default=0.999)

    # CFG
    ap.add_argument(
        "--cfg_drop", type=float, default=0.1, help="cond dropout prob during training"
    )
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument(
        "--cfg_sched", type=str, default="cosine", choices=["linear", "cosine", "exp"]
    )
    ap.add_argument("--cfg_strength", type=float, default=5.0)

    # Energy
    ap.add_argument("--use_energy", action="store_true", default=True)
    ap.add_argument("--energy_scale", type=float, default=0.5)
    ap.add_argument(
        "--energy_sched",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "exp"],
    )
    ap.add_argument("--energy_tau", type=float, default=0.07)
    ap.add_argument("--energy_gp", type=float, default=1e-4)

    # MV term
    ap.add_argument(
        "--mf_mode", type=str, default="rbf", choices=["none", "voxel", "rbf"]
    )
    ap.add_argument("--mf_lambda", type=float, default=0.05)
    ap.add_argument("--mf_bandwidth", type=float, default=0.5)

    # I/O
    ap.add_argument("--outdir", type=str, default="runs/mv_sde")
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--sample_steps", type=int, default=250)
    ap.add_argument(
        "--sample_label",
        type=int,
        default=-1,
        help="-1 uses batch labels; else forces this class id",
    )
    ap.add_argument("--energy_neg_k", type=int, default=8)
    args = ap.parse_args()

    # ---------------- data ----------------
    pack = load_packed_pt(args.data_pt, require_modules=True)
    print(describe(pack))
    train_ds, val_ds = train_val_split(pack, val_frac=0.1, seed=42)

    # features
    H, W, KS, C = train_ds.vol.shape[1:]  # (N,H,W,KS,3)
    print(f"N={len(train_ds)}  vol=(N,H,W,KS,C)=({len(train_ds)}, {H}, {W}, {KS}, {C})")

    # labels / classes
    num_classes = infer_num_classes(pack)

    # normalization (use the full pack to stabilize stats)
    mu, sigma = standardize_fit(np.asarray(pack["vol"]))  # (N,H,W,KS,3)

    # iterator (host numpy → we jnp-ify per step)
    train_iter = iterate_batches(
        train_ds, batch_size=args.batch, shuffle=True, seed=args.seed, epochs=None
    )

    # ---------------- models & states ----------------
    rng = seed_all(args.seed)
    rng, k_unet, k_energy, k_enc, k_y = jax.random.split(rng, 5)

    # label projection (fixed)
    y_dim = 128
    y_table = init_label_proj(k_y, num_classes=num_classes, y_dim=y_dim)  # (C,y_dim)

    # modules encoder
    enc = ModulesTrajectoryEncoder(out_dim=256)
    # init enc params with dummy shapes
    F = 13 + 16
    feats_d = jnp.zeros((1, 1, 16, F), jnp.float32)
    set_d = jnp.zeros((1, 1, 16, 1), jnp.float32)
    tim_d = jnp.ones((1, 1, 1), jnp.float32)
    enc_params = enc.init(k_enc, feats_d, set_d, tim_d)["params"]
    enc_state = create_encoder_state(enc.apply, enc_params, lr=args.lr_enc)
    mods_embed_fn = build_mods_embedder(enc, enc_params)

    # UNet
    unet = UNet3D_FiLM(ch=64)
    x_d = jnp.zeros((args.batch, H, W, KS, C), jnp.float32)
    # time embedding is computed inside losses or passed (we pass dummy for init)
    t0 = jnp.zeros((args.batch,), jnp.float32)
    t_emb0 = time_embed_fn(t0, 128)
    cond0 = jnp.zeros((args.batch, y_dim + 256), jnp.float32)
    unet_params = unet.init(k_unet, x_d, t_emb0, cond0)["params"]

    diff_state = create_diffusion_state(
        rng=rng,
        apply_fn=unet.apply,
        init_params=unet_params,
        T=args.T,
        lr=args.lr,
        v_prediction=args.v_pred,
        schedule=args.schedule,
        ema_decay=args.ema_decay,
    )

    E = EnergyNetwork(ch=64, cond_dim=y_dim + 256)
    E_params = E.init(k_energy, x_d, cond0)["params"]
    e_state = create_energy_state(
        E.apply,
        E_params,
        lr=args.lr_energy,
        tau=args.energy_tau,
        gp_lambda=args.energy_gp,
    )

    # ---------------- output dirs ----------------
    outdir = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    gen_dir = outdir / "generated"
    ensure_dir(ckpt_dir)
    ensure_dir(gen_dir)

    # ---------------- training ----------------
    for step in range(1, args.steps + 1):
        batch = next(
            train_iter
        )  # {"vol": (B,H,W,KS,3), "labels": (B,), "modules": list}
        vol = jnp.array(batch["vol"], dtype=jnp.float32)  # (B,H,W,KS,3)
        labels_np = np.asarray(batch["labels"])
        vol = standardize_apply(vol, mu, sigma)

        # cond vectors (computed OUTSIDE jit)
        y_emb = y_embed_from_table(y_table, labels_np)  # (B,y_dim)
        m_emb = mods_embed_fn(enc_state.params, batch["modules"])  # (B,256)
        cond_vec = jnp.concatenate(
            [y_emb, m_emb], axis=-1  # type:ignore (B, y_dim+256)
        )

        # classifier-free dropout on cond
        rng, k_drop = jax.random.split(rng)
        drop_mask = jax.random.bernoulli(
            k_drop, p=args.cfg_drop, shape=(vol.shape[0], 1)
        )
        cond_vec_train = jnp.where(drop_mask, jnp.zeros_like(cond_vec), cond_vec)

        # diffusion step (temb dummy; loss computes correct time internally or ignore temb)
        t_dummy = jnp.zeros((vol.shape[0],), jnp.float32)
        temb = time_embed_fn(t_dummy, 128)

        diff_state, dloss = diffusion_train_step(
            diff_state,
            vol,
            rng=jax.random.PRNGKey(step),
            v_prediction=bool(diff_state.v_prediction),
            temb=temb,
            cond_vec=cond_vec_train,
        )

        # energy step (if enabled)

        B = vol.shape[0]
        K = max(1, min(args.energy_neg_k, max(1, B - 1)))

        # build (B,K) negatives on host (no self-index)
        neg_idx = np.empty((B, K), dtype=np.int32)
        all_idx = np.arange(B, dtype=np.int32)
        for i in range(B):
            pool = np.concatenate([all_idx[:i], all_idx[i + 1 :]])  # exclude i
            replace = pool.shape[0] < K
            neg_idx[i] = np.random.choice(pool, size=K, replace=replace)
        neg_idx = jnp.array(neg_idx)  # send to device

        e_state, eloss_E = energy_step_E(e_state, vol, cond_vec, neg_idx)
        enc_state, eloss_Enc = energy_step_encoder(
            enc_state,
            e_state.apply_fn,
            e_state.params,
            mods_embed_fn,
            batch["modules"],
            y_emb,
            vol,
            neg_idx,
            tau=args.energy_tau,
        )
        eloss_value = float(eloss_E + eloss_Enc)

        if step % 20 == 0:
            print(
                f"step {step:05d} | ddpm_loss={float(dloss):.4f} | energy_loss={eloss_value:.4f}"
            )

        # -------------- checkpoint + sample --------------
        do_ckpt = (step % args.ckpt_every == 0) or (step == args.steps)
        do_sample = (step % args.sample_every == 0) or (step == args.steps)

        if do_ckpt:
            save_ckpt(diff_state, ckpt_dir / f"diffusion_step{step:06d}.ckpt")
            save_ckpt(enc_state, ckpt_dir / f"encoder_step{step:06d}.ckpt")
            if e_state is not None:
                save_ckpt(e_state, ckpt_dir / f"energy_step{step:06d}.ckpt")
            save_json(
                {
                    "step": step,
                    "ddpm_loss": float(dloss),
                    "energy_loss": eloss_value,
                    "mu": mu,
                    "sigma": sigma,
                    "args": vars(args),
                },
                ckpt_dir / f"train_log_step{step:06d}.json",
            )

        if do_sample:
            # Choose labels for sampling
            B = int(vol.shape[0])
            if args.sample_label >= 0:
                y_idx = np.full((B,), args.sample_label, dtype=np.int32)
            else:
                y_idx = labels_np  # last batch labels

            y_emb_s = y_embed_from_table(y_table, y_idx)
            # modules embedding: use encoder on the same batch's modules
            m_emb_s = mods_embed_fn(enc_state.params, batch["modules"])
            cond_vec_s = jnp.concatenate([y_emb_s, m_emb_s], axis=-1)  # type: ignore

            # energy guidance (fixed cond_vec in wrapper)
            guidance = None
            if args.use_energy:
                guidance = make_energy_guidance(
                    E_apply=e_state.apply_fn,  # type: ignore
                    eparams=e_state.params,  # type: ignore
                    cond_vec=cond_vec_s,
                )

            # CFG null vector (zeros)
            null_vec = jnp.zeros_like(cond_vec_s)

            samples = mv_sde_sample(
                unet_apply=diff_state.apply_fn,
                params=diff_state.ema_params,  # use EMA for sampling
                shape=(B, H, W, KS, C),
                betas=diff_state.betas,
                alphas=diff_state.alphas,
                alpha_bars=diff_state.alpha_bars,
                cond_vec=cond_vec_s,
                steps=args.sample_steps,
                rng=jax.random.PRNGKey(1234 + step),
                v_prediction=bool(diff_state.v_prediction),
                cfg_scale=args.cfg_scale,
                null_cond_vec=null_vec,
                cfg_schedule=args.cfg_sched,
                cfg_strength=args.cfg_strength,
                guidance_fn=guidance,
                guidance_scale=args.energy_scale if args.use_energy else 0.0,
                guidance_schedule=args.energy_sched,
                guidance_strength=3.0,
                mf_mode=args.mf_mode,
                mf_lambda=args.mf_lambda,
                mf_bandwidth=args.mf_bandwidth,
                prob_flow_ode=False,
                return_all=False,
            )
            samples_np = np.array(standardize_invert(samples, mu, sigma))
            save_numpy(samples_np, gen_dir / f"samples_step{step:06d}.npy")

    print("Training finished.")


if __name__ == "__main__":
    main()
