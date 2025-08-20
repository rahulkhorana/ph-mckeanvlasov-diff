# main.py
import argparse
import os
import json
from pathlib import Path
from typing import Tuple, List

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
    time_embed as time_embed_fn,  # time_embed exists in models; we only use dims at init
    EnergyNetwork,
)
from losses_steps import (
    DiffusionState,
    EnergyState,
    EncoderState,
    create_diffusion_state,
    create_energy_state,
    create_encoder_state,
    diffusion_train_step,
    apply_diffusion_updates,
    energy_step_E,
    apply_energy_updates_E,
    energy_step_encoder,
    apply_encoder_updates,
)
from sampling import mv_sde_sample, make_energy_guidance


# -------------------------- utils --------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    return jax_key


def one_hot(labels: np.ndarray, num_classes: int) -> jnp.ndarray:
    return jax.nn.one_hot(jnp.array(labels, dtype=jnp.int32), num_classes=num_classes)


def robust_fit_stats(x: np.ndarray) -> Tuple[float, float]:
    # ignore NaNs/Infs, keep std >= eps
    x = np.asarray(x, np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = float(x.mean())
    sigma = float(max(x.std(), 1e-6))
    return mu, sigma


def standardize_apply(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    z = (x - mu) / sigma
    # clamp to prevent blow-ups early
    z = jnp.clip(z, -10.0, 10.0)
    # replace weird values defensively
    z = jnp.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z


def standardize_invert(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x * sigma + mu


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


# Build a *learnable* modules trajectory encoder wrapper with the signature
# mods_embed_fn(enc_params, mods_batch) -> (B, out_dim)
def build_mods_embedder(enc: ModulesTrajectoryEncoder):
    def mods_embed_fn(params, mods_batch: List[List]):
        feats_list, set_list, time_list = [], [], []
        for mods in mods_batch:
            f, s, t = featurize_modules_trajectory([mods], T_max=1, S_max=16)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        Fb = jnp.stack(feats_list, 0)  # (B,1,S,F)
        Sb = jnp.stack(set_list, 0)  # (B,1,S,1)
        Tb = jnp.stack(time_list, 0)  # (B,1,1)
        # robustify
        Fb = jnp.nan_to_num(Fb, nan=0.0, posinf=0.0, neginf=0.0)
        Sb = jnp.nan_to_num(Sb, nan=0.0, posinf=0.0, neginf=0.0)
        Tb = jnp.nan_to_num(Tb, nan=0.0, posinf=0.0, neginf=0.0)
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
    return y_table[jnp.array(y_indices, dtype=jnp.int32)]  # (B, y_dim)


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


# Debug guard to detect state corruption immediately
def _assert_energy_state(tag: str, s):
    if not isinstance(s, EnergyState):
        raise TypeError(f"[{tag}] energy_state corrupted: got {type(s)}")


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
    ap.add_argument("--lr_energy", type=float, default=3e-4)  # cooler by default
    ap.add_argument("--lr_enc", type=float, default=1e-3)

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    ap.add_argument("--v_pred", action="store_true", default=True)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    # CFG (kept; you can ignore at sampling)
    ap.add_argument("--cfg_drop", type=float, default=0.05)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument(
        "--cfg_sched", type=str, default="cosine", choices=["linear", "cosine", "exp"]
    )
    ap.add_argument("--cfg_strength", type=float, default=5.0)

    # Energy
    ap.add_argument("--use_energy", action="store_true", default=True)
    ap.add_argument("--energy_scale", type=float, default=0.5)
    ap.add_argument("--energy_tau", type=float, default=0.10)  # a bit warmer
    ap.add_argument("--energy_gp", type=float, default=1e-4)

    # MV term (sampler)
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
    ap.add_argument("--sample_label", type=int, default=-1)
    args = ap.parse_args()

    # ---------------- data ----------------
    pack = load_packed_pt(args.data_pt, require_modules=True)
    print(describe(pack))
    train_ds, val_ds = train_val_split(pack, val_frac=0.1, seed=42)

    # features
    H, W, K, C = train_ds.vol.shape[1:]  # (N,H,W,K,C)
    print(f"N={len(train_ds)}  vol=(N,H,W,KS,C)=({len(train_ds)}, {H}, {W}, {K}, {C})")

    # labels / classes
    num_classes = int(pack["labels"].max() + 1)

    # normalization over train (robust)
    mu, sigma = robust_fit_stats(np.asarray(train_ds.vol))

    # iterator
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
    F = 13 + 16  # featurizer dim for S_max=16 with pos ids
    feats_d = jnp.zeros((1, 1, 16, F), jnp.float32)
    set_d = jnp.zeros((1, 1, 16, 1), jnp.float32)
    tim_d = jnp.ones((1, 1, 1), jnp.float32)
    enc_params = enc.init(k_enc, feats_d, set_d, tim_d)["params"]
    enc_state = create_encoder_state(enc.apply, enc_params, lr=args.lr_enc)
    mods_embed_fn = build_mods_embedder(enc)  # uses enc.apply internally

    # UNet
    unet = UNet3D_FiLM(ch=48)  # a bit slimmer to help memory
    x_d = jnp.zeros((args.batch, H, W, K, C), jnp.float32)
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

    # Energy net (ALWAYS create; gate use by args.use_energy)
    E = EnergyNetwork(ch=48, cond_dim=y_dim + 256)
    E_params = E.init(k_energy, x_d, cond0)["params"]
    e_state = create_energy_state(
        E.apply,
        E_params,
        lr=args.lr_energy,
        tau=args.energy_tau,
        gp_lambda=args.energy_gp,
    )
    _assert_energy_state("init", e_state)

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
        )  # {"vol": (B,H,W,K,C), "labels": (B,), "modules": list}
        vol = jnp.array(batch["vol"], dtype=jnp.float32)
        labels = np.asarray(batch["labels"])
        # robust standardization
        vol = standardize_apply(vol, mu, sigma)

        # cond vectors
        y_emb = y_embed_from_table(y_table, labels)  # (B,y_dim)
        m_emb = mods_embed_fn(enc_state.params, batch["modules"])  # (B,256)
        m_emb = jnp.nan_to_num(m_emb, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
        cond_vec = jnp.concatenate([y_emb, m_emb], axis=-1)  # (B, D)

        # classifier-free dropout (same shape, per-example)
        rng, k_drop, k_step = jax.random.split(rng, 3)
        drop_mask = jax.random.bernoulli(
            k_drop, p=args.cfg_drop, shape=(vol.shape[0], 1)
        )
        cond_vec_train = jnp.where(drop_mask, jnp.zeros_like(cond_vec), cond_vec)

        # diffusion step
        diff_state, dloss, dgrads = diffusion_train_step(
            diff_state,
            vol,
            k_step,
            bool(diff_state.v_prediction),
            cond_vec_train,
        )
        diff_state = apply_diffusion_updates(diff_state, dgrads)

        # energy steps (gate by flag, but e_state object always exists)
        e_state, eloss, egrads = energy_step_E(
            e_state, vol, cond_vec, e_state.apply_fn, chunk=4, gp_subset=4
        )

        e_state = apply_energy_updates_E(e_state, egrads)

        _assert_energy_state("post-E", e_state)

        # --- encoder update: build the same batch features explicitly ---
        feats_list, set_list, time_list = [], [], []
        for mods in batch["modules"]:
            f, s, t = featurize_modules_trajectory([mods], T_max=1, S_max=16)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        Fb = jnp.stack(feats_list, 0)
        Sb = jnp.stack(set_list, 0)
        Tb = jnp.stack(time_list, 0)
        Fb = jnp.nan_to_num(Fb, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
        Sb = jnp.nan_to_num(Sb, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
        Tb = jnp.nan_to_num(Tb, nan=0.0, posinf=10000, neginf=0.0)  # type: ignore

        enc_state, eloss_Enc, Enc_grads = energy_step_encoder(
            enc_state=enc_state,
            y_emb=y_emb,
            feats_b=Fb,
            set_b=Sb,
            time_b=Tb,
            E_apply=e_state.apply_fn,
            eparams=e_state.params,
            L=vol,
            tau=float(e_state.tau),
            chunk=4,
        )

        enc_state = apply_encoder_updates(enc_state, Enc_grads)

        eloss_E = float(eloss) + float(eloss_Enc)

        if step % 20 == 0:
            print(
                f"step {step:05d} | ddpm_loss={float(dloss):.4f} | energy_loss={float(eloss_E):.4f}"
            )

        # -------------- checkpoint + sample --------------
        do_ckpt = (step % args.ckpt_every == 0) or (step == args.steps)
        do_sample = (step % args.sample_every == 0) or (step == args.steps)

        if do_ckpt:
            save_ckpt(diff_state, ckpt_dir / f"diffusion_step{step:06d}.ckpt")
            save_ckpt(enc_state, ckpt_dir / f"encoder_step{step:06d}.ckpt")
            save_ckpt(e_state, ckpt_dir / f"energy_step{step:06d}.ckpt")
            save_json(
                {
                    "step": step,
                    "ddpm_loss": float(dloss),
                    "energy_loss": float(eloss_E),
                    "mu": mu,
                    "sigma": sigma,
                    "args": vars(args),
                },
                ckpt_dir / f"train_log_step{step:06d}.json",
            )

        if do_sample:
            # Choose labels for sampling
            B = vol.shape[0]
            if args.sample_label >= 0:
                y_idx = np.full((B,), args.sample_label, dtype=np.int32)
            else:
                y_idx = labels  # last batch labels

            y_emb_s = y_embed_from_table(y_table, y_idx)
            m_emb_s = mods_embed_fn(enc_state.params, batch["modules"])
            m_emb_s = jnp.nan_to_num(m_emb_s, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
            cond_vec_s = jnp.concatenate([y_emb_s, m_emb_s], axis=-1)

            guidance = make_energy_guidance(
                E_apply=e_state.apply_fn,  # type: ignore
                eparams=e_state.params,
                cond_vec=cond_vec_s,
            )

            null_vec = jnp.zeros_like(cond_vec_s)

            samples = mv_sde_sample(
                unet_apply=diff_state.apply_fn,
                params=diff_state.ema_params,  # EMA for sampling
                shape=(B, H, W, K, C),
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
                guidance_schedule="cosine",
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
