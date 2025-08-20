# main.py
import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

from dataloader import load_packed_pt, train_val_split, iterate_batches, describe
from models import (
    UNet3D_FiLM,
    ModulesTrajectoryEncoder,
    featurize_modules_trajectory,
    time_embed as time_embed_fn,  # only for init dims
    EnergyNetwork,
)

# diffusion bits
from losses_steps import (
    DiffusionState,
    create_diffusion_state,
    diffusion_train_step,
)

# energy MoCo queue (device-resident, small)
from energy_losses_steps import (
    EnergyState,
    create_energy_state,
    energy_step_E_bank,  # (state, L, cond_vec, rng, *, chunk) -> (new_state, loss, metrics)
)

# encoder trained against frozen energy + queue
from encoder_losses_steps import (
    EncoderState,
    create_encoder_state,
    energy_step_encoder,  # uses e_state.queue / e_state.queue_count
)

from sampling import mv_sde_sample, make_energy_guidance


# -------------------------- utils --------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def robust_fit_stats(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = float(x.mean())
    sigma = float(max(x.std(), 1e-6))
    return mu, sigma


def standardize_apply(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    z = (x - mu) / sigma
    z = jnp.clip(z, -10.0, 10.0)
    return jnp.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_invert(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x * sigma + mu


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


# Build a learnable modules-trajectory encoder wrapper:
# mods_embed_fn(enc_params, mods_batch) -> (B, out_dim)
def build_mods_embedder(enc: ModulesTrajectoryEncoder, S_max: int = 16):
    def mods_embed_fn(params, mods_batch: List[List]):
        feats_list, set_list, time_list = [], [], []
        for mods in mods_batch:
            f, s, t = featurize_modules_trajectory([mods], T_max=1, S_max=S_max)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        Fb = jnp.stack(feats_list, 0)  # (B,1,S,F)
        Sb = jnp.stack(set_list, 0)  # (B,1,S,1)
        Tb = jnp.stack(time_list, 0)  # (B,1,1)
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
    ap.add_argument("--lr_energy", type=float, default=3e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-3)

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    ap.add_argument("--v_pred", action="store_true", default=True)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    # CFG
    ap.add_argument("--cfg_drop", type=float, default=0.05)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument(
        "--cfg_sched", type=str, default="cosine", choices=["linear", "cosine", "exp"]
    )
    ap.add_argument("--cfg_strength", type=float, default=5.0)

    # Energy (these map to EnergyState config)
    ap.add_argument("--use_energy", action="store_true", default=True)
    ap.add_argument("--energy_scale", type=float, default=0.5)  # used at sampling
    ap.add_argument("--energy_tau", type=float, default=0.10)  # loss temperature
    ap.add_argument(
        "--energy_queue", type=int, default=4096
    )  # Q (must be multiple of chunk)
    ap.add_argument("--energy_topk", type=int, default=32)  # k_top
    ap.add_argument(
        "--energy_gumbel", type=float, default=0.2
    )  # gumbel scale (0 disables)

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
    train_ds, _ = train_val_split(pack, val_frac=0.1, seed=42)

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
    S_MAX = 16
    enc = ModulesTrajectoryEncoder(out_dim=256)
    F = 13 + S_MAX  # featurizer dim for S_max with pos ids
    feats_d = jnp.zeros((1, 1, S_MAX, F), jnp.float32)
    set_d = jnp.zeros((1, 1, S_MAX, 1), jnp.float32)
    tim_d = jnp.ones((1, 1, 1), jnp.float32)
    enc_params = enc.init(k_enc, feats_d, set_d, tim_d)["params"]
    enc_state = create_encoder_state(enc.apply, enc_params, lr=args.lr_enc)
    mods_embed_fn = build_mods_embedder(enc, S_max=S_MAX)  # uses enc.apply internally

    # UNet
    unet = UNet3D_FiLM(ch=48)  # slimmer to help memory
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

    # Energy net (device-resident tiny queue inside state)
    E = EnergyNetwork(ch=48, cond_dim=y_dim + 256)
    E_params = E.init(k_energy, x_d, cond0)["params"]

    e_state = create_energy_state(
        apply_fn=E.apply,
        init_params=E_params,
        D_cond=int(y_dim + 256),
        lr=args.lr_energy,
        tau=args.energy_tau,
        Q=int(args.energy_queue),
        k_top=int(args.energy_topk),
        gumbel=float(args.energy_gumbel),
    )

    # sanity print
    print(
        f"[energy] queue Q={e_state.queue_size}  k_top={e_state.k_top}  tau={e_state.tau}  gumbel={e_state.gumbel_scale}"
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
        )  # {"vol": (B,H,W,K,C), "labels": (B,), "modules": list}

        # inputs
        vol = jnp.array(batch["vol"], dtype=jnp.float32)
        labels = np.asarray(batch["labels"])

        # robust standardization
        vol = standardize_apply(vol, mu, sigma)

        # cond vectors
        y_emb = y_embed_from_table(y_table, labels)  # (B,y_dim)
        m_emb = mods_embed_fn(enc_state.params, batch["modules"])  # (B,256)
        m_emb = jnp.nan_to_num(m_emb, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
        cond_vec = jnp.concatenate([y_emb, m_emb], axis=-1)  # (B, D)

        # classifier-free dropout (per-example)
        rng, k_drop, k_step = jax.random.split(rng, 3)
        drop_mask = jax.random.bernoulli(
            k_drop, p=args.cfg_drop, shape=(vol.shape[0], 1)
        )
        cond_vec_train = jnp.where(drop_mask, jnp.zeros_like(cond_vec), cond_vec)

        # diffusion step
        diff_state, dloss = diffusion_train_step(
            diff_state,
            vol,
            k_step,
            bool(diff_state.v_prediction),
            cond_vec_train,
        )

        eloss_E = 0.0
        # energy step (only if enabled)
        if args.use_energy:
            rng, k_e = jax.random.split(rng)
            # chunk must divide Q; choose a small static chunk to cap VRAM
            chunk = 64 if (e_state.queue_size % 64 == 0) else 32
            e_state, eloss, e_metrics = energy_step_E_bank(
                e_state,
                vol,
                cond_vec,
                k_e,
                chunk=chunk,
            )
            eloss_E = float(eloss)

        # --- encoder update (InfoNCE-ish vs frozen E using the same queue) ---
        rng, k_encstep = jax.random.split(rng)
        # build explicit features for this batch (avoid re-featurizing twice)
        feats_list, set_list, time_list = [], [], []
        for mods in batch["modules"]:
            f, s, t = featurize_modules_trajectory([mods], T_max=1, S_max=S_MAX)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        Fb = jnp.stack(feats_list, 0)
        Sb = jnp.stack(set_list, 0)
        Tb = jnp.stack(time_list, 0)
        Fb = jnp.nan_to_num(Fb, nan=0.0, posinf=0.0, neginf=0.0)
        Sb = jnp.nan_to_num(Sb, nan=0.0, posinf=0.0, neginf=0.0)
        Tb = jnp.nan_to_num(Tb, nan=0.0, posinf=10000.0, neginf=0.0)

        # use the energy trainer's queue as negatives (frozen during encoder step)
        enc_state, eloss_Enc, enc_metrics = energy_step_encoder(
            enc_state=enc_state,
            E_apply=e_state.apply_fn,
            eparams=e_state.params,
            tau=float(e_state.tau),
            k_top=int(e_state.k_top),
            gumbel_scale=float(e_state.gumbel_scale),
            L=vol,
            y_emb=y_emb,
            feats_b=Fb,
            set_b=Sb,
            time_b=Tb,
            queue=e_state.queue,
            queue_count=e_state.queue_count,
            rng=k_encstep,
            chunk=2,
        )
        eloss_E += float(eloss_Enc)

        if step % 20 == 0:
            print(
                f"step {step:05d} | ddpm_loss={float(dloss):.4f} | energy_loss={eloss_E:.4f}"
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
            B = vol.shape[0]
            if args.sample_label >= 0:
                y_idx = np.full((B,), args.sample_label, dtype=np.int32)
            else:
                y_idx = labels

            y_emb_s = y_embed_from_table(y_table, y_idx)
            m_emb_s = mods_embed_fn(enc_state.params, batch["modules"])
            m_emb_s = jnp.nan_to_num(m_emb_s, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
            cond_vec_s = jnp.concatenate([y_emb_s, m_emb_s], axis=-1)

            guidance = make_energy_guidance(
                E_apply=e_state.apply_fn,
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
