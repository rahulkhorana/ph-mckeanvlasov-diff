# main.py
# Cond. diffusion for landscapes in JAX/Flax with Bayes-style energy guidance from modules.

import argparse
import numpy as np
import jax
import jax.numpy as jnp

# --- data ---
from dataloader import load_packed_pt, train_val_split, iterate_batches, describe

# --- models ---
from models import (
    TinyUNet,
    EnergyNetwork,
    ModulesTrajectoryEncoder,
    featurize_modules_trajectory,
)

# --- losses / training steps / sched ---
from losses_steps import (
    time_embed,
    DiffusionState,
    EnergyState,
    cosine_beta_schedule,
    diffusion_train_step,
    energy_train_step,  # expects (est, embed_fn, L, mods_batch)
    create_diffusion_state,  # requires optax installed
    create_energy_state,
)

# --- sampling ---
from sampling import ddim_sample, make_energy_guidance


def build_embed_fn(rng, out_dim=256, T_MAX=1, S_MAX=16):
    """
    Create a frozen modules-trajectory encoder and return an embed_fn(mods_batch)->(B,out_dim).
    For now T_MAX=1 because your modules are flat lists; upgrade to T>1 when you have trajectories.
    """
    enc = ModulesTrajectoryEncoder(out_dim=out_dim)

    # dummy shapes to init params
    F = 13 + S_MAX  # robust stats (13) + S_MAX pos-id features
    feats_d = jnp.zeros((1, T_MAX, S_MAX, F), jnp.float32)
    set_d = jnp.zeros((1, T_MAX, S_MAX, 1), jnp.float32)
    time_d = jnp.ones((1, T_MAX, 1), jnp.float32)
    enc_params = enc.init(rng, feats_d, set_d, time_d)["params"]

    def as_single_step_traj(mods_batch):
        # wrap flat list -> T=1 surrogate trajectory
        return [[mods] for mods in mods_batch]

    def embed_fn(mods_batch):
        traj_batch = as_single_step_traj(mods_batch)
        feats_list, set_list, time_list = [], [], []
        for traj in traj_batch:
            f, s, t = featurize_modules_trajectory(traj, T_max=T_MAX, S_max=S_MAX)
            feats_list.append(f)
            set_list.append(s)
            time_list.append(t)
        feats_b = jnp.stack(feats_list, 0)
        set_b = jnp.stack(set_list, 0)
        time_b = jnp.stack(time_list, 0)
        return enc.apply({"params": enc_params}, feats_b, set_b, time_b)  # (B,out_dim)

    return enc, enc_params, embed_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_pt",
        type=str,
        default="../../datasets/unified_topological_data_v6_semifast.pt",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_energy", type=float, default=1e-3)
    parser.add_argument("--diffusion_T", type=int, default=1000)
    parser.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    parser.add_argument("--v_pred", action="store_true", help="use v-prediction target")
    args = parser.parse_args()

    # ----- load data -----
    pack = load_packed_pt(args.data_pt, require_modules=True)
    print(describe(pack))
    train_ds, val_ds = train_val_split(pack, val_frac=0.1, seed=42)
    H, W, C = train_ds.lands.shape[1:]

    train_iter = iterate_batches(
        train_ds, batch_size=args.batch, shuffle=True, seed=args.seed, epochs=None
    )

    # ----- build models & states -----
    rng = jax.random.PRNGKey(args.seed)
    rng, k_unet, k_energy, k_enc = jax.random.split(rng, 4)

    # UNet init (epsilon- or v-predictor)
    unet = TinyUNet(ch=64)
    # dummy inputs to init params
    x_d = jnp.zeros((args.batch, H, W, C), jnp.float32)
    t_d = jnp.zeros((args.batch,), jnp.float32)
    temb = time_embed(t_d, dim=128)
    unet_params = unet.init(k_unet, x_d, temb)["params"]

    # create diffusion state
    diff_state = create_diffusion_state(
        rng=k_unet,
        model_apply=unet.apply,
        params=unet_params,
        T=args.diffusion_T,
        schedule=args.schedule,
        lr=args.lr,
        v_prediction=args.v_pred,
    )

    # Energy network (E_phi)
    E = EnergyNetwork(ch=64, m_dim=256)
    E_params = E.init(k_energy, x_d, jnp.zeros((args.batch, 256), jnp.float32))[
        "params"
    ]
    e_state = create_energy_state(E.apply, E_params, lr=args.lr_energy)

    # Modules encoder (frozen for now) -> embed_fn
    _, _, embed_fn = build_embed_fn(k_enc, out_dim=256, T_MAX=1, S_MAX=16)

    # ----- train loop -----
    for step in range(1, args.steps + 1):
        batch = next(train_iter)  # dict: lands (B,H,W,C), modules: list length B
        imgs = jnp.array(batch["lands"])
        m_emb = embed_fn(batch["modules"])

        # diffusion step
        diff_state, dloss = diffusion_train_step(
            diff_state, imgs, jax.random.PRNGKey(step)
        )

        # energy step (contrastive), encoder used via embed_fn
        e_state, eloss = energy_train_step(e_state, imgs, m_emb)

        if step % 20 == 0:
            print(
                f"step {step:05d} | ddpm_loss={float(dloss):.4f} | energy_loss={float(eloss):.4f}"
            )

    # ----- sampling with energy guidance (use last batch's modules) -----
    guidance = make_energy_guidance(E.apply, e_state.params, embed_fn, batch["modules"])  # type: ignore
    samples = ddim_sample(
        unet.apply,
        diff_state.params,
        shape=(args.batch, H, W, C),
        betas=diff_state.betas,
        alphas=diff_state.alphas,
        alpha_bars=diff_state.alpha_bars,
        guidance_fn=guidance,
        guidance_scale=1.0,
        steps=50,
        rng=jax.random.PRNGKey(123),
        v_prediction=args.v_pred,
    )
    arr = np.array(samples)
    print("sampled shape:", arr.shape)
    np.save("samples_landscapes.npy", arr)
    print("Wrote samples_landscapes.npy")


if __name__ == "__main__":
    main()
