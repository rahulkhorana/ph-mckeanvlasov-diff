# main.py
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax

from dataloader import load_packed_pt, train_val_split, iterate_batches, describe
from models import (
    TinyUNet,
    EnergyNetwork,
    ModulesTrajectoryEncoder,
    featurize_modules_trajectory,
)
from losses_steps import (
    diffusion_train_step,
    energy_train_step,
    create_diffusion_state,
    create_energy_state,
)
from sampling import ddim_sample, make_energy_guidance


# --------- small helper to build a frozen modules-encoder -> embed_fn ----------


def build_embed_fn(rng, out_dim=256, T_MAX=1, S_MAX=16):
    enc = ModulesTrajectoryEncoder(out_dim=out_dim)
    F = 13 + S_MAX  # robust stats + pos IDs
    feats_d = jnp.zeros((1, T_MAX, S_MAX, F), jnp.float32)
    set_d = jnp.zeros((1, T_MAX, S_MAX, 1), jnp.float32)
    time_d = jnp.ones((1, T_MAX, 1), jnp.float32)
    enc_params = enc.init(rng, feats_d, set_d, time_d)["params"]

    def as_single_step_traj(mods_batch):
        return [[mods] for mods in mods_batch]  # wrap flat list -> T=1

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

    return embed_fn


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

    # dataset normalization (helps a LOT)
    mu = float(train_ds.lands.mean())
    std = float(train_ds.lands.std() + 1e-6)
    norm = lambda x: (x - mu) / std
    denorm = lambda x: x * std + mu

    # ----- build models & states -----
    rng = jax.random.PRNGKey(args.seed)
    rng, k_unet, k_energy, k_enc = jax.random.split(rng, 4)

    unet = TinyUNet(ch=128)  # a bit more juice than 64
    diff_state = create_diffusion_state(
        rng=k_unet,
        unet=unet,
        input_shape=(args.batch, H, W, C),
        lr=args.lr,
        T=args.diffusion_T,
        schedule=args.schedule,
        v_prediction=args.v_pred,
        ema_decay=0.999,
    )

    # Energy network
    x_d = jnp.zeros((args.batch, H, W, C), jnp.float32)
    E = EnergyNetwork(ch=64, m_dim=256)
    E_params = E.init(k_energy, x_d, jnp.zeros((args.batch, 256), jnp.float32))[
        "params"
    ]
    e_state = create_energy_state(E.apply, E_params, lr=args.lr_energy)

    # Frozen modules encoder -> embed_fn (returns (B,256))
    embed_fn = build_embed_fn(k_enc, out_dim=256, T_MAX=1, S_MAX=16)

    # ----- training loop -----
    for step in range(1, args.steps + 1):
        batch = next(
            train_iter
        )  # dict: lands (B,H,W,C float32), modules: List[Any] len B
        imgs = jnp.array(norm(batch["lands"]))  # normalize
        m_emb = embed_fn(batch["modules"])  # (B,256) jnp

        # diffusion step
        diff_state, dloss = diffusion_train_step(
            diff_state, imgs, jax.random.PRNGKey(step)
        )

        # energy step
        e_state, eloss = energy_train_step(e_state, imgs, m_emb)

        if step % 20 == 0:
            print(
                f"step {step:05d} | ddpm_loss={float(dloss):.4f} | energy_loss={float(eloss):.4f}"
            )

    # ----- sampling with energy guidance (use last batch's modules) -----
    guidance = make_energy_guidance(
        e_state.apply_fn, e_state.params, embed_fn, batch["modules"], diff_state.T  # type: ignore
    )

    samples = ddim_sample(
        diff_state.apply_fn,
        diff_state.ema_params,
        (args.batch, H, W, C),
        diff_state.betas,
        diff_state.alphas,
        diff_state.alpha_bars,
        guidance_fn=guidance,
        guidance_scale=1.0,
        steps=200,
        rng=jax.random.PRNGKey(123),
        v_prediction=diff_state.v_prediction,
        return_all=False,
    )
    samples = np.array(denorm(samples))  # back to data scale

    np.save("samples_landscapes.npy", samples)
    print("sampled shape:", samples.shape)
    print("Wrote samples_landscapes.npy")


if __name__ == "__main__":
    main()
