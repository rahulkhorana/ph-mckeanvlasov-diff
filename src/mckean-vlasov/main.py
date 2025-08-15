# main.py — module-bridge conditioned 3D diffusion on MPL volumes
import argparse
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
from sampling import ddim_sample, make_module_bridge


def one_hot(y: np.ndarray, num_classes: int) -> jnp.ndarray:
    return jax.nn.one_hot(jnp.array(y), num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_pt",
        type=str,
        default="../../datasets/unified_topological_data_v6_semifast.pt",
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_energy", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    ap.add_argument("--v_pred", action="store_true", help="v-prediction")
    ap.add_argument("--cfg_drop", type=float, default=0.1, help="cond dropout prob")
    ap.add_argument("--S_max", type=int, default=16, help="max elements per module set")
    args = ap.parse_args()

    pack = load_packed_pt(args.data_pt, require_modules=True)
    print(describe(pack))
    train_ds, val_ds = train_val_split(pack, val_frac=0.1, seed=42)
    H, W, K, C = train_ds.vol.shape[1:]
    num_classes = (
        int(pack["labels"].max() + 1)
        if "labels" in pack
        else int(np.max(train_ds.labels) + 1)
    )

    # --- embedder for modules (frozen) ---
    rng = jax.random.PRNGKey(args.seed)
    rng, k_unet, k_energy, k_enc = jax.random.split(rng, 4)
    embed_fn = build_modules_embedder(k_enc, out_dim=256, T_max=1, S_max=args.S_max)

    cond_dim = 256 + num_classes  # modules (256) + one-hot label

    # --- UNet init ---
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
        v_prediction=args.v_pred,
        schedule=args.schedule,
        ema_decay=0.999,
    )

    # --- Energy net init ---
    E = EnergyNetwork(ch=64, cond_dim=cond_dim)
    E_params = E.init(k_energy, x_d, c_d)["params"]
    e_state = create_energy_state(E.apply, E_params, lr=args.lr_energy)

    # --- data iterator ---
    train_it = iterate_batches(
        train_ds, batch_size=args.batch, shuffle=True, seed=args.seed, epochs=None
    )

    # --- training ---
    for step in range(1, args.steps + 1):
        batch = next(train_it)
        vol = jnp.array(batch["vol"])  # (B,H,W,K,C)
        m_emb = embed_fn(batch["modules"])  # (B,256)
        y_oh = one_hot(batch["labels"], num_classes)  # (B,num_classes)
        cond = jnp.concatenate([m_emb, y_oh], axis=-1)  # (B,cond_dim)

        # classifier-free dropout on cond for diffusion
        key, rng = jax.random.split(rng)
        drop = jax.random.bernoulli(key, p=args.cfg_drop, shape=(vol.shape[0], 1))
        cond_train = cond * (1.0 - drop)

        # diffusion step
        key, rng = jax.random.split(rng)
        diff_state, dloss = diffusion_train_step(
            diff_state, vol, cond_train, key, v_prediction=bool(diff_state.v_prediction)
        )

        # energy step (no dropout)
        e_state, eloss = energy_train_step(e_state, vol, cond)

        if step % 20 == 0:
            print(
                f"step {step:05d} | ddpm_loss={float(dloss):.4f} | energy_loss={float(eloss):.4f}"
            )

    # --- sampling (conditional) ---
    # choose a target class and a set of modules from validation (or training) for guidance
    # here: take the first val batch
    val_it = iterate_batches(
        val_ds, batch_size=args.batch, shuffle=False, seed=123, epochs=1
    )
    val_batch = next(val_it)
    vol_ref = jnp.array(val_batch["vol"])  # for size only
    m_emb_ref = embed_fn(val_batch["modules"])
    y_oh_ref = one_hot(val_batch["labels"], num_classes)
    cond_ref = jnp.concatenate([m_emb_ref, y_oh_ref], axis=-1)

    # bridge uses the energy net on x0_hat with this cond_ref
    bridge = make_module_bridge(
        E_apply=e_state.train.apply_fn,
        eparams=e_state.train.params,
        cond_vec=cond_ref,
        T=diff_state.T,
    )

    samples = ddim_sample(
        unet_apply=diff_state.train.apply_fn,
        params=diff_state.ema_params,  # EMA weights for sampling
        shape=(cond_ref.shape[0], H, W, K, C),
        betas=diff_state.betas,
        alphas=diff_state.alphas,
        alpha_bars=diff_state.alpha_bars,
        cond_vec=cond_ref,
        steps=200,
        rng=jax.random.PRNGKey(123),
        v_prediction=bool(diff_state.v_prediction),
        bridge_fn=bridge,
        bridge_scale=1.0,
        return_all=False,
    )

    arr = np.array(samples)  # (B,H,W,K,C)
    np.save("samples_landscapes.npy", arr)
    print("Saved → samples_landscapes.npy", arr.shape)


if __name__ == "__main__":
    main()
