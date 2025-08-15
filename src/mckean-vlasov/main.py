import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax

from dataloader import load_packed_pt, train_val_split, iterate_batches, describe
from models import TinyUNet3D, EnergyNetwork, time_embed
from models import ModulesTrajectoryEncoder  # your encoder
from losses_steps import create_diffusion_state, diffusion_train_step
from sampling import ddim_sample, make_energy_guidance


def build_embedder(rng, out_dim=256, T_max=1, S_max=16):
    enc = ModulesTrajectoryEncoder(out_dim=out_dim)
    F = 13 + S_max
    feats_d = jnp.zeros((1, T_max, S_max, F), jnp.float32)
    set_d = jnp.zeros((1, T_max, S_max, 1), jnp.float32)
    time_d = jnp.ones((1, T_max, 1), jnp.float32)
    params = enc.init(rng, feats_d, set_d, time_d)["params"]

    from models import featurize_modules_trajectory  # if this lives with your encoder

    def wrap_trajectory(mods_batch):
        return [[mods] for mods in mods_batch]  # T=1

    def embed_fn(mods_batch):
        trajs = wrap_trajectory(mods_batch)
        feats, sets, times = [], [], []
        for tr in trajs:
            f, s, t = featurize_modules_trajectory(tr, T_max=T_max, S_max=S_max)
            feats.append(f)
            sets.append(s)
            times.append(t)
        F_b = jnp.stack(feats, 0)
        S_b = jnp.stack(sets, 0)
        T_b = jnp.stack(times, 0)
        return enc.apply({"params": params}, F_b, S_b, T_b)

    return embed_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_pt",
        type=str,
        default="../../datasets/unified_topological_data_v6_semifast.pt",
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_energy", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument(
        "--schedule", type=str, default="cosine", choices=["cosine", "linear"]
    )
    ap.add_argument("--v_pred", action="store_true")
    args = ap.parse_args()

    pack = load_packed_pt(args.data_pt, require_modules=True)
    print(describe(pack))
    train_ds, _ = train_val_split(pack, val_frac=0.1, seed=42)
    # (N,H,W,K,C) landscapes
    H, W, K, C = train_ds.lands.shape[1:]
    print(f"N={len(train_ds)}  vol=(N,H,W,K,C)=({len(train_ds)}, {H}, {W}, {K}, {C})")  # type: ignore

    rng = jax.random.PRNGKey(args.seed)
    rng, k_unet, k_e, k_enc, k_t = jax.random.split(rng, 5)

    B, H, W, K, C = args.batch, 128, 128, 3, 3  # from your data
    unet = TinyUNet3D(ch=64, m_dim=256)

    x_d = jnp.zeros((B, H, W, K, C), jnp.float32)
    t_d = jnp.zeros((B,), jnp.float32)
    temb = time_embed(t_d, dim=128)  # <- uses function above
    m_d = jnp.zeros((B, 256), jnp.float32)  # dummy modules embedding

    unet_params = unet.init(jax.random.PRNGKey(0), x_d, temb, m_d)["params"]
    y = unet.apply({"params": unet_params}, x_d, temb, m_d)
    assert y.shape == x_d.shape  # type: ignore

    # embedder
    embed_fn = build_embedder(k_enc, out_dim=256, T_max=1, S_max=16)

    # diffusion state
    diff_state = create_diffusion_state(
        rng=k_t,
        apply_fn=unet.apply,
        init_params=unet_params,
        T=args.T,
        lr=args.lr,
        v_prediction=args.v_pred,
        schedule=args.schedule,
    )

    # Energy net
    E = EnergyNetwork(ch=64, m_dim=256)
    E_params = E.init(k_e, x_d, m_d)["params"]
    e_tx = optax.adam(args.lr_energy)

    import flax.struct
    from flax.training.train_state import TrainState

    @flax.struct.dataclass
    class EnergyState(TrainState):
        pass

    e_state = EnergyState.create(apply_fn=E.apply, params=E_params, tx=e_tx)

    # data iterator
    it = iterate_batches(
        train_ds, batch_size=args.batch, shuffle=True, seed=args.seed, epochs=None
    )

    # train
    for step in range(1, args.steps + 1):
        batch = next(it)
        imgs = jnp.array(batch["lands"])  # (B,H,W,K,C)
        m_emb = embed_fn(batch["modules"])  # (B,256)

        diff_state, dloss = diffusion_train_step(
            diff_state,
            imgs,  # (B,H,W,K,C)
            jax.random.PRNGKey(step),
            v_prediction=bool(
                diff_state.v_prediction
            ),  # <- Python bool, becomes static
        )

        def energy_loss(params):
            e_pos = e_state.apply_fn({"params": params}, imgs, m_emb)  # (B,)
            e_neg = e_state.apply_fn(
                {"params": params}, imgs, jnp.roll(m_emb, 1, axis=0)  # type: ignore
            )
            return jnp.mean(jax.nn.softplus(e_pos - e_neg))

        eloss, grads = jax.value_and_grad(energy_loss)(e_state.params)
        e_state = e_state.apply_gradients(grads=grads)

        if step % 20 == 0:
            print(
                f"step {step:05d} | ddpm_loss={float(dloss):.4f} | energy_loss={float(eloss):.4f}"
            )

    # sampling with fixed modules from last batch
    last_m_emb = embed_fn(batch["modules"])  # type: ignore
    guidance = make_energy_guidance(e_state.apply_fn, e_state.params, last_m_emb)
    samples = ddim_sample(
        unet_apply=diff_state.apply_fn,
        params=diff_state.params,
        shape=(args.batch, H, W, K, C),
        betas=diff_state.betas,
        alphas=diff_state.alphas,
        alpha_bars=diff_state.alpha_bars,
        guidance_fn=guidance,
        guidance_scale=1.0,
        steps=50,
        rng=jax.random.PRNGKey(123),
        v_prediction=diff_state.v_prediction,
        return_all=False,
        m_emb=last_m_emb,
    )
    arr = np.array(samples)
    np.save("samples_landscapes.npy", arr)
    print("Saved → samples_landscapes.npy", arr.shape)


if __name__ == "__main__":
    main()
