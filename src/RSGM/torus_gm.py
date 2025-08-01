import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))


# Import the core components
from score_match import ScoreNet, train, sample_sde
from manifold_toy_models.manifold_utils import (
    ManifoldWrapper,
)


def visualize_3d_samples(x_samples: jnp.ndarray, title: str, filename: str):
    """Visualizes 3D points and saves the plot to a file."""

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x_samples[:, 0], x_samples[:, 1], x_samples[:, 2], alpha=0.5, c="r")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.set_title(title)  # type: ignore
    ax.set_box_aspect([1.0, 1.0, 1.0])  # type: ignore
    ax.view_init(elev=20.0, azim=30)  # type: ignore

    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    plt.close(fig)


def main():
    """Main function to run the 3D donut (T^2) experiment."""
    key = jax.random.PRNGKey(50)  # New key for a new experiment
    model_key, train_key, sample_key, data_key = jax.random.split(key, 4)

    # === 1. Set up manifold and visualize target data ===
    # T^2 manifold, which we have defined to be embedded in R^3
    manifold = ManifoldWrapper("torus", dim=2)
    data_dim = manifold.embedded_dim  # This will correctly be 3

    print(f"--- Manifold: T^2 | Data Dimension: {data_dim} ---")
    print("--- Visualizing Target Distribution (3D Donut) ---")
    x0_viz = manifold.sample_wrapped_normal(2048, key=data_key)
    visualize_3d_samples(
        x0_viz, "Target Distribution on 3D Torus", "target_dist_donut.png"  # type: ignore
    )

    # === 2. Initialize model ===
    model = ScoreNet(dim=data_dim, width=256, depth=4, time_embed_dim=32, key=model_key)

    # === 3. Train the model ===
    print("\n--- Starting Training ---")
    model = train(model, manifold, steps=7000, lr=3e-4, batch_size=1024, key=train_key)
    print("--- Training Finished ---")

    # === 4. Generate samples from the trained model ===
    print("\n--- Starting Sampling ---")
    samples = sample_sde(
        model, manifold, shape=(2048, data_dim), n_steps=1000, key=sample_key
    )
    print("--- Sampling Finished ---")

    # === 5. Visualize the generated samples ===
    print("\n--- Visualizing Generated Samples ---")
    visualize_3d_samples(
        samples, "Generated Samples on 3D Torus", "generated_samples_donut.png"
    )

    print("\nScript finished successfully. Exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
