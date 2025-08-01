import jax
import jax.numpy as jnp
import matplotlib

# Set the backend to 'Agg' to prevent interactive window errors (segfaults)
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add parent directory to path to import score_match,
# assuming score_match.py is in the parent directory of this script's location.
# Adjust the path if your file structure is different.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the core components
from score_match import ScoreNet, ManifoldWrapper, train, sample_sde


def visualize_on_3d_torus(
    x_samples: jnp.ndarray,
    title: str = "Generated samples on T^3",
    filename: str = "generated_samples_torus_3d.png",
):
    """Visualizes 3D points on a 3-torus and saves the plot to a file."""
    if x_samples.shape[-1] != 3:
        print(
            f"3-Torus visualization only supports 3D, but data has dimension {x_samples.shape[-1]}. Skipping."
        )
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the generated samples. For T^3, we can just plot the points in a cube.
    ax.scatter(
        x_samples[:, 0],
        x_samples[:, 1],
        x_samples[:, 2],
        alpha=0.5,
        c="r",
        label="Generated Samples",
    )

    ax.set_xlabel("X_1")
    ax.set_ylabel("X_2")
    ax.set_zlabel("X_3")  # type: ignore
    ax.set_title(title)
    # The torus is defined on [-pi, pi] for each axis
    ax.set_xlim(-jnp.pi, jnp.pi)
    ax.set_ylim(-jnp.pi, jnp.pi)
    ax.set_zlim(-jnp.pi, jnp.pi)  # type: ignore
    ax.set_box_aspect([1.0, 1.0, 1.0])  # type: ignore
    ax.legend()
    # plt.show()
    # Save the figure to a file
    directory = os.path.dirname(filename)
    if directory:  # Check if a directory path is part of the filename
        os.makedirs(directory, exist_ok=True)
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    plt.close(fig)  # Close the figure to free memory


def main():
    """Main function to run the 3-torus experiment."""
    key = jax.random.PRNGKey(44)  # Use a different key for a new experiment
    model_key, train_key, sample_key, data_key = jax.random.split(key, 4)

    # === 1. Set up manifold and visualize target data ===
    # A 3D Torus (T^3)
    manifold_dim = 2
    manifold = ManifoldWrapper("torus", dim=manifold_dim)

    # For a torus, the data dimension is the same as the intrinsic dimension
    data_dim = manifold.embedded_dim

    print(f"--- Visualizing Target Distribution (T^3) ---")
    x0_viz = manifold.sample_wrapped_normal(2048, key=data_key)
    visualize_on_3d_torus(
        x0_viz,  # type: ignore
        title="Target Distribution on T^3",
        filename="../plots/target_distribution_torus3d.png",
    )

    # === 2. Initialize model ===
    # The model's dimension must match the data's dimension.
    model = ScoreNet(dim=data_dim, width=256, depth=4, time_embed_dim=32, key=model_key)

    # === 3. Train the model ===
    print("\n--- Starting Training ---")
    model = train(model, manifold, steps=5000, lr=3e-4, batch_size=1024, key=train_key)
    print("--- Training Finished ---")

    # === 4. Generate samples from the trained model ===
    print("\n--- Starting Sampling ---")
    # The shape for sampling must also match the data's dimension.
    samples = sample_sde(
        model, manifold, shape=(2048, data_dim), n_steps=1000, key=sample_key
    )
    print("--- Sampling Finished ---")

    # === 5. Visualize the generated samples ===
    print("\n--- Visualizing Generated Samples ---")
    visualize_on_3d_torus(
        samples,
        title="Generated Samples on T^3",
        filename="../plots/generated_samples_torus3d.png",
    )

    # Use os._exit(0) for an immediate exit to prevent segfaults
    # during the interpreter's cleanup phase.
    print("\nScript finished successfully. Exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
