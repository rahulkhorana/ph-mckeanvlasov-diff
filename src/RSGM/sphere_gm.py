import jax
import jax.numpy as jnp
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

# Import the corrected, unified components
from score_match import ScoreNet, train, sample_sde
from manifold_toy_models.manifold_utils import ManifoldWrapper


def visualize_on_sphere(
    x_samples: jnp.ndarray,
    title: str = "Generated samples on S^2",
    filename: str = "generated_samples.png",
):
    """Visualizes 3D points on a sphere and saves the plot to a file."""
    # We will visualize a 3D slice of the 4D data
    if x_samples.shape[-1] < 3:
        print(
            f"Visualization requires at least 3D data, but data has dimension {x_samples.shape[-1]}. Skipping."
        )
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the generated samples (taking the first 3 components)
    ax.scatter(
        x_samples[:, 0],
        x_samples[:, 1],
        x_samples[:, 2],
        alpha=0.7,
        c="r",
        label="Generated Samples (3D slice)",
    )

    # Draw a wireframe sphere for context
    u = jnp.linspace(0, 2 * jnp.pi, 100)
    v = jnp.linspace(0, jnp.pi, 100)
    x_sphere = jnp.outer(jnp.cos(u), jnp.sin(v))
    y_sphere = jnp.outer(jnp.sin(u), jnp.sin(v))
    z_sphere = jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
    ax.plot_wireframe(  # type: ignore
        x_sphere, y_sphere, z_sphere, color="b", rstride=10, cstride=10, alpha=0.2
    )

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")  # type: ignore
    ax.set_title(title)
    # Ensure the plot is not distorted
    ax.set_box_aspect([1.0, 1.0, 1.0])  # type: ignore
    ax.legend()

    # Save the figure to a file instead of showing it
    # plt.show()
    directory = os.path.dirname(filename)
    if directory:  # Check if a directory path is part of the filename
        os.makedirs(directory, exist_ok=True)
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    plt.close(fig)  # Close the figure to free memory


def main():
    """Main function to run the sphere experiment."""
    key = jax.random.PRNGKey(42)
    model_key, train_key, sample_key, data_key = jax.random.split(key, 4)

    # === 1. Set up manifold and visualize target data ===
    # The intrinsic dimension of the sphere is 3 (S^3)
    manifold_intrinsic_dim = 3
    manifold = ManifoldWrapper("sphere", dim=manifold_intrinsic_dim)

    # The data lives in the embedding dimension (dim + 1 for sphere)
    data_dim = manifold.embedded_dim

    print(f"--- Visualizing Target Distribution (S^3 in R^4) ---")
    x0_viz = manifold.sample_wrapped_normal(1024, key=data_key)
    visualize_on_sphere(
        x0_viz,  # type: ignore
        title="Target Distribution (3D slice of S^3)",
        filename="../plots/sphere_target_distribution.png",
    )

    # === 2. Initialize model ===
    # The model's dimension must match the data's EMBEDDING dimension.
    model = ScoreNet(dim=data_dim, width=256, depth=4, time_embed_dim=32, key=model_key)

    # === 3. Train the model ===
    print("\n--- Starting Training ---")
    model = train(model, manifold, steps=5000, lr=3e-4, batch_size=1024, key=train_key)
    print("--- Training Finished ---")

    # === 4. Generate samples from the trained model ===
    print("\n--- Starting Sampling ---")
    # The shape for sampling must also match the data's EMBEDDING dimension.
    samples = sample_sde(
        model, manifold, shape=(1024, data_dim), n_steps=1000, key=sample_key
    )
    print("--- Sampling Finished ---")

    # === 5. Visualize the generated samples ===
    print("\n--- Visualizing Generated Samples ---")
    visualize_on_sphere(
        samples,
        title="Generated Samples (3D slice of S^3)",
        filename="../plots/sphere_generated_samples.png",
    )

    # CORRECTED: Add a clean exit to prevent segmentation fault during interpreter shutdown.
    print("\nScript finished successfully. Exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
