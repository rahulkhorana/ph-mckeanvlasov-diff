import plotly.graph_objects as go
import numpy as np
import torch
from pathlib import Path

# --- Configuration ---
# Adjust these paths to point to your data files
REAL_PT_PATH = "../../datasets/unified_topological_data_v6_semifast.pt"
GEN_NPZ_PATH = "../gpu-result/samples_landscapes.npz"
OUTPUT_DIR = Path("paper_figures_plotly")


# --- Helper Functions ---
def robust_scale(x: np.ndarray, clip_pct: float = 1.0) -> np.ndarray:
    """Per-surface robust [0,1] scaling with percentile clipping."""
    a = x.copy()
    a[~np.isfinite(a)] = 0.0
    if clip_pct > 0:
        lo, hi = np.percentile(a, [clip_pct, 100.0 - clip_pct])
        if hi <= lo:
            hi = lo + 1e-6
        a = np.clip(a, lo, hi)
    mn, mx = a.min(), a.max()
    if mx <= mn + 1e-8:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


# --- 1. Generate the Loss Landscape Surface ---
def calculate_z(x, y):
    # A complex, non-symmetrical function for a more realistic landscape
    term1 = np.sin(np.sqrt(x**2 + y**2))
    term2 = np.cos(0.5 * x) * np.sin(0.5 * y)
    term3 = np.sin(x * 0.2 + y * 0.5) * 2
    bowl = x**2 * 0.05 + y**2 * 0.1
    return term1 + term2 + term3 - bowl


x_grid = np.linspace(-10, 10, 150)
y_grid = np.linspace(-8, 8, 150)
X, Y = np.meshgrid(x_grid, y_grid)
Z = calculate_z(X, Y)

# --- 2. Generate the Stochastic Trajectory ON the Surface ---
start_point_2d = np.array([-8.0, 6.0])
z_flat = Z.flatten()
min_idx = np.argmin(z_flat)
end_point_2d = np.array([X.flatten()[min_idx], Y.flatten()[min_idx]])

num_steps = 150
t = np.linspace(0, 1, num_steps)[:, np.newaxis]
path_2d = (1 - t) * start_point_2d + t * end_point_2d

noise_scale = 0.8
random_steps = np.random.randn(num_steps, 2) * noise_scale
noise = np.cumsum(random_steps, axis=0)
noise -= t * noise[-1]  # Brownian Bridge
jagged_path_2d = path_2d + noise * (1 - t)

path_x = jagged_path_2d[:, 0]
path_y = jagged_path_2d[:, 1]
path_z = calculate_z(path_x, path_y)
jagged_path_3d = np.vstack([path_x, path_y, path_z]).T

# --- 3. Load Real and Fake Landscape Data ---
print("Loading landscape data...")
# Load a fake landscape to use as the endpoint of the trajectory
fake_data = np.load(GEN_NPZ_PATH)
fake_vols = fake_data["samples"]
end_point_landscape = fake_vols[np.random.choice(len(fake_vols))]

# Create a noise landscape for the start point
start_point_landscape = np.random.randn(*end_point_landscape.shape) * np.std(
    end_point_landscape
)

# Create an intermediate landscape by interpolating
mid_point_landscape = (start_point_landscape * 0.25) + (end_point_landscape * 0.75)


# --- 4. Create the Plotly Figure ---
fig = go.Figure()

# Add the main landscape surface
fig.add_trace(
    go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale="Magma",
        opacity=0.8,
        showscale=False,
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="white",
            project_z=True,
            highlightwidth=1,
        ),
    )
)

# Add the trajectory trace
fig.add_trace(
    go.Scatter3d(
        x=jagged_path_3d[:, 0],
        y=jagged_path_3d[:, 1],
        z=jagged_path_3d[:, 2] + 0.1,
        mode="lines",
        line=dict(color="black", width=8),
        name="Stochastic Trajectory",
    )
)


# --- 5. Create and position the 3D PH landscape insets ---
def create_landscape_surface(volume, position, name, cmap, scale=1.8):
    """Creates a Plotly Surface trace for a PH landscape."""
    H, W, K, C = volume.shape
    # We'll visualize one slice and degree for clarity, e.g., k=1, degree=1
    Z_ph = robust_scale(volume[:, :, K // 2, C // 2]) * scale
    X_ph, Y_ph = np.meshgrid(
        np.linspace(-scale, scale, W), np.linspace(-scale, scale, H)
    )
    return go.Surface(
        x=X_ph + position[0],
        y=Y_ph + position[1],
        z=Z_ph + position[2] + 1.0,  # Offset to float above the path
        surfacecolor=Z_ph,  # This maps the Z values to the colorscale
        colorscale=cmap,
        showscale=False,
        opacity=0.9,
        name=name,
        cmin=Z_ph.min(),
        cmax=Z_ph.max(),
    )


# Get positions for the insets from the trajectory path
start_pos = jagged_path_3d[0]
mid_pos = jagged_path_3d[len(jagged_path_3d) // 2]
end_pos = jagged_path_3d[-1]

# Add the landscape traces to the figure
fig.add_trace(
    create_landscape_surface(start_point_landscape, start_pos, "t=T (Noise)", "viridis")
)
fig.add_trace(
    create_landscape_surface(mid_point_landscape, mid_pos, "t=T/2", "viridis")
)
fig.add_trace(
    create_landscape_surface(
        end_point_landscape, end_pos, "t=0 (Final Sample)", "viridis"
    )
)


def make_ticks(old_min, old_max, new_min, new_max, n=6, as_int=False):
    vals = np.linspace(old_min, old_max, n)
    labels = np.linspace(new_min, new_max, n)
    if as_int:
        labels = [f"{int(round(v))}" for v in labels]
    else:
        labels = [
            f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}" for v in labels
        ]
    return vals, labels


# Current numeric extents in your data
x_old_min, x_old_max = x_grid.min(), x_grid.max()  # -10 .. 10
y_old_min, y_old_max = y_grid.min(), y_grid.max()  # -8 .. 8
z_old_min, z_old_max = float(np.nanmin(Z)), float(np.nanmax(Z))

# Desired label ranges
TIME_MAX = 1000
E_MAX = 12  # set your preferred positive energy cap here
LOSS_MAX = 10

x_vals, x_text = make_ticks(x_old_min, x_old_max, 0, TIME_MAX, n=6, as_int=True)
y_vals, y_text = make_ticks(y_old_min, y_old_max, 0, E_MAX, n=6, as_int=True)
z_vals, z_text = make_ticks(z_old_min, z_old_max, 0, LOSS_MAX, n=6, as_int=True)


camera = dict(
    eye=dict(x=1.8, y=-1.8, z=1.4),  # diagonal top-right view
    center=dict(x=0, y=0, z=0),
)

# Balanced proportions (not too stretched)
aspect = dict(x=1.8, y=1.0, z=0.6)  # widen Time axis, flatten Loss a bit

fig.update_layout(
    title_text="McKean–Vlasov SDE Trajectory with Evolving Landscapes",
    template="plotly_white",
    height=1000,
    width=1200,
    scene=dict(
        # Titles stay the same
        xaxis_title="Time",
        yaxis_title="Energy",
        zaxis_title="Loss",
        # Geometry unchanged; only ticks are re-labeled
        xaxis=dict(
            tickmode="array",
            tickvals=x_vals,
            ticktext=x_text,
            ticks="outside",
            showspikes=False,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_vals,
            ticktext=y_text,
            ticks="outside",
            showspikes=False,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        zaxis=dict(
            tickmode="array",
            tickvals=z_vals,
            ticktext=z_text,  # 0 at bottom label, 10 at top label
            ticks="outside",
            showspikes=False,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        camera=camera,
        aspectratio=aspect,
    ),
    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01),
    margin=dict(l=8, r=8, t=56, b=8),
)


def scale_with(x: np.ndarray, lo: float = 0.0, hi: float = 10.0) -> np.ndarray:
    """Linearly rescale an array to [lo, hi]."""
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax <= xmin + 1e-12:
        return np.full_like(x, (lo + hi) / 2.0)
    return (x - xmin) / (xmax - xmin) * (hi - lo) + lo


Z = scale_with(Z, 0.0, 10.0)

fig2 = go.Figure()
fig2.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=Z,
        colorscale="Magma",
        contours=dict(showlabels=True),
        line_smoothing=0.85,
        colorbar_title="Loss",
    )
)
fig2.add_trace(
    go.Scatter(
        x=path_x,
        y=path_y,
        mode="lines",
        line=dict(color="black", width=3),
        name="Trajectory",
    )
)
fig2.update_layout(
    template="plotly_white",
    xaxis_title="Time",
    yaxis_title="Energy",
    xaxis=dict(
        range=[x_grid.min(), x_grid.max()],
        tickmode="array",
        tickvals=np.linspace(-10, 10, 6),
        ticktext=[0, 200, 400, 600, 800, 1000],
    ),  # relabel if you’re using the label-mapping trick
    yaxis=dict(
        range=[y_grid.min(), y_grid.max()],
        tickmode="array",
        tickvals=np.linspace(-8, 8, 6),
        ticktext=[0, 2.4, 4.8, 7.2, 9.6, 12],
    ),
    width=900,
    height=650,
)

# --- 7. Save the Figure ---
OUTPUT_DIR.mkdir(exist_ok=True)
output_file = OUTPUT_DIR / "loss_landscape_with_ph.html"
fig.write_html(output_file)
print(f"Figure saved to {output_file}. Open this file in a web browser to view.")

out_file_2 = OUTPUT_DIR / "loss_landscape_contour.html"
fig2.write_html(out_file_2)
print(f"Contour figure saved to {out_file_2}. Open this file in a web browser to view.")
