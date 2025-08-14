import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = "../datasets/unified_topological_data_v6_semifast.pt"
OUT_DIR = "dataset_3d_surfaces"
SAMPLE_ID = 2  # which item in the dataset
CLIP_PCT = 0.5  # percentile clipping for nicer contrast (0 = off)

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load packed dataset
pack = torch.load(DATA_PATH, map_location="cpu")
lands = pack["landscapes"]  # (B, 3, KS, H, W) float32
B, D, KS, H, W = lands.shape
print(f"Loaded landscapes: {lands.shape}, dtype={lands.dtype}")

# NOTE: we didn't persist the (birth,death) box when generating.
# We'll assume [0,1] x [0,1] for axes. If you want exact axes,
# store 'box=bimod.get_box()' per-sample during data gen.
xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)  # (H,W)

# 2) Pick a sample and plot a 3 x KS grid: rows = H0,H1,H2 ; cols = k=0..KS-1
Z = lands[SAMPLE_ID].numpy()  # (3, KS, H, W)

fig = plt.figure(figsize=(4.5 * KS, 4.5 * 3), dpi=150)
for d in range(D):  # degree 0..2
    for k in range(KS):  # k-layer
        ax = fig.add_subplot(3, KS, d * KS + k + 1, projection="3d")
        z = Z[d, k]  # (H,W)

        # optional contrast clipping
        if CLIP_PCT > 0:
            lo = np.percentile(z, CLIP_PCT)
            hi = np.percentile(z, 100.0 - CLIP_PCT)
            if hi <= lo:
                hi = lo + 1e-6
            z = np.clip(z, lo, hi)

        # normalize for nice viewing (doesn't change data)
        zmin, zmax = float(z.min()), float(z.max())
        if zmax > zmin:
            zn = (z - zmin) / (zmax - zmin)
        else:
            zn = z * 0.0
        assert isinstance(ax, Axes3D)
        ax.plot_surface(
            X,
            Y,
            zn,
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.set_zlabel("λ")
        ax.set_title(f"H_{d}, k={k}")
        ax.view_init(elev=35, azim=-135)  # adjust view if you like

plt.tight_layout()
out_path = os.path.join(OUT_DIR, f"sample_{SAMPLE_ID:03d}_3d.png")
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved 3D surfaces → {out_path}")
# plt.show()  # uncomment for interactive rotation
