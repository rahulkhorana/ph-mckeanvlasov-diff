# Usage: python viz_generated.py --npy samples_landscapes.npy --out generated_samples

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D proj)

plt.ioff()


def load_samples(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D array (B,H,W,K,C); got {arr.shape}")
    B, H, W, K, C = arr.shape
    if C != 3:
        raise ValueError(f"Last dim must be degrees=3; got C={C}")
    return arr.astype(np.float32)  # (B,H,W,K,3)


def robust_scale(x: np.ndarray, clip_pct: float = 1.0) -> np.ndarray:
    """Per-surface robust [0,1] scaling with percentile clipping."""
    a = x.copy()
    a[~np.isfinite(a)] = 0.0
    if clip_pct and clip_pct > 0:
        lo = np.percentile(a, clip_pct)
        hi = np.percentile(a, 100.0 - clip_pct)
        if hi <= lo:
            hi = lo + 1e-6
        a = np.clip(a, lo, hi)
    mn = a.min()
    mx = a.max()
    if mx <= mn + 1e-8:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def plot_sample_surfaces(
    sample: np.ndarray,
    out_path: str,
    stride: int = 2,
    elev: float = 45,
    azim: float = -135,
    cmap: str = "viridis",
    title: str | None = None,
):
    """
    sample: (H,W,K,3) float32
    saves a figure with 3 rows (degrees 0/1/2) x K columns (ks slices)
    """
    H, W, K, C = sample.shape
    assert C == 3, f"expected degrees=3, got {C}"
    X, Y = np.meshgrid(np.arange(W), np.arange(H))  # note: X is cols, Y is rows

    # figure size scaled by K
    fig = plt.figure(figsize=(4 * K, 9))  # width grows with K, 3 rows tall
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    for d in range(3):  # degrees rows
        for k in range(K):  # ks columns
            Z = robust_scale(sample[:, :, k, d])  # (H,W)
            ax = fig.add_subplot(3, K, d * K + (k + 1), projection="3d")
            assert isinstance(ax, Axes3D)
            ax.plot_surface(
                X[::stride, ::stride],
                Y[::stride, ::stride],
                Z[::stride, ::stride],
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                cmap=cmap,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])  # type: ignore
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1, 1, 0.35))  # flatter z so it’s readable
            ax.set_title(f"deg {d}, k {k}", pad=6, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", type=str, default="samples_landscapes.npy")
    ap.add_argument("--out", type=str, default="generated_samples")
    ap.add_argument("--max", type=int, default=16, help="max samples to render")
    ap.add_argument("--stride", type=int, default=2, help="surface downsample stride")
    ap.add_argument("--elev", type=float, default=45.0)
    ap.add_argument("--azim", type=float, default=-135.0)
    ap.add_argument("--cmap", type=str, default="viridis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    arr = load_samples(args.npy)  # (B,H,W,K,3)
    B, H, W, K, C = arr.shape
    print(f"loaded: {args.npy}  shape={arr.shape}")

    # also save a quick 2D heatmap grid for sanity (degree=1 by default)
    for i in range(min(B, args.max)):
        sample = arr[i]  # (H,W,K,3)
        # 3D grid figure
        out3d = os.path.join(args.out, f"sample_{i:03d}_3d.png")
        plot_sample_surfaces(
            sample,
            out3d,
            stride=args.stride,
            elev=args.elev,
            azim=args.azim,
            cmap=args.cmap,
        )

        # optional 2D quicklook: degree=1 across all k in a row
        fig, axes = plt.subplots(1, K, figsize=(3 * K, 3))
        for k in range(K):
            img = robust_scale(sample[:, :, k, 1])  # degree-1 heat
            axes[k].imshow(img, cmap=args.cmap)  # type: ignore
            axes[k].set_title(f"deg 1, k {k}", fontsize=9)  # type: ignore
            axes[k].axis("off")  # type: ignore
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"sample_{i:03d}_2d.png"), dpi=150)
        plt.close(fig)

    print(f"wrote 3D+2D previews to {args.out}/")


if __name__ == "__main__":
    main()
