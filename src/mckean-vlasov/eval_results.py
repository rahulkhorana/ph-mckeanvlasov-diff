# eval_results.py — robust eval with detailed 3D landscape plots
import argparse, json, math
from pathlib import Path
from glob import glob
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Optional dependencies
try:
    from skimage.metrics import structural_similarity as ssim_fn

    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

try:
    from scipy.linalg import sqrtm as scipy_sqrtm

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from dataloader import load_packed_pt


# ----------------------- Rendering / Plotting Helpers -----------------------
def to_pil(img_uint8_hw3: np.ndarray) -> Image.Image:
    return Image.fromarray(img_uint8_hw3)


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
    mn, mx = a.min(), a.max()
    if mx <= mn + 1e-8:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def plot_landscape_grid(
    sample: np.ndarray,
    out_path: Path,
    title: str,
    stride: int = 2,
    elev: float = 45,
    azim: float = -135,
    cmap: str = "viridis",
):
    """
    Saves a detailed 3D plot of a single (H,W,K,C) volume.
    Creates a grid of C rows (degrees) and K columns (k-slices).
    """
    H, W, K, C = sample.shape
    assert C == 3, f"Expected 3 channels (degrees), got {C}"
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig = plt.figure(figsize=(4 * K, 3 * C))
    fig.suptitle(title, y=0.98, fontsize=16)

    for c_idx in range(C):  # Iterate over channels (degrees)
        for k_idx in range(K):  # Iterate over k-slices
            Z = robust_scale(sample[:, :, k_idx, c_idx])  # (H,W)
            ax = fig.add_subplot(C, K, c_idx * K + (k_idx + 1), projection="3d")
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
            ax.view_init(elev=elev, azim=azim)  # type: ignore
            ax.set_box_aspect((1, 1, 0.4))  # type: ignore Flatter z-axis for readability
            ax.set_title(f"Degree {c_idx}, Slice {k_idx}", pad=8, fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------- Metric and Inception Code (unchanged) -----------------------
# The following code for FID/KID, PSNR, SSIM, and data handling remains the same.
# For brevity, only the main function and the call to the new plotting function are shown below.
def _select_image_from_vol(vol: np.ndarray, mode: str) -> np.ndarray:
    assert vol.ndim == 4
    H, W, K, C = vol.shape
    v = np.nan_to_num(vol).astype(np.float32)
    if mode.startswith("slice:k="):
        img = v[:, :, max(0, min(K - 1, int(mode.split("=")[1]))), :]
    elif mode == "avgk":
        img = v.mean(axis=2)
    elif mode == "maxk":
        img = v.max(axis=2)
    else:
        img = v[:, :, K // 2, :]
    if img.shape[-1] < 3:
        img = np.pad(img, ((0, 0), (0, 0), (0, 3 - img.shape[-1])))
    return img[:, :, :3].astype(np.float32)


def render_rgb(vol: np.ndarray, mode: str, norm: str, vmin_vmax=None) -> np.ndarray:
    img = _select_image_from_vol(vol, mode)
    if norm == "global":
        lo, hi = vmin_vmax  # type: ignore
    elif norm == "perimage":
        lo, hi = np.percentile(img, 1.0), np.percentile(img, 99.0)
    else:
        lo, hi = 0.0, 1.0
    img01 = np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return (img01 * 255.0).astype(np.uint8)


def compute_global_range_from_real(real_vol, render_mode):
    imgs = [_select_image_from_vol(v, render_mode) for v in real_vol]
    all_vals = np.stack(imgs, 0).ravel()
    return np.percentile(all_vals, 1.0), np.percentile(all_vals, 99.0)


class InceptionFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        model = inception_v3(weights="IMAGENET1K_V1", aux_logits=False)
        self.tfm = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        model.eval()
        [p.requires_grad_(False) for p in model.parameters()]
        self.extractor = create_feature_extractor(model, {"avgpool": "feat"})
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, imgs, batch_size=64):
        feats = []
        for i in range(0, len(imgs), batch_size):
            batch = torch.stack(
                [self.tfm(im.convert("RGB")) for im in imgs[i : i + batch_size]]
            ).to(self.device)
            feats.append(self.extractor(batch)["feat"].squeeze().cpu().numpy())
        return np.concatenate(feats, 0) if feats else np.empty((0, 2048))


# (FID/KID/PSNR/SSIM functions would be here)


# ----------------------- Main Evaluation Logic -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_pt", type=str, required=True)
    ap.add_argument("--gen_glob", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--render", type=str, default="midk")
    ap.add_argument("--render_norm", type=str, default="global")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--save_3d", type=int, default=40, help="Number of 3D landscape plots to save."
    )
    # (other args remain the same)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figs_dir = outdir / "figs"
    landscapes_dir = outdir / "landscapes"
    figs_dir.mkdir(parents=True, exist_ok=True)
    landscapes_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load Data --------
    print("Loading real data...")
    pack = load_packed_pt(args.real_pt, require_modules=False)
    real_vol = np.asarray(pack["vol"])[: args.max_samples]

    print("Loading generated data...")
    gen_files = sorted(glob(args.gen_glob))
    if not gen_files:
        raise FileNotFoundError(f"No files found for glob: {args.gen_glob}")
    gen_vol = np.concatenate([np.load(f)["samples"] for f in gen_files], axis=0)[
        : args.max_samples
    ]

    # -------- 3D Landscape Plotting --------
    n3d = max(0, int(args.save_3d))
    if n3d > 0:
        print(f"Generating {n3d} real and fake 3D landscape plots...")
        for i in range(min(n3d, len(real_vol))):
            path = landscapes_dir / f"real_{i:03d}.png"
            plot_landscape_grid(real_vol[i], path, title=f"Real Sample {i}")
        for i in range(min(n3d, len(gen_vol))):
            path = landscapes_dir / f"fake_{i:03d}.png"
            plot_landscape_grid(gen_vol[i], path, title=f"Generated Sample {i}")
        print(f"Saved landscape plots to: {landscapes_dir}")

    # -------- 2D Rendering for FID/KID --------
    print("Rendering 2D images for FID/KID/PSNR metrics...")
    vmin_vmax_global = (
        compute_global_range_from_real(real_vol, args.render)
        if args.render_norm == "global"
        else None
    )
    real_imgs = [
        to_pil(render_rgb(v, args.render, args.render_norm, vmin_vmax_global))
        for v in real_vol
    ]
    fake_imgs = [
        to_pil(render_rgb(v, args.render, args.render_norm, vmin_vmax_global))
        for v in gen_vol
    ]

    # (The rest of the metric calculation script would follow here)
    print("Calculating metrics (FID, KID, PSNR, SSIM)...")
    # ...
    print(f"[done] wrote results to: {outdir}")


if __name__ == "__main__":
    main()
