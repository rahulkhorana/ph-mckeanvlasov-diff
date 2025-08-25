# eval_results.py — robust eval with LPIPS, FID, PSNR, and 3D plots
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

import lpips

from dataloader import load_packed_pt


# ----------------------- Rendering / Plotting Helpers -----------------------
# (These functions remain unchanged)
def robust_scale(x: np.ndarray, clip_pct: float = 1.0) -> np.ndarray:
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


def plot_landscape_grid(
    sample: np.ndarray,
    out_path: Path,
    title: str,
    stride: int = 2,
    elev: float = 45,
    azim: float = -135,
    cmap: str = "viridis",
):
    H, W, K, C = sample.shape
    assert C == 3
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    fig = plt.figure(figsize=(4 * K, 3 * C))
    fig.suptitle(title, y=0.98, fontsize=16)
    for c_idx in range(C):
        for k_idx in range(K):
            Z = robust_scale(sample[:, :, k_idx, c_idx])
            ax = fig.add_subplot(C, K, c_idx * K + k_idx + 1, projection="3d")
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
            ax.set_box_aspect((1, 1, 0.4))
            ax.set_title(f"Degree {c_idx}, Slice {k_idx}", pad=8, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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


# ----------------------- Metric Calculation Classes & Functions -----------------------
def psnr_img(a, b, data_range=255.0):
    return 20 * math.log10(data_range) - 10 * math.log10(np.mean((a - b) ** 2))


def ssim_img(a, b):
    return (
        float(ssim_fn(a, b, channel_axis=-1, data_range=255.0))  # type: ignore
        if HAS_SKIMAGE
        else np.nan
    )


def psnr_3d(a, b, data_range):
    return 20 * math.log10(data_range) - 10 * math.log10(np.mean((a - b) ** 2))


class InceptionFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
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


class LPIPSMetric:
    """Calculates Learned Perceptual Image Patch Similarity."""

    def __init__(self, device):
        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()
        self.device = device
        # LPIPS model expects images normalized to [-1, 1]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @torch.no_grad()
    def calculate(self, img1: Image.Image, img2: Image.Image) -> float:
        img1_t = self.transform(img1).unsqueeze(0).to(self.device)  # type: ignore
        img2_t = self.transform(img2).unsqueeze(0).to(self.device)  # type: ignore
        dist = self.model(img1_t, img2_t)
        return dist.item()


def fid_from_feats(fr, ff):
    mu_r, Cr = np.mean(fr, 0), np.cov(fr, rowvar=False)
    mu_f, Cf = np.mean(ff, 0), np.cov(ff, rowvar=False)
    d = np.sum((mu_r - mu_f) ** 2)
    Csr = scipy_sqrtm(Cr @ Cf).real if HAS_SCIPY else np.zeros_like(Cr)  # type: ignore
    return max(0, d + np.trace(Cr + Cf - 2 * Csr))


def kid_from_feats(fr, ff, n_sub=100, sub_size=1000, rng=None):
    rng = rng or np.random.RandomState(123)
    m = min(sub_size, fr.shape[0], ff.shape[0])
    if m < 20:
        return np.nan, np.nan
    vals = []
    for _ in range(n_sub):
        xr, xf = fr[rng.choice(len(fr), m, False)], ff[rng.choice(len(ff), m, False)]
        k_rr = (xr @ xr.T / xr.shape[1] + 1) ** 3
        np.fill_diagonal(k_rr, 0)
        k_ff = (xf @ xf.T / xf.shape[1] + 1) ** 3
        np.fill_diagonal(k_ff, 0)
        k_rf = (xr @ xf.T / xr.shape[1] + 1) ** 3
        mmd2 = k_rr.sum() / (m * (m - 1)) + k_ff.sum() / (m * (m - 1)) - 2 * k_rf.mean()
        vals.append(mmd2)
    return np.mean(vals), np.std(vals)


# ----------------------- Utilities -----------------------
def to_pil(img):
    return Image.fromarray(img)


def group_indices(labels):
    out = {}
    [out.setdefault(c, []).append(i) for i, c in enumerate(labels.astype(int))]
    return {k: np.array(v) for k, v in out.items()}


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
        "--pairing", type=str, default="class", choices=["class", "random", "paired"]
    )
    ap.add_argument("--per_class_fid", action="store_true")
    ap.add_argument("--min_fid_n", type=int, default=20)
    ap.add_argument("--save_3d", type=int, default=40)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "landscapes").mkdir(parents=True, exist_ok=True)

    pack = load_packed_pt(args.real_pt, require_modules=False)
    real_vol, real_labels = (
        np.asarray(pack["vol"])[: args.max_samples],
        np.asarray(pack.get("labels"))[: args.max_samples],
    )
    gen_files = sorted(glob(args.gen_glob))
    if not gen_files:
        raise FileNotFoundError(f"No files for glob: {args.gen_glob}")
    gen_vol = np.concatenate([np.load(f)["samples"] for f in gen_files])[
        : args.max_samples
    ]
    gen_labels = np.concatenate([np.load(f)["labels"] for f in gen_files])[
        : args.max_samples
    ]

    vmin_vmax = (
        compute_global_range_from_real(real_vol, args.render)
        if args.render_norm == "global"
        else None
    )
    real_imgs = [
        to_pil(render_rgb(v, args.render, args.render_norm, vmin_vmax))
        for v in real_vol
    ]
    fake_imgs = [
        to_pil(render_rgb(v, args.render, args.render_norm, vmin_vmax)) for v in gen_vol
    ]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    inc = InceptionFeatures(device)
    feats_real, feats_fake = inc(real_imgs), inc(fake_imgs)
    lpips_metric = LPIPSMetric(device)

    global_fid = fid_from_feats(feats_real, feats_fake)
    global_kid_mean, global_kid_std = kid_from_feats(feats_real, feats_fake)

    rng = np.random.RandomState(123)
    if args.pairing == "paired":
        pairs = list(
            zip(
                range(min(len(real_imgs), len(fake_imgs))),
                range(min(len(real_imgs), len(fake_imgs))),
            )
        )
    elif args.pairing == "class" and real_labels is not None and gen_labels is not None:
        gr, gf = group_indices(real_labels), group_indices(gen_labels)
        pairs = []
        [
            pairs.extend(zip(ir[: min(len(ir), len(jf))], jf[: min(len(ir), len(jf))]))
            for c in sorted(set(gr.keys()) & set(gf.keys()))
            for ir, jf in [(gr[c], gf[c])]
            for _ in [rng.shuffle(ir), rng.shuffle(jf)]
        ]
    else:
        m = min(len(real_imgs), len(fake_imgs))
        pairs = list(
            zip(
                rng.choice(len(real_imgs), m, False),
                rng.choice(len(fake_imgs), m, False),
            )
        )

    psnrs, ssims, psnrs3d, lpips_scores = [], [], [], []
    v_min, v_max = np.percentile(real_vol, [0.1, 99.9])
    for ir, jf in pairs:
        psnrs.append(psnr_img(np.array(real_imgs[ir]), np.array(fake_imgs[jf])))
        ssims.append(ssim_img(np.array(real_imgs[ir]), np.array(fake_imgs[jf])))
        psnrs3d.append(psnr_3d(real_vol[ir], gen_vol[jf], data_range=v_max - v_min))
        if lpips_metric:
            lpips_scores.append(lpips_metric.calculate(real_imgs[ir], fake_imgs[jf]))

    per_class_metrics = {}
    if args.per_class_fid and real_labels is not None and gen_labels is not None:
        gr, gf = group_indices(real_labels), group_indices(gen_labels)
        for c in sorted(set(gr.keys()) & set(gf.keys())):
            ir, jf = gr[c], gf[c]
            if len(ir) >= args.min_fid_n and len(jf) >= args.min_fid_n:
                per_class_metrics[str(c)] = {
                    "fid": fid_from_feats(feats_real[ir], feats_fake[jf])
                }

    n3d = max(0, int(args.save_3d))
    for i in range(min(n3d, len(real_vol))):
        plot_landscape_grid(
            real_vol[i], outdir / f"landscapes/real_{i:03d}.png", f"Real {i}"
        )
    for i in range(min(n3d, len(gen_vol))):
        plot_landscape_grid(
            gen_vol[i], outdir / f"landscapes/fake_{i:03d}.png", f"Fake {i}"
        )

    metrics = {
        "counts": {"real": len(real_vol), "fake": len(gen_vol), "pairs": len(pairs)},
        "PSNR_mean": float(np.mean(psnrs)) if psnrs else None,
        "PSNR_std": float(np.std(psnrs)) if psnrs else None,
        "SSIM_mean": float(np.nanmean(ssims)) if ssims else None,
        "SSIM_std": float(np.nanstd(ssims)) if ssims else None,
        "LPIPS_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
        "LPIPS_std": float(np.std(lpips_scores)) if lpips_scores else None,
        "PSNR_3D_mean": float(np.mean(psnrs3d)) if psnrs3d else None,
        "PSNR_3D_std": float(np.std(psnrs3d)) if psnrs3d else None,
        "global_FID": float(global_fid) if global_fid is not None else None,
        "global_KID_mean": (
            float(global_kid_mean) if global_kid_mean is not None else None
        ),
        "global_KID_std": float(global_kid_std) if global_kid_std is not None else None,
        "per_class": per_class_metrics,
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[done] wrote results to: {outdir}")


if __name__ == "__main__":
    main()
