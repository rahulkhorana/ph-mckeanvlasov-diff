"""
eval_plots_metrics.py  — class-paired by default

Evaluate generated persistence landscapes (volumes) vs. real ones.

Key features
------------
- Loads real vols (and labels) from your packed .pt via dataloader.load_packed_pt
- Loads generated volumes from one or more *.npy files (glob)
- Renders (H, W, K, C) -> RGB for metrics/plots (mode: avgk / maxk / midk / slice:k=i)
- **Class-paired metrics by default**:
    * If --gen_labels is provided → strict class pairing.
    * Else → auto-assign fake samples to classes via nearest real class centroid
      in Inception feature space (pseudo-labels), then pair within class.
- Computes global FID/KID, per-class FID/KID, and PSNR/SSIM on class-matched pairs.
- Writes metrics.json and confusion.json (when pseudo-labeling), and plots.

Run (examples)
--------------
python eval_results.py \
  --real_pt ../../datasets/unified_topological_data_v6_semifast.pt \
  --gen_glob "runs/20250821_084559/generated/samples_step020000.npy" \
  --outdir gen_results_fig/ \
  --max_samples 2000 \
  --pairing class \
  --render avgk \
  --device cpu

Optional (if your saved .npy are still standardized):
  --gen_is_standardized --mu <mu> --sigma <sigma>

Dependencies
------------
numpy, torch, torchvision, pillow, matplotlib
Optional: scikit-image (SSIM), scipy (matrix sqrt for FID)

"""

import argparse
import json
import math
from pathlib import Path
from glob import glob
from typing import Tuple, List, Optional, Dict

import numpy as np

# Torch / vision (FID/KID & feature extraction)
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

# Plots
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Optional SSIM + sqrtm
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

# Your repo loader
from dataloader import load_packed_pt


# ----------------------- Rendering (H,W,K,C) -> (H,W,3) -----------------------


def render_rgb(vol: np.ndarray, mode: str = "avgk") -> np.ndarray:
    """
    vol: (H, W, K, C) float
    Returns uint8 (H, W, 3) for visualization / metrics.

    mode: 'avgk' | 'maxk' | 'midk' | 'slice:k=<int>'
    """
    assert vol.ndim == 4, f"Expected (H,W,K,C), got {vol.shape}"
    H, W, K, C = vol.shape
    if mode.startswith("slice:k="):
        ki = int(mode.split("=")[1])
        ki = max(0, min(K - 1, ki))
        img = vol[:, :, ki, :]
    elif mode == "avgk":
        img = vol.mean(axis=2)
    elif mode == "maxk":
        img = vol.max(axis=2)
    elif mode == "midk":
        img = vol[:, :, K // 2, :]
    else:
        raise ValueError(f"Unknown render mode: {mode}")

    if C >= 3:
        img = img[:, :, :3]
    else:
        pad = np.zeros((H, W, 3 - C), dtype=img.dtype)
        img = np.concatenate([img, pad], axis=-1)

    vmin = np.percentile(img, 1.0)
    vmax = np.percentile(img, 99.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    img01 = (img - vmin) / (vmax - vmin)
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


# ----------------------- PSNR / SSIM (paired) -----------------------


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_img(a: np.ndarray, b: np.ndarray) -> float:
    if not HAS_SKIMAGE:
        return float("nan")
    a01 = a.astype(np.float32) / 255.0
    b01 = b.astype(np.float32) / 255.0
    return float(ssim_fn(a01, b01, channel_axis=-1, data_range=1.0))  # type: ignore


# ----------------------- Inception features / FID / KID -----------------------


class InceptionFeatures(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = getattr(inception_v3, "IMAGENET1K_V1", None)
        model = inception_v3(weights=weights, transform_input=False, aux_logits=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.extractor = create_feature_extractor(
            model, return_nodes={"avgpool": "feat"}
        )
        self.device = device
        self.to(device)

        self.tfm = transforms.Compose(
            [
                transforms.Resize(
                    (299, 299),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    @torch.no_grad()
    def forward(
        self, pil_images: List[Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        xs = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            t = torch.stack([self.tfm(im.convert("RGB")) for im in batch], dim=0).to(  # type: ignore
                self.device
            )
            feats = self.extractor(t)["feat"].squeeze(-1).squeeze(-1)  # (N, 2048)
            xs.append(feats.cpu().numpy().astype(np.float64))
        return np.concatenate(xs, axis=0)


def _cov_mean(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    xc = feats - mu
    cov = (xc.T @ xc) / max(1, (feats.shape[0] - 1))
    return mu, cov


def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    if HAS_SCIPY:
        X = scipy_sqrtm(A)  # type: ignore
        if np.iscomplexobj(X):  # type: ignore
            X = X.real  # type: ignore
        return X  # type: ignore
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def fid_from_feats(fr: np.ndarray, ff: np.ndarray) -> float:
    mu_r, cov_r = _cov_mean(fr)
    mu_f, cov_f = _cov_mean(ff)
    diff = mu_r - mu_f
    cov_sqrt = _sqrtm_psd(cov_r @ cov_f)
    fid = float(diff @ diff + np.trace(cov_r + cov_f - 2.0 * cov_sqrt))
    return fid if np.isfinite(fid) else float("nan")


def _poly_kernel_3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3


def kid_from_feats(
    fr: np.ndarray,
    ff: np.ndarray,
    num_subsets: int = 10,
    subset_size: int = 1000,
    rng: np.random.RandomState | None = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.RandomState(123)
    n_r, n_f = fr.shape[0], ff.shape[0]
    m = min(subset_size, n_r, n_f)
    vals = []
    for _ in range(num_subsets):
        ir = rng.choice(n_r, m, replace=False)
        ifg = rng.choice(n_f, m, replace=False)
        xr, xf = fr[ir], ff[ifg]
        k_rr = _poly_kernel_3(xr, xr)
        np.fill_diagonal(k_rr, 0.0)
        k_ff = _poly_kernel_3(xf, xf)
        np.fill_diagonal(k_ff, 0.0)
        k_rf = _poly_kernel_3(xr, xf)
        mmd2 = (
            k_rr.sum() / (m * (m - 1)) + k_ff.sum() / (m * (m - 1)) - 2.0 * k_rf.mean()
        )
        vals.append(mmd2)
    vals = np.array(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std())


# ----------------------- Utilities -----------------------


def to_pil(img_uint8_hw3: np.ndarray) -> Image.Image:
    return Image.fromarray(img_uint8_hw3, mode="RGB")


def grid_image(
    pils: List[Image.Image], nrow: int = 8, pad: int = 2, bg: int = 255
) -> Image.Image:
    if len(pils) == 0:
        return Image.new("RGB", (10, 10), (bg, bg, bg))
    w, h = pils[0].size
    ncol = nrow
    nrows = (len(pils) + ncol - 1) // ncol
    W = ncol * w + (ncol - 1) * pad
    H = nrows * h + (nrows - 1) * pad
    canvas = Image.new("RGB", (W, H), (bg, bg, bg))
    for i, im in enumerate(pils):
        r, c = divmod(i, ncol)
        canvas.paste(im, (c * (w + pad), r * (h + pad)))
    return canvas


def save_hist(data_real: np.ndarray, data_fake: np.ndarray, path: Path, title: str):
    plt.figure(figsize=(6, 4))
    plt.hist(data_real.flatten(), bins=100, alpha=0.5, label="real", density=True)
    plt.hist(data_fake.flatten(), bins=100, alpha=0.5, label="fake", density=True)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def load_gen_labels(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".npy":
        return np.load(p)
    if p.suffix.lower() == ".npz":
        z = np.load(p)
        if "labels" in z:
            return z["labels"]
        # else: take first array
        for k in z.files:
            return z[k]
    if p.suffix.lower() in [".pt", ".pth"]:
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["labels", "y", "class", "classes"]:
                if k in obj:
                    return np.array(obj[k])
        if isinstance(obj, (list, tuple)):
            arr = np.array(obj)
            if arr.ndim == 1:
                return arr
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
    if p.suffix.lower() in [".txt", ".csv"]:
        return np.loadtxt(p, dtype=np.int64, delimiter=None)
    raise ValueError(f"Unsupported label file: {p}")


def class_centroids(feats: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        cents[int(c)] = feats[idx].mean(axis=0)
    return cents


def assign_to_centroids(feats: np.ndarray, cents: Dict[int, np.ndarray]) -> np.ndarray:
    classes = np.array(sorted(cents.keys()), dtype=np.int64)
    Cs = np.stack([cents[int(c)] for c in classes], axis=0)  # (C, D)
    # Euclidean distances
    # dist^2 = ||x||^2 + ||c||^2 - 2 x·c (argmin equiv. to argmax x·c - 0.5||c||^2)
    x2 = np.sum(feats * feats, axis=1, keepdims=True)  # (N,1)
    c2 = np.sum(Cs * Cs, axis=1, keepdims=True).T  # (1,C)
    dots = feats @ Cs.T  # (N,C)
    d2 = x2 + c2 - 2.0 * dots  # (N,C)
    idx = np.argmin(d2, axis=1)
    return classes[idx]


def make_class_pairs(
    labels_r: np.ndarray,
    labels_f: np.ndarray,
    limit_per_class: int,
    rng: np.random.RandomState,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    inter = np.intersect1d(np.unique(labels_r), np.unique(labels_f))
    for c in inter:
        ir = np.where(labels_r == c)[0]
        jf = np.where(labels_f == c)[0]
        rng.shuffle(ir)
        rng.shuffle(jf)
        n = min(len(ir), len(jf), limit_per_class if limit_per_class > 0 else 10**9)
        pairs.extend(list(zip(ir[:n], jf[:n])))
    return pairs


def choose_pairs_random(
    Nr: int, Nf: int, n: Optional[int], rng: np.random.RandomState
) -> List[Tuple[int, int]]:
    m = min(Nr, Nf) if n is None else min(n, Nr, Nf)
    ir = rng.choice(Nr, m, replace=False)
    jf = rng.choice(Nf, m, replace=False)
    return list(zip(ir, jf))


# ----------------------- Main -----------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real_pt",
        type=str,
        required=True,
        help="Path to packed .pt with real volumes (+labels)",
    )
    ap.add_argument(
        "--gen_glob",
        type=str,
        required=True,
        help='Glob for generated .npy files, e.g. ".../generated/samples_step*.npy"',
    )
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=2000)

    # Pairing strategy (default: class)
    ap.add_argument(
        "--pairing",
        type=str,
        default="class",
        choices=["class", "random", "paired"],
        help="class: pair within labels; random: unpaired random pairs; paired: index-aligned",
    )
    ap.add_argument(
        "--gen_labels",
        type=str,
        default=None,
        help="Optional labels for generated samples (.npy/.npz/.pt/.txt). If absent and pairing=class, fake labels are pseudo-assigned via Inception centroids.",
    )
    ap.add_argument(
        "--limit_per_class",
        type=int,
        default=1000,
        help="Cap pairs used per class for PSNR/SSIM",
    )

    ap.add_argument(
        "--render",
        type=str,
        default="avgk",
        choices=["avgk", "maxk", "midk"] + [f"slice:k={i}" for i in range(0, 512)],
        help="Map (H,W,K,C) -> (H,W,3) for metrics/plots",
    )

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch for Inception feature extraction",
    )

    ap.add_argument(
        "--per_class_fid", action="store_true", help="Compute per-class FID/KID"
    )
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument(
        "--gen_is_standardized",
        action="store_true",
        help="If generated .npy are standardized, pass mu/sigma to invert",
    )
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    # -------- Real data --------
    pack = load_packed_pt(args.real_pt, require_modules=False)
    if hasattr(pack, "vol"):
        real_vol = np.asarray(pack.vol)  # type: ignore
        real_labels = np.asarray(pack.labels) if hasattr(pack, "labels") else None  # type: ignore
    elif isinstance(pack, dict) and ("vol" in pack):
        real_vol = np.asarray(pack["vol"])
        real_labels = np.asarray(pack.get("labels")) if "labels" in pack else None
    else:
        raise ValueError("Could not find 'vol' in loaded pack.")

    Nr_full = real_vol.shape[0]
    Nr = min(args.max_samples, Nr_full)
    real_vol = real_vol[:Nr]
    if real_labels is not None:
        real_labels = real_labels[:Nr]

    # -------- Generated data --------
    gen_files = sorted(glob(args.gen_glob))
    if len(gen_files) == 0:
        raise ValueError(f"No generated files matched: {args.gen_glob}")
    gens = []
    for f in gen_files:
        arr = np.load(f)
        if arr.ndim != 5:
            raise ValueError(
                f"Expected generated shape (B,H,W,K,C) in {f}, got {arr.shape}"
            )
        gens.append(arr)
    gen_vol = np.concatenate(gens, axis=0)
    Nf_full = gen_vol.shape[0]
    Nf = min(args.max_samples, Nf_full)
    gen_vol = gen_vol[:Nf]

    # Optional inverse standardization for generated
    if args.gen_is_standardized:
        gen_vol = gen_vol.astype(np.float32) * float(args.sigma) + float(args.mu)

    # -------- Render to RGB (uint8) --------
    def render_many(vols: np.ndarray, mode: str) -> List[Image.Image]:
        return [to_pil(render_rgb(vols[i], mode=mode)) for i in range(vols.shape[0])]

    real_imgs = render_many(real_vol, args.render)
    fake_imgs = render_many(gen_vol, args.render)

    # Save quick grids
    grid_real = grid_image(real_imgs[:64], nrow=8)
    grid_fake = grid_image(fake_imgs[:64], nrow=8)
    grid_real.save(outdir / "figs/real_grid.png")
    grid_fake.save(outdir / "figs/fake_grid.png")

    # Histogram of voxel intensities (pre-render)
    try:
        data_r = real_vol[:Nr].astype(np.float32)
        data_f = gen_vol[:Nf].astype(np.float32)
        plt.figure(figsize=(6, 4))
        plt.hist(data_r.flatten(), bins=100, alpha=0.5, label="real", density=True)
        plt.hist(data_f.flatten(), bins=100, alpha=0.5, label="fake", density=True)
        plt.legend()
        plt.title("Voxel intensity hist")
        plt.tight_layout()
        plt.savefig(outdir / "figs/hist_real_vs_fake.png")
        plt.close()
    except Exception:
        pass

    # -------- Inception features (for FID/KID & pseudo-labeling) --------
    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    feat_net = InceptionFeatures(device=device)
    feats_real = feat_net(real_imgs, batch_size=args.batch_size)
    feats_fake = feat_net(fake_imgs, batch_size=args.batch_size)

    # -------- Determine fake labels for class pairing --------
    fake_labels: Optional[np.ndarray] = None
    pseudo_info = {"mode": None}

    if args.pairing == "paired":
        # index-aligned pairs; labels optional
        pass
    elif args.pairing == "class":
        if real_labels is None:
            print("[warn] real labels missing; falling back to random pairing.")
            args.pairing = "random"
        else:
            # try strict labels if provided
            fake_labels = load_gen_labels(args.gen_labels) if args.gen_labels else None
            if fake_labels is not None:
                fake_labels = np.asarray(fake_labels)[:Nf]
                pseudo_info["mode"] = "provided"  # type: ignore
            else:
                # pseudo-label by nearest real class centroid (Inception space)
                cents = class_centroids(feats_real, real_labels)
                fake_labels = assign_to_centroids(feats_fake, cents)
                pseudo_info["mode"] = "inception_nn"  # type: ignore
                # save confusion-like counts (class sizes)
                conf = {}
                for c in sorted(
                    set(int(x) for x in np.unique(real_labels))
                    | set(int(x) for x in np.unique(fake_labels))
                ):
                    conf[str(c)] = {
                        "real_count": (
                            int((real_labels == c).sum())
                            if real_labels is not None
                            else 0
                        ),
                        "fake_count": int((fake_labels == c).sum()),
                    }
                (outdir / "diagnostics").mkdir(exist_ok=True, parents=True)
                with open(outdir / "diagnostics/confusion_counts.json", "w") as f:
                    json.dump(conf, f, indent=2)

    # -------- FID/KID (global) --------
    fid_all = fid_from_feats(feats_real, feats_fake)
    rng_np = np.random.RandomState(args.seed)
    kid_mean, kid_std = kid_from_feats(
        feats_real, feats_fake, num_subsets=10, subset_size=1000, rng=rng_np
    )

    # -------- Per-class FID/KID (if requested and labels available) --------
    per_class: Dict[str, Dict[str, float]] = {}
    if args.per_class_fid and real_labels is not None:
        if args.pairing == "class" and fake_labels is not None:
            classes = sorted(
                set(np.unique(real_labels).tolist())
                & set(np.unique(fake_labels).tolist())
            )
            for c in classes:
                ir = np.where(real_labels == c)[0]
                jf = np.where(fake_labels == c)[0]
                if len(ir) >= 32 and len(jf) >= 32:
                    fr_c = feats_real[ir]
                    ff_c = feats_fake[jf]
                    fid_c = fid_from_feats(fr_c, ff_c)
                    kid_c_m, kid_c_s = kid_from_feats(
                        fr_c,
                        ff_c,
                        num_subsets=10,
                        subset_size=min(500, fr_c.shape[0], ff_c.shape[0]),
                        rng=rng_np,
                    )
                    per_class[str(int(c))] = {
                        "FID": float(fid_c),
                        "KID_mean": float(kid_c_m),
                        "KID_std": float(kid_c_s),
                    }
        else:
            # fallback: FID(real per-class) vs all fake (coarse)
            for c in np.unique(real_labels):
                ir = np.where(real_labels == c)[0]
                if len(ir) >= 32 and feats_fake.shape[0] >= 32:
                    fr_c = feats_real[ir]
                    fid_c = fid_from_feats(fr_c, feats_fake)
                    kid_c_m, kid_c_s = kid_from_feats(
                        fr_c,
                        feats_fake,
                        num_subsets=10,
                        subset_size=min(500, fr_c.shape[0], feats_fake.shape[0]),
                        rng=rng_np,
                    )
                    per_class[str(int(c))] = {
                        "FID_vs_all_fake": float(fid_c),
                        "KID_mean": float(kid_c_m),
                        "KID_std": float(kid_c_s),
                    }

    # -------- Build pairs for PSNR/SSIM --------
    if args.pairing == "paired":
        n = min(len(real_imgs), len(fake_imgs))
        pairs = [(i, i) for i in range(n)]
    elif args.pairing == "random":
        pairs = choose_pairs_random(len(real_imgs), len(fake_imgs), n=None, rng=rng)
    else:
        # class pairing
        pairs = make_class_pairs(real_labels, fake_labels, args.limit_per_class, rng)  # type: ignore

    # PSNR/SSIM over pairs
    psnrs, ssims = [], []
    for ir, jf in pairs:
        ar = np.array(real_imgs[ir], dtype=np.uint8)
        bf = np.array(fake_imgs[jf], dtype=np.uint8)
        psnrs.append(psnr(ar, bf, data_range=255.0))
        ssims.append(ssim_img(ar, bf))
    psnr_mean = float(np.mean(psnrs)) if len(psnrs) else float("nan")
    psnr_std = float(np.std(psnrs)) if len(psnrs) else float("nan")
    ssim_mean = float(np.nanmean(ssims)) if len(ssims) else float("nan")
    ssim_std = float(np.nanstd(ssims)) if len(ssims) else float("nan")

    # -------- Save per-class grids (if class pairing) --------
    if args.pairing == "class":
        (outdir / "figs/class_grids").mkdir(parents=True, exist_ok=True)
        # collect a few per class
        class_to_pairs: Dict[int, List[Tuple[int, int]]] = {}
        for ir, jf in pairs:
            c = int(fake_labels[jf]) if fake_labels is not None else -1
            class_to_pairs.setdefault(c, []).append((ir, jf))
        for c, plist in class_to_pairs.items():
            show = plist[:32]
            tiles = []
            for ir, jf in show:
                A = real_imgs[ir].copy()
                B = fake_imgs[jf].copy()
                w, h = A.size
                canvas = Image.new("RGB", (w * 2 + 4, h), (255, 255, 255))
                canvas.paste(A, (0, 0))
                canvas.paste(B, (w + 4, 0))
                tiles.append(canvas)
            if tiles:
                grid = grid_image(tiles, nrow=8)
                grid.save(outdir / f"figs/class_grids/class_{c}.png")

    # Also save overall pairs grid
    show_pairs = pairs[:64]
    pair_tiles = []
    for ir, jf in show_pairs:
        A = real_imgs[ir].copy()
        B = fake_imgs[jf].copy()
        w, h = A.size
        canvas = Image.new("RGB", (w * 2 + 4, h), (255, 255, 255))
        canvas.paste(A, (0, 0))
        canvas.paste(B, (w + 4, 0))
        pair_tiles.append(canvas)
    grid_pairs = grid_image(pair_tiles, nrow=8)
    grid_pairs.save(outdir / "figs/pairs_grid.png")

    # Hist of PSNR/SSIM
    if len(psnrs) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist([x for x in psnrs if np.isfinite(x)], bins=50)
        plt.title(f"PSNR histogram ({args.pairing})")
        plt.tight_layout()
        plt.savefig(outdir / "figs/psnr_hist.png")
        plt.close()
    if len(ssims) > 0 and np.isfinite(np.nanmean(ssims)):
        plt.figure(figsize=(6, 4))
        plt.hist([x for x in ssims if np.isfinite(x)], bins=50)
        plt.title(f"SSIM histogram ({args.pairing})")
        plt.tight_layout()
        plt.savefig(outdir / "figs/ssim_hist.png")
        plt.close()

    # -------- Write metrics --------
    metrics = {
        "counts": {"real": int(Nr), "fake": int(Nf), "pairs": int(len(pairs))},
        "render_mode": args.render,
        "pairing": args.pairing,
        "limit_per_class": int(args.limit_per_class),
        "PSNR_mean": psnr_mean,
        "PSNR_std": psnr_std,
        "SSIM_mean": ssim_mean,
        "SSIM_std": ssim_std,
        "FID": float(fid_all),
        "KID_mean": float(kid_mean),
        "KID_std": float(kid_std),
        "per_class": per_class,
        "notes": {
            "real_pt": args.real_pt,
            "gen_glob": args.gen_glob,
            "gen_labels_mode": (
                "provided"
                if (args.gen_labels is not None)
                else (pseudo_info.get("mode") or "none")
            ),
            "gen_is_standardized": bool(args.gen_is_standardized),
            "mu": float(args.mu),
            "sigma": float(args.sigma),
            "device_used": str(device),
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[done] wrote results to: {outdir}")


if __name__ == "__main__":
    main()
