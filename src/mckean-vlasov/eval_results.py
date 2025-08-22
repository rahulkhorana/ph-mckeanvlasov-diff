# eval_results.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
    v = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if mode.startswith("slice:k="):
        ki = int(mode.split("=")[1])
        ki = max(0, min(K - 1, ki))
        img = v[:, :, ki, :]
    elif mode == "avgk":
        img = v.mean(axis=2)
    elif mode == "maxk":
        img = v.max(axis=2)
    elif mode == "midk":
        img = v[:, :, K // 2, :]
    else:
        raise ValueError(f"Unknown render mode: {mode}")

    # ensure 3 channels
    if img.shape[-1] >= 3:
        img = img[:, :, :3]
    else:
        pad = np.zeros((H, W, 3 - img.shape[-1]), dtype=img.dtype)
        img = np.concatenate([img, pad], axis=-1)

    # robust percentile scaling
    vmin = np.percentile(img, 1.0)
    vmax = np.percentile(img, 99.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(img.min()), float(img.max()) + 1e-6

    img01 = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


# ----------------------- PSNR / SSIM (paired) -----------------------
def psnr_img(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(float(mse))


def psnr_3d(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    if mse <= 1e-20:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(float(mse))


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
        # Use official weights and their transforms if available
        try:
            from torchvision.models import Inception_V3_Weights

            weights = Inception_V3_Weights.IMAGENET1K_V1
            model = inception_v3(weights=weights, aux_logits=True)  # <-- must be True
            self.tfm = weights.transforms()
        except Exception:
            # Fallback for older torchvision
            weights = getattr(inception_v3, "IMAGENET1K_V1", None)
            model = inception_v3(
                weights=weights, aux_logits=True, transform_input=False
            )
            self.tfm = transforms.Compose(
                [
                    transforms.Resize(
                        (299, 299),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # We only need the main trunk’s global avgpool features.
        # 'avgpool' node exists regardless of aux head presence.
        self.extractor = create_feature_extractor(
            model, return_nodes={"avgpool": "feat"}
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, imgs: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        xs = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            if not batch:
                break
            t = torch.stack([self.tfm(im.convert("RGB")) for im in batch], 0).to(self.device)  # type: ignore
            f = self.extractor(t)["feat"].squeeze(-1).squeeze(-1)  # (N, 2048)
            xs.append(f.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(xs, 0) if xs else np.empty((0, 2048), np.float64)


def _cov_mean(
    X: np.ndarray, shrink: float = 0.1, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Ledoit-Wolf style diagonal shrinkage to avoid insane FID when N is small."""
    mu = X.mean(axis=0)
    Z = X - mu
    n, d = Z.shape
    if n <= 1:
        return mu, np.eye(d, dtype=np.float64)
    S = (Z.T @ Z) / max(1, n - 1)
    tr_d = float(np.trace(S)) / max(1, d)
    cov = (1.0 - shrink) * S + shrink * tr_d * np.eye(d, dtype=np.float64)
    cov = cov + (eps * tr_d + 1e-12) * np.eye(d, dtype=np.float64)
    return mu, cov


def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    if HAS_SCIPY:
        X = scipy_sqrtm(A)  # type: ignore
        return X.real if np.iscomplexobj(X) else X  # type: ignore
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def fid_from_feats(
    fr: np.ndarray, ff: np.ndarray, shrink: float = 0.1
) -> Optional[float]:
    if fr.shape[0] == 0 or ff.shape[0] == 0:
        return None
    mu_r, Cr = _cov_mean(fr, shrink)
    mu_f, Cf = _cov_mean(ff, shrink)
    d = mu_r - mu_f
    Csr = _sqrtm_psd(Cr @ Cf)
    val = float(d @ d + np.trace(Cr + Cf - 2.0 * Csr))
    if not np.isfinite(val):
        return None
    # never negative due to numeric noise
    return max(val, 0.0)


def _poly_kernel_3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3


def kid_from_feats(
    fr: np.ndarray,
    ff: np.ndarray,
    num_subsets=10,
    subset_size=1000,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if fr.shape[0] == 0 or ff.shape[0] == 0:
        return None, None
    rng = rng or np.random.RandomState(123)
    m = min(subset_size, fr.shape[0], ff.shape[0])
    if m < 20:  # too small; skip
        return None, None
    vals = []
    for _ in range(num_subsets):
        ir = rng.choice(fr.shape[0], m, replace=False)
        jf = rng.choice(ff.shape[0], m, replace=False)
        xr, xf = fr[ir], ff[jf]
        k_rr = _poly_kernel_3(xr, xr)
        np.fill_diagonal(k_rr, 0.0)
        k_ff = _poly_kernel_3(xf, xf)
        np.fill_diagonal(k_ff, 0.0)
        k_rf = _poly_kernel_3(xr, xf)
        mmd2 = (
            k_rr.sum() / (m * (m - 1)) + k_ff.sum() / (m * (m - 1)) - 2.0 * k_rf.mean()
        )
        vals.append(mmd2)
    vals = np.asarray(vals, np.float64)
    return float(vals.mean()), float(vals.std())


# ----------------------- Utilities -----------------------
def to_pil(img_uint8_hw3: np.ndarray) -> Image.Image:
    return Image.fromarray(img_uint8_hw3)  # Pillow infers "RGB" from HxWx3 uint8


def grid_image(
    pils: List[Image.Image], nrow: int = 8, pad: int = 2, bg: int = 255
) -> Image.Image:
    if len(pils) == 0:
        return Image.new("RGB", (10, 10), (bg, bg, bg))
    w, h = pils[0].size
    ncol = nrow
    rows = (len(pils) + ncol - 1) // ncol
    W = ncol * w + (ncol - 1) * pad
    H = rows * h + (rows - 1) * pad
    canvas = Image.new("RGB", (W, H), (bg, bg, bg))
    for i, im in enumerate(pils):
        r, c = divmod(i, ncol)
        canvas.paste(im, (c * (w + pad), r * (h + pad)))
    return canvas


def group_indices(labels: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, List[int]] = {}
    for i, c in enumerate(labels.astype(int).tolist()):
        out.setdefault(c, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def pairs_within_class(
    ir: np.ndarray, jf: np.ndarray, limit: int, rng: np.random.RandomState
) -> List[Tuple[int, int]]:
    ir = ir.copy()
    jf = jf.copy()
    rng.shuffle(ir)
    rng.shuffle(jf)
    n = min(len(ir), len(jf), limit if limit > 0 else 10**9)
    return list(zip(ir[:n], jf[:n]))


# ----------------------- 3D viridis plots -----------------------
def save_3x3_surface_grid(vol: np.ndarray, path: Path, title: str):
    v = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    H, W, K, C = v.shape
    Kp = min(3, K)
    Cp = min(3, C)
    X, Y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    fig = plt.figure(figsize=(8, 8))
    idx = 1
    for ci in range(Cp):
        for ki in range(Kp):
            ax = fig.add_subplot(Cp, Kp, idx, projection="3d")
            Z = v[:, :, ki, ci]
            ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
            ax.set_title(f"ch={ci}, k={ki}", fontsize=9)
            ax.view_init(elev=35, azim=-60)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])  # type: ignore
            idx += 1
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ----------------------- Pseudo-label helpers -----------------------
def mean_image_centroids(
    imgs: List[Image.Image], labels: np.ndarray
) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    L = labels.astype(int)
    for c in np.unique(L):
        idx = np.where(L == c)[0]
        if len(idx) == 0:
            continue
        arr = np.stack([np.asarray(imgs[i], np.float32) for i in idx], 0)
        cents[int(c)] = arr.mean(0)
    return cents


def assign_by_mean_image(
    imgs: List[Image.Image], cents: Dict[int, np.ndarray]
) -> np.ndarray:
    classes = np.array(sorted(cents.keys()), dtype=np.int64)
    Cimgs = np.stack([cents[int(c)] for c in classes], 0)  # (C,H,W,3)
    f = np.stack([np.asarray(im, np.float32) for im in imgs], 0)  # (N,H,W,3)
    dif = f[:, None] - Cimgs[None]  # (N,C,H,W,3)
    d2 = np.mean(dif * dif, axis=(2, 3, 4))
    return classes[np.argmin(d2, axis=1)]


def class_centroids_incep(
    feats: np.ndarray, labels: np.ndarray
) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    L = labels.astype(int)
    for c in np.unique(L):
        idx = np.where(L == c)[0]
        if len(idx) == 0:
            continue
        cents[int(c)] = feats[idx].mean(axis=0)
    return cents


def assign_to_centroids_incep(
    feats: np.ndarray, cents: Dict[int, np.ndarray]
) -> np.ndarray:
    classes = np.array(sorted(cents.keys()), dtype=np.int64)
    Cs = np.stack([cents[int(c)] for c in classes], 0)
    x2 = np.sum(feats * feats, axis=1, keepdims=True)
    c2 = np.sum(Cs * Cs, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (feats @ Cs.T)
    return classes[np.argmin(d2, axis=1)]


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_pt", type=str, required=True)
    ap.add_argument("--gen_glob", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=2000)

    # CLI you’re using
    ap.add_argument(
        "--pairing", type=str, default="paired", choices=["class", "random", "paired"]
    )
    ap.add_argument(
        "--render",
        type=str,
        default="midk",
        choices=["avgk", "maxk", "midk"] + [f"slice:k={i}" for i in range(512)],
    )
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # sensible knobs
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit_per_class", type=int, default=1000)
    ap.add_argument("--per_class_fid", action="store_true")
    ap.add_argument(
        "--min_fid_n", type=int, default=200, help="Skip FID/KID if any side < this"
    )
    ap.add_argument(
        "--pseudo_label",
        type=str,
        default="inception",
        choices=["meanrgb", "inception", "none"],
        help="Used only if pairing=class and gen labels are missing.",
    )
    ap.add_argument("--save_3d", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    outdir = Path(args.outdir)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "diagnostics").mkdir(parents=True, exist_ok=True)

    # -------- Real data --------
    pack = load_packed_pt(args.real_pt, require_modules=False)
    if hasattr(pack, "vol"):
        real_vol = np.asarray(pack.vol)  # type: ignore
        real_labels = (
            np.asarray(getattr(pack, "labels", None))
            if hasattr(pack, "labels")
            else None
        )
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

    # -------- Render to RGB (uint8) --------
    real_imgs = [to_pil(render_rgb(v, args.render)) for v in real_vol]
    fake_imgs = [to_pil(render_rgb(v, args.render)) for v in gen_vol]

    grid_image(real_imgs[:64], nrow=8).save(outdir / "figs/real_grid.png")
    grid_image(fake_imgs[:64], nrow=8).save(outdir / "figs/fake_grid.png")

    # -------- Inception feats (for FID/KID or pseudo-label if needed) --------
    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    inc = InceptionFeatures(device=device)
    feats_real = inc(real_imgs, batch_size=args.batch_size)
    feats_fake = inc(fake_imgs, batch_size=args.batch_size)

    # -------- Pseudo-labeling (only if pairing=class and we don't have gen labels saved) --------
    fake_labels = None
    pseudo_mode = None
    if args.pairing == "class":
        if real_labels is None:
            print("[warn] real labels missing; falling back to random pairing.")
            args.pairing = "random"
        else:
            if args.pseudo_label == "meanrgb":
                cents = mean_image_centroids(real_imgs, real_labels)
                fake_labels = assign_by_mean_image(fake_imgs, cents)
                pseudo_mode = "meanrgb"
            elif args.pseudo_label == "inception":
                cents = class_centroids_incep(feats_real, real_labels)
                fake_labels = assign_to_centroids_incep(feats_fake, cents)
                pseudo_mode = "inception"
            else:
                fake_labels = None
                pseudo_mode = "none"
            with open(outdir / "diagnostics/pseudo_label_mode.txt", "w") as f:
                f.write(str(pseudo_mode) + "\n")

    # -------- FID/KID (global) with guards --------
    global_FID = None
    global_KID_mean = None
    global_KID_std = None
    if min(feats_real.shape[0], feats_fake.shape[0]) >= args.min_fid_n:
        global_FID = fid_from_feats(feats_real, feats_fake)
        global_KID_mean, global_KID_std = kid_from_feats(
            feats_real,
            feats_fake,
            num_subsets=10,
            subset_size=1000,
            rng=np.random.RandomState(args.seed),
        )

    # -------- Pair selection for PSNR/SSIM --------
    def choose_pairs_random(
        Nr: int, Nf: int, rng: np.random.RandomState
    ) -> List[Tuple[int, int]]:
        m = min(Nr, Nf)
        ir = rng.choice(Nr, m, replace=False)
        jf = rng.choice(Nf, m, replace=False)
        return list(zip(ir, jf))

    if args.pairing == "paired":
        m = min(len(real_imgs), len(fake_imgs))
        pairs = [(i, i) for i in range(m)]
    elif args.pairing == "random" or (args.pairing == "class" and fake_labels is None):
        if args.pairing == "class" and fake_labels is None:
            print(
                "[warn] pseudo-labels not available; using random pairing for PSNR/SSIM."
            )
        pairs = choose_pairs_random(len(real_imgs), len(fake_imgs), rng)
    else:
        # class pairing
        gr = group_indices(real_labels)  # type: ignore
        gf = group_indices(fake_labels)  # type: ignore
        # confusion counts
        all_keys = sorted(set(gr.keys()) | set(gf.keys()))
        conf = {
            str(c): {
                "real_count": int(len(gr.get(c, []))),
                "fake_count": int(len(gf.get(c, []))),
            }
            for c in all_keys
        }
        with open(outdir / "diagnostics/confusion_counts.json", "w") as f:
            json.dump(conf, f, indent=2)
        # pairs within class
        pairs = []
        for c in sorted(set(gr.keys()) & set(gf.keys())):
            pairs += pairs_within_class(gr[c], gf[c], args.limit_per_class, rng)

    # -------- PSNR/SSIM (2D renders) + 3D PSNR --------
    # choose global 3D data_range from real volumes
    vmin = float(np.percentile(real_vol, 0.1))
    vmax = float(np.percentile(real_vol, 99.9))
    data_range_3d = max(vmax - vmin, 1e-6)

    psnrs, ssims, psnrs3d = [], [], []
    for ir, jf in pairs:
        A = np.array(real_imgs[ir], dtype=np.uint8)
        B = np.array(fake_imgs[jf], dtype=np.uint8)
        psnrs.append(psnr_img(A, B, data_range=255.0))
        ssims.append(ssim_img(A, B))
        psnrs3d.append(psnr_3d(real_vol[ir], gen_vol[jf], data_range=data_range_3d))

    # -------- Histograms --------
    if psnrs:
        plt.figure(figsize=(6, 4))
        plt.hist([x for x in psnrs if np.isfinite(x)], bins=50)
        plt.title(f"PSNR (render={args.render}, pairing={args.pairing})")
        plt.tight_layout()
        plt.savefig(outdir / "figs/psnr_hist.png")
        plt.close()
    if ssims and np.isfinite(np.nanmean(ssims)):
        plt.figure(figsize=(6, 4))
        plt.hist([x for x in ssims if np.isfinite(x)], bins=50)
        plt.title(f"SSIM (render={args.render}, pairing={args.pairing})")
        plt.tight_layout()
        plt.savefig(outdir / "figs/ssim_hist.png")
        plt.close()

    # -------- 3D viridis snapshots --------
    n3d = max(0, int(args.save_3d))
    for i in range(min(n3d, len(real_vol))):
        save_3x3_surface_grid(real_vol[i], outdir / f"figs/real_{i:03d}.png", "real")
    for i in range(min(n3d, len(gen_vol))):
        save_3x3_surface_grid(gen_vol[i], outdir / f"figs/fake_{i:03d}.png", "fake")

    # -------- Save paired grid for quick eyeballing --------
    show_pairs = pairs[:64]
    tiles = []
    for ir, jf in show_pairs:
        A = real_imgs[ir].copy()
        B = fake_imgs[jf].copy()
        w, h = A.size
        canv = Image.new("RGB", (w * 2 + 4, h), (255, 255, 255))
        canv.paste(A, (0, 0))
        canv.paste(B, (w + 4, 0))
        tiles.append(canv)
    if tiles:
        grid_image(tiles, nrow=8).save(outdir / "figs/pairs_grid.png")

    # -------- Write metrics --------
    def _nanmean(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    def _nanstd(x):
        return float(np.nanstd(x)) if len(x) else float("nan")

    per_class: Dict[str, Dict[str, Optional[float]]] = {}
    if (
        args.per_class_fid
        and args.pairing == "class"
        and (fake_labels is not None)
        and (real_labels is not None)
    ):
        gr = group_indices(real_labels)
        gf = group_indices(fake_labels)
        for c in sorted(set(gr.keys()) & set(gf.keys())):
            ir = gr[c]
            jf = gf[c]
            FID_c = None
            KID_m = None
            KID_s = None
            if min(len(ir), len(jf)) >= args.min_fid_n:
                FID_c = fid_from_feats(feats_real[ir], feats_fake[jf])
                KID_m, KID_s = kid_from_feats(
                    feats_real[ir],
                    feats_fake[jf],
                    num_subsets=10,
                    subset_size=min(500, len(ir), len(jf)),
                    rng=np.random.RandomState(args.seed),
                )
            per_class[str(c)] = {
                "real_count": int(len(ir)),
                "fake_count": int(len(jf)),
                "FID": None if FID_c is None else float(FID_c),
                "KID_mean": None if KID_m is None else float(KID_m),
                "KID_std": None if KID_s is None else float(KID_s),
            }

    metrics = {
        "counts": {"real": int(Nr), "fake": int(Nf), "pairs": int(len(pairs))},
        "render_mode": args.render,
        "pairing": args.pairing,
        "limit_per_class": int(args.limit_per_class),
        "PSNR_mean": _nanmean(psnrs),
        "PSNR_std": _nanstd(psnrs),
        "PSNR_3D_mean": _nanmean(psnrs3d),
        "PSNR_3D_std": _nanstd(psnrs3d),
        "SSIM_mean": float(np.nanmean(ssims)) if ssims else float("nan"),
        "SSIM_std": float(np.nanstd(ssims)) if ssims else float("nan"),
        "global_FID": None if global_FID is None else float(global_FID),
        "global_KID_mean": None if global_KID_mean is None else float(global_KID_mean),
        "global_KID_std": None if global_KID_std is None else float(global_KID_std),
        "per_class": per_class,
        "notes": {
            "real_pt": args.real_pt,
            "gen_glob": args.gen_glob,
            "device_used": str(device),
            "min_fid_n": int(args.min_fid_n),
            "pseudo_label": str(pseudo_mode) if args.pairing == "class" else "n/a",
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[done] wrote results to: {outdir}")


if __name__ == "__main__":
    main()
