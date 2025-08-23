# eval_results.py — robust eval with explicit normalization, guarded FID/KID, pairs save, 3D plots
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

from dataloader import load_packed_pt


# ----------------------- Rendering helpers -----------------------
def _select_image_from_vol(vol: np.ndarray, mode: str) -> np.ndarray:
    """Return float32 (H,W,C) BEFORE any scaling; handles channel padding."""
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
    return img.astype(np.float32)


def render_rgb(
    vol: np.ndarray,
    mode: str = "midk",
    norm: str = "global",  # "global" | "perimage" | "none"
    vmin_vmax: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Render (H,W,K,C) -> uint8 (H,W,3).

    - norm="global": expects vmin_vmax (lo,hi) computed from REAL renders; applied to both real & fake.
    - norm="perimage": robust percentiles computed per image.
    - norm="none": no rescale; just clip to [0,1].
    """
    img = _select_image_from_vol(vol, mode)

    if norm == "none":
        img01 = np.clip(img, 0.0, 1.0)
        return (img01 * 255.0 + 0.5).astype(np.uint8)

    if norm == "global":
        if vmin_vmax is None:
            raise ValueError("render_rgb(norm='global') requires vmin_vmax=(lo,hi).")
        lo, hi = float(vmin_vmax[0]), float(vmin_vmax[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # fallback to per-image robust range
            lo = float(np.percentile(img, 1.0))
            hi = float(np.percentile(img, 99.0)) + 1e-6
    elif norm == "perimage":
        lo = float(np.percentile(img, 1.0))
        hi = float(np.percentile(img, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(img.min()), float(img.max()) + 1e-6
    else:
        raise ValueError(f"Unknown render norm: {norm}")

    img01 = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def compute_global_range_from_real(
    real_vol: np.ndarray, render_mode: str, pct_lo: float = 1.0, pct_hi: float = 99.0
) -> Tuple[float, float]:
    """
    Compute (lo,hi) in IMAGE space for the chosen render_mode using robust percentiles
    over ALL real renders (no scaling applied yet).
    """
    # To avoid huge memory spikes, stream through volumes
    lows, highs = [], []
    for v in real_vol:
        img = _select_image_from_vol(v, render_mode)
        lo = np.percentile(img, pct_lo)
        hi = np.percentile(img, pct_hi)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            lows.append(lo)
            highs.append(hi)
    if not lows:
        # ultimate fallback from raw tensor
        lo = float(np.percentile(real_vol, pct_lo))
        hi = float(np.percentile(real_vol, pct_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(real_vol.min()), float(real_vol.max()) + 1e-6
        return lo, hi

    # Use medians of per-image percentiles (robust across outliers)
    lo = float(np.median(np.array(lows)))
    hi = float(np.median(np.array(highs)))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


# ----------------------- PSNR / SSIM -----------------------
def psnr_img(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(float(mse))


def psnr_3d(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    a = np.nan_to_num(a, nan=0.0).astype(np.float32)
    b = np.nan_to_num(b, nan=0.0).astype(np.float32)
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
        try:
            from torchvision.models import Inception_V3_Weights

            weights = Inception_V3_Weights.IMAGENET1K_V1
            model = inception_v3(weights=weights, aux_logits=True)
            self.tfm = weights.transforms()
        except Exception:
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
            f = self.extractor(t)["feat"].squeeze(-1).squeeze(-1)  # (N,2048)
            xs.append(f.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(xs, 0) if xs else np.empty((0, 2048), np.float64)


def _cov_mean(
    X: np.ndarray, shrink: float = 0.1, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
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
    if m < 20:
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
    return Image.fromarray(img_uint8_hw3)


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
    v = np.nan_to_num(vol, nan=0.0).astype(np.float32)
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


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_pt", type=str, required=True)
    ap.add_argument("--gen_glob", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=2000)

    ap.add_argument(
        "--pairing", type=str, default="paired", choices=["class", "random", "paired"]
    )
    ap.add_argument(
        "--render",
        type=str,
        default="midk",
        choices=["avgk", "maxk", "midk"] + [f"slice:k={i}" for i in range(512)],
    )
    ap.add_argument(
        "--render_norm",
        type=str,
        default="global",
        choices=["global", "perimage", "none"],
        help="global: one (vmin,vmax) from real renders; perimage: per-image; none: clip to [0,1].",
    )
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit_per_class", type=int, default=1000)
    ap.add_argument("--per_class_fid", action="store_true")
    ap.add_argument("--min_fid_n", type=int, default=10)
    ap.add_argument(
        "--pseudo_label",
        type=str,
        default="inception",
        choices=["meanrgb", "inception", "none"],
    )
    ap.add_argument("--save_3d", type=int, default=40)
    ap.add_argument("--save_pairs_grid", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    outdir = Path(args.outdir)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "diagnostics").mkdir(parents=True, exist_ok=True)

    # -------- Real data --------
    pack = load_packed_pt(args.real_pt, require_modules=False)
    if isinstance(pack, dict):
        real_vol = np.asarray(pack["vol"])
        real_labels = np.asarray(pack.get("labels")) if "labels" in pack else None
    else:
        real_vol = np.asarray(pack.vol)  # type: ignore
        real_labels = (
            np.asarray(getattr(pack, "labels", None))
            if hasattr(pack, "labels")
            else None
        )

    Nr_full = real_vol.shape[0]
    Nr = min(args.max_samples, Nr_full)
    real_vol = real_vol[:Nr]
    if real_labels is not None:
        real_labels = real_labels[:Nr]

    # -------- Generated data (.npz or .npy) --------
    gen_files = sorted(glob(args.gen_glob))
    if len(gen_files) == 0:
        raise ValueError(f"No generated files matched: {args.gen_glob}")

    gens = []
    gen_labels = None
    for f in gen_files:
        if f.endswith(".npz"):
            z = np.load(f)
            arr = z["samples"]
            gens.append(arr)
            if "labels" in z.files:
                gen_labels = (
                    z["labels"]
                    if gen_labels is None
                    else np.concatenate([gen_labels, z["labels"]], 0)
                )
        else:
            arr = np.load(f)
            if arr.ndim != 5:
                raise ValueError(f"Expected (B,H,W,K,C) in {f}, got {arr.shape}")
            gens.append(arr)

    gen_vol = np.concatenate(gens, axis=0)
    Nf_full = gen_vol.shape[0]
    Nf = min(args.max_samples, Nf_full)
    gen_vol = gen_vol[:Nf]
    if gen_labels is not None:
        gen_labels = np.asarray(gen_labels)[:Nf]

    # -------- Normalization stats (global -> image-space robust range from REAL renders) --------
    vmin_vmax_global: Optional[Tuple[float, float]] = None
    if args.render_norm == "global":
        vmin_vmax_global = compute_global_range_from_real(
            real_vol, args.render, 1.0, 99.0
        )

    # -------- Render to RGB --------
    real_imgs = [
        to_pil(
            render_rgb(
                v, args.render, norm=args.render_norm, vmin_vmax=vmin_vmax_global
            )
        )
        for v in real_vol
    ]
    fake_imgs = [
        to_pil(
            render_rgb(
                v, args.render, norm=args.render_norm, vmin_vmax=vmin_vmax_global
            )
        )
        for v in gen_vol
    ]

    grid_image(real_imgs[:64], nrow=8).save(outdir / "figs/real_grid.png")
    grid_image(fake_imgs[:64], nrow=8).save(outdir / "figs/fake_grid.png")

    # -------- Inception feats --------
    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    inc = InceptionFeatures(device=device)
    feats_real = inc(real_imgs, batch_size=args.batch_size)
    feats_fake = inc(fake_imgs, batch_size=args.batch_size)

    # -------- Labels for pairing --------
    fake_labels = None
    pseudo_mode: Optional[str] = None
    if args.pairing == "class":
        if real_labels is None and gen_labels is None:
            print("[warn] no labels; falling back to random.")
            args.pairing = "random"
        else:
            if gen_labels is not None:
                fake_labels = np.asarray(gen_labels)
                pseudo_mode = "provided"
            elif real_labels is not None:
                if args.pseudo_label == "meanrgb":
                    cents: Dict[int, np.ndarray] = {}
                    L = real_labels.astype(int)
                    for c in np.unique(L):
                        idx = np.where(L == c)[0]
                        arr = np.stack(
                            [np.asarray(real_imgs[i], np.float32) for i in idx], 0
                        )
                        cents[int(c)] = arr.mean(0)
                    classes = np.array(sorted(cents.keys()), dtype=np.int64)
                    Cimgs = np.stack([cents[int(c)] for c in classes], 0)
                    f = np.stack([np.asarray(im, np.float32) for im in fake_imgs], 0)
                    dif = f[:, None] - Cimgs[None]
                    d2 = np.mean(dif * dif, axis=(2, 3, 4))
                    fake_labels = classes[np.argmin(d2, axis=1)]
                    pseudo_mode = "meanrgb"
                elif args.pseudo_label == "inception":
                    cents: Dict[int, np.ndarray] = {}
                    L = real_labels.astype(int)
                    for c in np.unique(L):
                        idx = np.where(L == c)[0]
                        cents[int(c)] = feats_real[idx].mean(axis=0)
                    classes = np.array(sorted(cents.keys()), dtype=np.int64)
                    Cs = np.stack([cents[int(c)] for c in classes], 0)
                    x2 = np.sum(feats_fake * feats_fake, axis=1, keepdims=True)
                    c2 = np.sum(Cs * Cs, axis=1, keepdims=True).T
                    d2 = x2 + c2 - 2.0 * (feats_fake @ Cs.T)
                    fake_labels = classes[np.argmin(d2, axis=1)]
                    pseudo_mode = "inception"

    # -------- FID/KID (global, guarded) --------
    global_FID: Optional[float] = None
    global_KID_mean: Optional[float] = None
    global_KID_std: Optional[float] = None
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
    rng_np = np.random.RandomState(args.seed)

    def choose_pairs_random(Nr: int, Nf: int) -> List[Tuple[int, int]]:
        m = min(Nr, Nf, 1024)  # cap so we don’t go wild
        ir = rng_np.choice(Nr, m, replace=False)
        jf = rng_np.choice(Nf, m, replace=False)
        return list(zip(ir, jf))

    if args.pairing == "paired":
        m = min(len(real_imgs), len(fake_imgs))
        pairs = [(i, i) for i in range(m)]
    elif args.pairing == "random" or (args.pairing == "class" and fake_labels is None):
        if args.pairing == "class" and fake_labels is None:
            print(
                "[warn] pseudo-labels not available; using random pairing for PSNR/SSIM."
            )
        pairs = choose_pairs_random(len(real_imgs), len(fake_imgs))
    else:
        gr = group_indices(real_labels)  # type: ignore
        gf = group_indices(fake_labels)  # type: ignore
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
        pairs = []
        for c in sorted(set(gr.keys()) & set(gf.keys())):
            pairs += pairs_within_class(gr[c], gf[c], args.limit_per_class, rng_np)

    # Save the actual index pairs for transparency
    pairs_json = [{"real_idx": int(ir), "fake_idx": int(jf)} for ir, jf in pairs]
    with open(outdir / "diagnostics/pairs.json", "w") as f:
        json.dump({"pairs": pairs_json}, f, indent=2)

    # Optional: save a grid of paired comparisons
    if args.save_pairs_grid and len(pairs) > 0:
        tiles = []
        for ir, jf in pairs[:64]:
            A = real_imgs[ir].copy()
            B = fake_imgs[jf].copy()
            w, h = A.size
            canv = Image.new("RGB", (w * 2 + 4, h), (255, 255, 255))
            canv.paste(A, (0, 0))
            canv.paste(B, (w + 4, 0))
            tiles.append(canv)
        grid_image(tiles, nrow=8).save(outdir / "figs/pairs_grid.png")

    # -------- PSNR/SSIM + 3D PSNR --------
    vmin = float(np.percentile(real_vol, 0.1))
    vmax = float(np.percentile(real_vol, 99.9))
    data_range_3d = max(vmax - vmin, 1e-6)
    psnrs, ssims, psnrs3d = [], [], []
    for ir, jf in pairs:
        A = np.array(real_imgs[ir], dtype=np.uint8)
        B = np.array(fake_imgs[jf], dtype=np.uint8)
        psnrs.append(psnr_img(A, B, 255.0))
        ssims.append(ssim_img(A, B))
        psnrs3d.append(psnr_3d(real_vol[ir], gen_vol[jf], data_range=data_range_3d))

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

    # Diagnostic: flatness vs SSIM
    fake_stds = [float(np.std(np.asarray(im, np.float32))) for im in fake_imgs[:256]]
    flat_ratio = float(np.mean([s < 1.0 for s in fake_stds])) if fake_stds else 0.0
    if np.isfinite(np.nanmean(ssims)) and flat_ratio > 0.5 and np.nanmean(ssims) > 0.6:
        print(
            "[warn] Many fake renders are near-constant but SSIM is high. "
            "Re-check --render_norm and pairing."
        )
    with open(outdir / "diagnostics/flat_stats.json", "w") as f:
        json.dump(
            {
                "fake_std_mean": float(np.mean(fake_stds) if fake_stds else 0.0),
                "fake_std_median": float(np.median(fake_stds) if fake_stds else 0.0),
                "flat_ratio_<1.0": flat_ratio,
            },
            f,
            indent=2,
        )

    # -------- 3D viridis snapshots --------
    n3d = max(0, int(args.save_3d))
    for i in range(min(n3d, len(real_vol))):
        save_3x3_surface_grid(real_vol[i], outdir / f"figs/real_{i:03d}.png", "real")
    for i in range(min(n3d, len(gen_vol))):
        save_3x3_surface_grid(gen_vol[i], outdir / f"figs/fake_{i:03d}.png", "fake")

    # -------- Write metrics --------
    def _nanmean(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    def _nanstd(x):
        return float(np.nanstd(x)) if len(x) else float("nan")

    metrics = {
        "counts": {
            "real": int(real_vol.shape[0]),
            "fake": int(gen_vol.shape[0]),
            "pairs": int(len(pairs)),
        },
        "render_mode": args.render,
        "render_norm": args.render_norm,
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
        "per_class": {},
        "notes": {
            "real_pt": args.real_pt,
            "gen_glob": args.gen_glob,
            "device_used": str(device),
            "min_fid_n": int(args.min_fid_n),
            "has_gen_labels": bool(gen_labels is not None),
            "pseudo_label": (
                "provided"
                if gen_labels is not None
                else (
                    pseudo_mode
                    if pseudo_mode is not None
                    else (args.pseudo_label if args.pairing == "class" else "n/a")
                )
            ),
        },
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[done] wrote results to: {outdir}")


if __name__ == "__main__":
    main()
