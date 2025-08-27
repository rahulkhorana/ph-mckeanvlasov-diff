# eval_results.py — robust eval with LPIPS, FID, L2, and Wasserstein Distance + Baselines
import argparse, json, math
from pathlib import Path
from glob import glob
from typing import Tuple, List, Optional, Dict
from collections import defaultdict

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
    from scipy.stats import wasserstein_distance, wilcoxon

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import lpips

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

from dataloader import load_packed_pt


# ----------------------- Rendering / Plotting Helpers -----------------------
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
def landscape_l2_distance(vol_real: np.ndarray, vol_fake: np.ndarray) -> float:
    real = np.nan_to_num(vol_real)
    fake = np.nan_to_num(vol_fake)
    return float(np.linalg.norm(real - fake))


def landscape_wasserstein_distance(vol_real: np.ndarray, vol_fake: np.ndarray) -> float:
    """Calculates the 1-Wasserstein distance between the flattened landscape distributions."""
    if not HAS_SCIPY:
        return np.nan
    real = np.nan_to_num(vol_real).ravel()
    fake = np.nan_to_num(vol_fake).ravel()
    return float(wasserstein_distance(real, fake))  # type: ignore


def get_real_vs_real_baseline_scores_by_class(
    real_vol: np.ndarray,
    real_labels: np.ndarray,
    metric_fn,
    max_pairs_per_class: int = 1000,
) -> Dict[int, List[float]]:
    """Calculates all real-vs-real distance scores, grouped by class."""
    scores_by_class = defaultdict(list)
    if real_labels is None:
        return scores_by_class
    print(f"Calculating real-vs-real baseline scores for {metric_fn.__name__}...")
    grouped_indices = group_indices(real_labels)
    rng = np.random.RandomState(42)
    for class_label, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        pairs = [
            (indices[i], indices[j])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]
        if len(pairs) > max_pairs_per_class:
            rng.shuffle(pairs)
            pairs = pairs[:max_pairs_per_class]
        for idx1, idx2 in pairs:
            dist = metric_fn(real_vol[idx1], real_vol[idx2])
            scores_by_class[int(class_label)].append(dist)
    return dict(scores_by_class)


def perform_nonparametric_equivalence_tost(
    gen_dists_by_class: Dict[int, List[float]],
    baseline_dists_by_class: Dict[int, List[float]],
    margin_ratio: float,
    alpha: float = 0.05,
) -> Dict:
    """
    Performs a non-parametric paired samples TOST using Wilcoxon signed-rank tests.
    Null Hypothesis (H0): The difference is large (NOT equivalent).
    Alternative Hypothesis (Ha): The difference is small (IS equivalent).
    """
    if not HAS_SCIPY:
        return {"notes": "SciPy not found, skipping TOST."}

    gen_means, baseline_means = [], []
    common_classes = sorted(
        set(gen_dists_by_class.keys()) & set(baseline_dists_by_class.keys())
    )

    for c in common_classes:
        if gen_dists_by_class.get(c) and baseline_dists_by_class.get(c):
            gen_means.append(np.mean(gen_dists_by_class[c]))
            baseline_means.append(np.mean(baseline_dists_by_class[c]))

    if len(gen_means) < 8:  # Wilcoxon is less reliable with very few pairs
        return {"notes": f"Not enough paired classes ({len(gen_means)}) for TOST."}

    gen_means = np.array(gen_means)
    baseline_means = np.array(baseline_means)

    mean_baseline = np.mean(baseline_means)
    # The margin is the maximum allowed difference between gen_mean and baseline_mean
    upper_bound_delta = mean_baseline * (margin_ratio - 1.0)
    lower_bound_delta = -upper_bound_delta

    diffs = gen_means - baseline_means
    median_diff = np.median(diffs)

    # H0_upper: median_diff >= upper_bound_delta -> Ha_upper: median_diff < upper_bound_delta
    # We test if (diffs - upper_bound_delta) is significantly less than 0
    try:
        _, p_upper = wilcoxon(diffs - upper_bound_delta, alternative="less")  # type: ignore
    except ValueError:
        p_upper = 1.0  # Cannot reject null if all diffs are the same

    # H0_lower: median_diff <= lower_bound_delta -> Ha_lower: median_diff > lower_bound_delta
    # We test if (diffs - lower_bound_delta) is significantly greater than 0
    try:
        _, p_lower = wilcoxon(diffs - lower_bound_delta, alternative="greater")  # type: ignore
    except ValueError:
        p_lower = 1.0

    tost_p_value = max(p_lower, p_upper)  # type: ignore
    is_equivalent = tost_p_value < alpha

    return {
        "test_type": "Non-Parametric Paired Equivalence Test (Wilcoxon-TOST)",
        "null_hypothesis": "Difference is large (not equivalent)",
        "alternative_hypothesis": "Difference is small (equivalent)",
        "num_classes_paired": len(diffs),
        "equivalence_margin_ratio": margin_ratio,
        "equivalence_margin_abs_diff": upper_bound_delta,
        "median_difference": median_diff,
        "tost_p_value": tost_p_value,
        "is_equivalent": bool(is_equivalent),
    }


def psnr_img(a, b, data_range=255.0):
    return 20 * math.log10(data_range) - 10 * math.log10(np.mean((a - b) ** 2))


def ssim_img(a, b):
    return (
        float(ssim_fn(a, b, channel_axis=-1, data_range=255.0))  # type: ignore
        if HAS_SKIMAGE
        else np.nan
    )


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
    def __init__(self, device):
        if not HAS_LPIPS:
            raise ImportError("LPIPS metric requires 'pip install lpips'")
        self.model = lpips.LPIPS(net="alex").to(device)  # type: ignore
        self.model.eval()
        self.device = device
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
        return self.model(img1_t, img2_t).item()


def fid_from_feats(fr, ff):
    mu_r, Cr = np.mean(fr, 0), np.cov(fr, rowvar=False)
    mu_f, Cf = np.mean(ff, 0), np.cov(ff, rowvar=False)
    d = np.sum((mu_r - mu_f) ** 2)
    Csr = scipy_sqrtm(Cr @ Cf).real if HAS_SCIPY else np.zeros_like(Cr)  # type: ignore
    return max(0, d + np.trace(Cr + Cf - 2 * Csr))


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
    ap.add_argument("--save_3d", type=int, default=40)
    ap.add_argument(
        "--eval_mode", type=str, default="slice", choices=["render", "slice"]
    )
    ap.add_argument(
        "--equivalence_margin_wass",
        type=float,
        default=2.5,
        help="Margin for TOST",
    )
    ap.add_argument(
        "--equivalence_margin_l2",
        type=float,
        default=2,
        help="Margin for TOST",
    )
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

    baseline_l2_by_class = get_real_vs_real_baseline_scores_by_class(
        real_vol, real_labels, landscape_l2_distance
    )
    baseline_w_by_class = get_real_vs_real_baseline_scores_by_class(
        real_vol, real_labels, landscape_wasserstein_distance
    )

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
    lpips_metric = LPIPSMetric(device) if HAS_LPIPS else None

    global_fid = fid_from_feats(feats_real, feats_fake)

    rng = np.random.RandomState(123)
    if args.pairing == "paired":
        pairs = list(zip(range(len(real_vol)), range(len(gen_vol))))
    elif args.pairing == "class" and real_labels is not None and gen_labels is not None:
        gr, gf = group_indices(real_labels), group_indices(gen_labels)
        pairs = []
        for c in sorted(set(gr.keys()) & set(gf.keys())):
            ir, jf = gr[c], gf[c]
            rng.shuffle(ir)
            rng.shuffle(jf)
            min_len = min(len(ir), len(jf))
            pairs.extend(zip(ir[:min_len], jf[:min_len]))
    else:
        m = min(len(real_vol), len(gen_vol))
        pairs = list(
            zip(rng.choice(len(real_vol), m, False), rng.choice(len(gen_vol), m, False))
        )

    ssims, lpips_scores = [], []
    gen_l2_by_class = defaultdict(list)
    gen_w_by_class = defaultdict(list)

    for ir, jf in pairs:
        class_label = int(real_labels[ir])
        real_v, fake_v = real_vol[ir], gen_vol[jf]

        gen_l2_by_class[class_label].append(landscape_l2_distance(real_v, fake_v))
        gen_w_by_class[class_label].append(
            landscape_wasserstein_distance(real_v, fake_v)
        )

        if args.eval_mode == "slice":
            K = real_v.shape[2]
            slice_lpips, slice_ssims = [], []
            for k in range(K):
                real_slice_img = to_pil(
                    render_rgb(real_v, f"slice:k={k}", args.render_norm, vmin_vmax)
                )
                fake_slice_img = to_pil(
                    render_rgb(fake_v, f"slice:k={k}", args.render_norm, vmin_vmax)
                )
                if lpips_metric:
                    slice_lpips.append(
                        lpips_metric.calculate(real_slice_img, fake_slice_img)
                    )
                if HAS_SKIMAGE:
                    slice_ssims.append(
                        ssim_img(np.array(real_slice_img), np.array(fake_slice_img))
                    )
            if slice_lpips:
                lpips_scores.append(min(slice_lpips))
            if slice_ssims:
                ssims.append(max(slice_ssims))
        else:  # 'render' mode
            if lpips_metric:
                lpips_scores.append(
                    lpips_metric.calculate(real_imgs[ir], fake_imgs[jf])
                )
            if HAS_SKIMAGE:
                ssims.append(ssim_img(np.array(real_imgs[ir]), np.array(fake_imgs[jf])))

    n3d = max(0, int(args.save_3d))
    for i in range(min(n3d, len(real_vol))):
        plot_landscape_grid(
            real_vol[i], outdir / f"landscapes/real_{i:03d}.png", f"Real {i}"
        )
    for i in range(min(n3d, len(gen_vol))):
        plot_landscape_grid(
            gen_vol[i], outdir / f"landscapes/fake_{i:03d}.png", f"Fake {i}"
        )

    # --- Non-Parametric Equivalence Testing (TOST) Block ---
    stats_results = {}
    stats_results["L2_Distance_TOST"] = perform_nonparametric_equivalence_tost(
        dict(gen_l2_by_class), baseline_l2_by_class, args.equivalence_margin_l2
    )
    stats_results["Wasserstein_TOST"] = perform_nonparametric_equivalence_tost(
        dict(gen_w_by_class), baseline_w_by_class, args.equivalence_margin_wass
    )

    # --- Aggregate stats for reporting ---
    all_l2_gen = [item for sublist in gen_l2_by_class.values() for item in sublist]
    all_w_gen = [item for sublist in gen_w_by_class.values() for item in sublist]
    all_l2_base = [
        item for sublist in baseline_l2_by_class.values() for item in sublist
    ]
    all_w_base = [item for sublist in baseline_w_by_class.values() for item in sublist]

    metrics = {
        "counts": {"real": len(real_vol), "fake": len(gen_vol), "pairs": len(pairs)},
        "eval_mode": args.eval_mode,
        "Landscape_L2_Distance_mean": (
            float(np.mean(all_l2_gen)) if all_l2_gen else None
        ),
        "Landscape_L2_Distance_std": float(np.std(all_l2_gen)) if all_l2_gen else None,
        "Real_vs_Real_L2_Baseline_mean": (
            float(np.mean(all_l2_base)) if all_l2_base else None
        ),
        "Real_vs_Real_L2_Baseline_std": (
            float(np.std(all_l2_base)) if all_l2_base else None
        ),
        "Wasserstein_Distance_mean": float(np.mean(all_w_gen)) if all_w_gen else None,
        "Wasserstein_Distance_std": float(np.std(all_w_gen)) if all_w_gen else None,
        "Real_vs_Real_Wasserstein_Baseline_mean": (
            float(np.mean(all_w_base)) if all_w_base else None
        ),
        "Real_vs_Real_Wasserstein_Baseline_std": (
            float(np.std(all_w_base)) if all_w_base else None
        ),
        "SSIM_mean": float(np.nanmean(ssims)) if ssims else None,
        "SSIM_std": float(np.nanstd(ssims)) if ssims else None,
        "LPIPS_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
        "LPIPS_std": float(np.std(lpips_scores)) if lpips_scores else None,
        "global_FID": float(global_fid) if global_fid is not None else None,
    }
    metrics.update(stats_results)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[done] wrote results to: {outdir}")


if __name__ == "__main__":
    main()
