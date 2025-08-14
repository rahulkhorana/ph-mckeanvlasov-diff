# make_synth_data.py
import os
import platform

# --- Threading & KeOps before heavy imports ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# Disable KeOps OpenMP to avoid macOS + multiprocessing issues
os.environ.setdefault("KEOPS_OMP", "0")

# Apple Silicon / Homebrew libomp wiring
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("HOMEBREW_PREFIX", "/opt/homebrew")
    hb = os.environ["HOMEBREW_PREFIX"]
    os.environ.setdefault("CC", "clang")
    os.environ.setdefault("CXX", "clang++")
    cflags = f"-O3 -fopenmp -I{hb}/opt/libomp/include"
    ldflags = f"-L{hb}/opt/libomp/lib -lomp"
    os.environ["CFLAGS"] = f"{cflags} " + os.environ.get("CFLAGS", "")
    os.environ["CXXFLAGS"] = f"{cflags} " + os.environ.get("CXXFLAGS", "")
    os.environ["LDFLAGS"] = f"{ldflags} " + os.environ.get("LDFLAGS", "")
    os.environ["DYLD_LIBRARY_PATH"] = f"{hb}/opt/libomp/lib:" + os.environ.get(
        "DYLD_LIBRARY_PATH", ""
    )

import uuid
import shutil
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import multiprocessing as mp
from tqdm import tqdm

import multipers as mpers
from multipers.filtrations import RipsCodensity

# Force KeOps to use a shared cache folder
os.environ.setdefault("PYKEOPS_CACHE_DIR", os.path.join(os.getcwd(), ".keops_cache"))
os.makedirs(os.environ["PYKEOPS_CACHE_DIR"], exist_ok=True)

# ---------------- Configuration ----------------
TARGET_DIMENSION = 6
TOTAL_POINTS = int(os.getenv("TOTAL_POINTS", "1000"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "9"))
CHUNKSIZE = int(os.getenv("CHUNKSIZE", "8"))
MAXTASKSPERCHILD = int(os.getenv("MAXTASKSPERCHILD", "50"))

KEEP_MODULES = os.getenv("KEEP_MODULES", "1") == "1"  # default ON
LANDSCAPE_RES = int(os.getenv("LANDSCAPE_RES", "128"))
KS_MAX = int(os.getenv("KS_MAX", "3"))
POINTS_PER_CLOUD = int(os.getenv("POINTS_PER_CLOUD", "100"))

OUT_DIR = os.getenv("OUT_DIR", os.path.join(os.getcwd(), "tmp_samples"))
FINAL_DATASET = os.getenv("FINAL_DATASET", "unified_topological_data_v6_semifast.pt")

SEED = os.getenv("SEED")
if SEED is not None:
    np.random.seed(int(SEED))
    torch.manual_seed(int(SEED))


# ---------------- Manifold Sampling ----------------
def sample_hybrid_klein_bottle(n=500, scale=1.0, hybrid_strength=0.3, noise_amp=0.1):
    u = np.random.uniform(0, 2 * np.pi, n)
    v = np.random.uniform(0, 2 * np.pi, n)
    x = np.cos(u) * (1 + 0.5 * np.sin(v)) + hybrid_strength * np.sin(2 * u + v)
    y = np.sin(u) * (1 + 0.5 * np.cos(v)) + hybrid_strength * np.cos(2 * u - v)
    z = 0.5 * np.cos(v) * np.sin(3 * u) + hybrid_strength * np.sin(u + v)
    w = 0.5 * np.sin(v) * np.cos(5 * u) + hybrid_strength * np.cos(2 * u + 2 * v)
    noise = np.random.normal(0, noise_amp, size=(n, 4)) if noise_amp > 0 else 0
    return np.column_stack([x, y, z, w]) * scale + noise


def sample_twist_klein_bottle(n=500, scale=1.0, twist=0.0, warp_freq=0.0, warp_amp=0.0):
    u = np.random.uniform(0, 2 * np.pi, n)
    v = np.random.uniform(0, 2 * np.pi, n)
    v_twisted = v + twist * np.sin(u)
    x = np.cos(u) * (1 + 0.5 * np.sin(v_twisted))
    y = np.sin(u) * (1 + 0.5 * np.sin(v_twisted))
    z = 0.5 * np.cos(v_twisted) * np.cos(u / 2)
    w = 0.5 * np.cos(v_twisted) * np.sin(u / 2)
    if warp_amp > 0 and warp_freq > 0:
        x += warp_amp * np.sin(warp_freq * u)
        y += warp_amp * np.cos(warp_freq * v)
    return np.column_stack([x, y, z, w]) * scale


def sample_klein_bottle(n=500, scaled=0.8):
    u = np.random.uniform(0, 2 * np.pi, n)
    v = np.random.uniform(0, 2 * np.pi, n)
    x = np.cos(u) * (1 + 0.5 * np.sin(v))
    y = np.sin(u) * (1 + 0.5 * np.sin(v))
    z = 0.5 * np.cos(v) * np.cos(u / 2)
    w = 0.5 * np.cos(v) * np.sin(u / 2)
    return np.column_stack([scaled * x, scaled * y, scaled * z, scaled * w])


def sample_param_sphere(n=1000, radius=1.0):
    theta = np.arccos(1 - 2 * np.random.rand(n))
    phi = 2 * np.pi * np.random.rand(n)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z])


def sample_bumpy_sphere(n=1000, base_radius=1.0, bump_freq=5.0, bump_amp=0.2):
    theta = np.arccos(1 - 2 * np.random.rand(n))
    phi = 2 * np.pi * np.random.rand(n)
    r = base_radius + bump_amp * np.sin(bump_freq * theta) * np.cos(bump_freq * phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])


def sample_sphere_pack(n_spheres=5, points_per_sphere=200, spread=3.0, bump=False):
    points = []
    for _ in range(n_spheres):
        center = np.random.uniform(-spread, spread, size=3)
        sphere = (
            sample_bumpy_sphere(points_per_sphere)
            if bump
            else sample_param_sphere(points_per_sphere)
        )
        points.append(sphere + center)
    return np.vstack(points)


def sample_3_torus(n=500, r1=1.0, r2=0.2, r3=1.5, noise_std=0.0):
    t1, t2, t3 = [np.random.uniform(0, 2 * np.pi, n) for _ in range(3)]
    x1 = (r1 + r2 * np.cos(t2)) * np.cos(t1)
    y1 = (r1 + r2 * np.cos(t2)) * np.sin(t1)
    x2 = (r3 + r2 * np.sin(t2)) * np.cos(t3)
    y2 = (r3 + r2 * np.sin(t2)) * np.sin(t3)
    x3 = r2 * np.cos(t2) * np.sin(t3)
    y3 = r2 * np.sin(t2) * np.cos(t3)
    X = np.column_stack([x1, y1, x2, y2, x3, y3])
    if noise_std > 0:
        X += np.random.normal(0, noise_std, size=X.shape)
    return X


# ---------------- Adaptive filtration + features ----------------
def extract_np_arrays(data):
    if isinstance(data, np.ndarray):
        yield data
    elif isinstance(data, (list, tuple)):
        for item in data:
            yield from extract_np_arrays(item)


def _pairwise_dists(X: np.ndarray, k_sample: int = 512) -> np.ndarray:
    n = len(X)
    idx = np.random.choice(n, size=min(k_sample, n), replace=False)
    Y = X[idx]
    D2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    D = np.sqrt(np.maximum(D2, 0.0))
    return D[D > 0].ravel()


def _auto_rips_params(X: np.ndarray) -> tuple[float, float]:
    D = _pairwise_dists(X)
    if D.size == 0:
        return 0.2, 0.6  # fallback
    d50 = float(np.percentile(D, 50.0))
    d90 = float(np.percentile(D, 90.0))
    bandwidth = max(d50, 1e-6)
    threshold = max(d90 * 1.25, bandwidth * 1.5)
    return 0.2, 0.6


def compute_mods_and_landscapes(X: np.ndarray, expansion_dim: int):
    bw, thr = _auto_rips_params(X)

    simplextree = RipsCodensity(
        points=X,
        bandwidth=bw,
        threshold_radius=thr,
        kernel="gaussian",
        return_log=False,
    )
    simplextree.expansion(expansion_dim)

    bimod = mpers.module_approximation(simplextree)

    # Modules: keep fp32
    module_tensors = None
    if KEEP_MODULES:
        module_data = bimod.dump()
        module_tensors = [
            torch.from_numpy(arr).to(torch.float32)
            for arr in extract_np_arrays(module_data)
        ][:3]

    # Landscapes: degrees up to 1; pad to 3; fp32 with per-sample max-norm
    degrees = list(range(min(expansion_dim + 1, 2)))  # H0/H1 at most
    landscapes: List[torch.Tensor] = []
    for degree in degrees:
        land = bimod.landscapes(
            degree=degree,
            ks=range(KS_MAX),
            plot=False,
            box=bimod.get_box(),
            resolution=(LANDSCAPE_RES, LANDSCAPE_RES),
        ).astype(np.float32)
        m = float(np.max(land)) if np.isfinite(np.max(land)) else 0.0
        if m > 1e-8:
            land = land / m
        landscapes.append(torch.from_numpy(land))

    while len(landscapes) < 3:
        landscapes.append(
            torch.zeros(KS_MAX, LANDSCAPE_RES, LANDSCAPE_RES, dtype=torch.float32)
        )

    all_lands = torch.stack(landscapes, dim=0).to(torch.float32)  # (3,KS,H,W)
    return module_tensors, all_lands


# ---------------- Worker ----------------
def generate_datapoint(args):
    specific_label, numerical_label = args
    n_points = POINTS_PER_CLOUD

    # Generate point cloud
    if specific_label == "klein_hybrid":
        X = sample_hybrid_klein_bottle(
            n=n_points,
            scale=np.random.uniform(1, 2),
            hybrid_strength=np.random.uniform(0.7, 1.5),
            noise_amp=np.random.uniform(0.05, 0.25),
        )
    elif specific_label == "klein_twist":
        X = sample_twist_klein_bottle(
            n=n_points,
            scale=np.random.uniform(0.6, 1.6),
            twist=np.random.uniform(0.4, 1.4),
            warp_freq=np.random.uniform(0.4, 1.6),
            warp_amp=np.random.uniform(0.2, 0.8),
        )
    elif specific_label == "klein_standard":
        X = sample_klein_bottle(n=n_points, scaled=np.random.uniform(0.7, 1.5))
    elif specific_label == "sphere_parametric":
        X = sample_param_sphere(n=n_points, radius=np.random.uniform(0.6, 1.6))
    elif specific_label == "sphere_bumpy":
        X = sample_bumpy_sphere(
            n=n_points,
            bump_freq=np.random.uniform(2.0, 8.0),
            bump_amp=np.random.uniform(0.05, 0.35),
        )
    elif specific_label == "sphere_pack_smooth":
        X = sample_sphere_pack(
            n_spheres=3,
            points_per_sphere=n_points // 3 + 1,
            spread=np.random.uniform(1.0, 4.0),
            bump=False,
        )[:n_points]
    elif specific_label == "sphere_pack_bumpy":
        X = sample_sphere_pack(
            n_spheres=3,
            points_per_sphere=n_points // 3 + 1,
            spread=np.random.uniform(1.0, 4.0),
            bump=True,
        )[:n_points]
    elif specific_label == "torus_3d":
        X = sample_3_torus(
            n=n_points,
            r1=np.random.uniform(0.6, 1.5),
            r2=np.random.uniform(0.15, 0.5),
            r3=np.random.uniform(0.8, 1.8),
            noise_std=np.random.uniform(0.0, 0.15),
        )
    else:
        raise ValueError(f"Unknown label: {specific_label}")

    # Pad to TARGET_DIMENSION
    if X.shape[1] < TARGET_DIMENSION:
        padding = np.zeros((X.shape[0], TARGET_DIMENSION - X.shape[1]), dtype=X.dtype)
        X = np.hstack([X, padding])

    # Normalize point cloud to unit max-abs
    max_abs = np.max(np.abs(X))
    if max_abs > 0:
        X = X / max_abs

    pc = torch.from_numpy(X.astype(np.float32))  # (N,6) fp32

    # Expansion dimension
    expansion_dim = 3 if "torus" in specific_label else 2

    modules, landscapes = compute_mods_and_landscapes(X, expansion_dim)

    # Persist to disk and return metadata
    os.makedirs(OUT_DIR, exist_ok=True)
    uid = str(uuid.uuid4())
    path = os.path.join(OUT_DIR, f"sample_{uid}.pt")
    torch.save(
        {
            "pc": pc,  # (n_points, 6) fp32
            "y": int(numerical_label),
            "landscapes": landscapes,  # (3, KS, H, W) fp32
            "modules": modules if KEEP_MODULES else None,  # list[Tensor] fp32
        },
        path,
    )
    return {"path": path, "y": int(numerical_label)}


# ---------------- Dataset (debug helper) ----------------
class OnDiskTopologicalDataset(Dataset):
    def __init__(self, index_file: str):
        info = torch.load(index_file, map_location="cpu")
        self.paths = info["paths"]
        self.labels = info["labels"]
        self.label_map = info["label_map"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rec = torch.load(self.paths[idx], map_location="cpu")
        return rec["pc"], rec["y"], rec["modules"], rec["landscapes"]


# ---------------- Main ----------------
def _prewarm_keops():
    try:
        X = np.random.randn(64, TARGET_DIMENSION).astype(np.float32)
        _ = compute_mods_and_landscapes(X, expansion_dim=2)  # compile once
        print("[Prewarm] KeOps kernels compiled/cached.")
    except Exception as e:
        print(f"[Prewarm] Failed: {e}")


def main():
    torch.set_num_threads(1)
    print(
        f"\nStarting data generation for {TOTAL_POINTS} points using {NUM_WORKERS} workers."
    )
    print(
        f"RES={LANDSCAPE_RES}  KS_MAX={KS_MAX}  Pts/Cloud={POINTS_PER_CLOUD}  KEEP_MODULES={KEEP_MODULES}"
    )

    label_map = {
        0: "klein_hybrid",
        1: "klein_twist",
        2: "klein_standard",
        3: "sphere_parametric",
        4: "sphere_bumpy",
        5: "sphere_pack_smooth",
        6: "sphere_pack_bumpy",
        7: "torus_3d",
    }
    reverse_label_map = {name: num for num, name in label_map.items()}

    all_label_names = list(label_map.values())
    task_labels = np.random.choice(
        all_label_names, size=TOTAL_POINTS, replace=True
    ).tolist()
    tasks = [(label, reverse_label_map[label]) for label in task_labels]

    ctx = mp.get_context("spawn")
    saved_paths, saved_labels = [], []

    _prewarm_keops()

    try:
        with ctx.Pool(processes=NUM_WORKERS, maxtasksperchild=MAXTASKSPERCHILD) as pool:
            for rec in tqdm(
                pool.imap_unordered(generate_datapoint, tasks, chunksize=CHUNKSIZE),
                total=len(tasks),
                desc="Generating Data",
            ):
                saved_paths.append(rec["path"])
                saved_labels.append(rec["y"])
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down workers...")
        raise

    # Lightweight index
    index = {"paths": saved_paths, "labels": saved_labels, "label_map": label_map}
    index_file = os.path.join(os.getcwd(), "unified_topological_index.pt")
    torch.save(index, index_file)

    # -------- Final single-file pack --------
    print("Packing final single .pt (this may take a moment)…")
    pcs, ys, lands, mods = [], [], [], []
    for pth, lab in tqdm(
        zip(saved_paths, saved_labels), total=len(saved_paths), desc="Packing"
    ):
        rec = torch.load(pth, map_location="cpu")
        pcs.append(rec["pc"].to(torch.float32))  # (P,6)
        ys.append(int(rec["y"]))
        lands.append(rec["landscapes"].to(torch.float32))  # (3,KS,H,W)
        if KEEP_MODULES:
            mods.append(rec["modules"])

    pcs = torch.stack(pcs, dim=0)  # (N, P, 6) fp32
    ys = torch.tensor(ys, dtype=torch.long)  # (N,)
    lands = torch.stack(lands, dim=0)  # (N, 3, KS, H, W) fp32

    payload = {
        "pcs": pcs,
        "labels": ys,
        "landscapes": lands,
        "modules": mods if KEEP_MODULES else None,
        "label_map": label_map,
        "meta": {
            "target_dim": TARGET_DIMENSION,
            "ks_max": KS_MAX,
            "landscape_res": LANDSCAPE_RES,
            "keep_modules": KEEP_MODULES,
            "points_per_cloud": POINTS_PER_CLOUD,
        },
    }
    torch.save(payload, FINAL_DATASET)

    # Sanity probe
    L0 = lands[0].float().numpy()  # (3,KS,H,W)
    print(f"Wrote FINAL packed dataset → {FINAL_DATASET}")
    print(
        f"Shapes: pcs={tuple(pcs.shape)}, labels={tuple(ys.shape)}, landscapes={tuple(lands.shape)}"
    )
    print(
        f"[Sanity] sample0 landscape min={float(L0.min()):.4g} max={float(L0.max()):.4g}"
    )

    # -------- Cleanup temp shards --------
    try:
        for p in saved_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(OUT_DIR)
        except OSError:
            shutil.rmtree(OUT_DIR, ignore_errors=True)
        print("Deleted temporary shard files.")
    except Exception as e:
        print(f"[Cleanup] Could not remove temp files: {e}")

    print(f"Total items: {len(saved_paths)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
