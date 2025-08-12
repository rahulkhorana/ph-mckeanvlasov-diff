import os

# --- Threading controls before heavy imports ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# IMPORTANT: Disable KeOps's use of OpenMP to prevent multiprocessing crashes on macOS
os.environ.setdefault("KEOPS_OMP", "0")

import uuid
import math
import tempfile
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import multiprocessing as mp
from tqdm import tqdm

import multipers as mpers
from multipers.filtrations import RipsCodensity

# ---------------- Configuration ----------------
TARGET_DIMENSION = 6
TOTAL_POINTS = int(os.getenv("TOTAL_POINTS", "1000"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "9"))  # user asked for 9
CHUNKSIZE = int(os.getenv("CHUNKSIZE", "8"))
MAXTASKSPERCHILD = int(os.getenv("MAXTASKSPERCHILD", "50"))
KEEP_MODULES = os.getenv("KEEP_MODULES", "0") == "1"  # off by default
LANDSCAPE_RES = int(os.getenv("LANDSCAPE_RES", "64"))  # 64x64 default
KS_MAX = int(os.getenv("KS_MAX", "2"))  # ks=0..KS_MAX-1 ; default 2
OUT_DIR = os.getenv("OUT_DIR", os.path.join(os.getcwd(), "tmp_samples"))
FINAL_DATASET = os.getenv("FINAL_DATASET", "unified_topological_data_v6_semifast.pt")

# Ensure deterministic-ish behavior across workers (optional)
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
            sample_bumpy_sphere(n=points_per_sphere)
            if bump
            else sample_param_sphere(n=points_per_sphere)
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


# ---------------- Topological Features ----------------


def extract_np_arrays(data):
    if isinstance(data, np.ndarray):
        yield data
    elif isinstance(data, (list, tuple)):
        for item in data:
            yield from extract_np_arrays(item)


def compute_mods_and_landscapes(X: np.ndarray, expansion_dim: int):
    simplextree = RipsCodensity(
        points=X,
        bandwidth=0.2,
        threshold_radius=0.6,
        kernel="gaussian",
        return_log=False,
    )
    simplextree.expansion(expansion_dim)

    bimod = mpers.module_approximation(simplextree)

    module_tensors = None
    if KEEP_MODULES:
        module_data = bimod.dump()
        # Extract and keep only small slices to avoid explosions
        module_tensors = [
            torch.from_numpy(arr).to(torch.float16)
            for arr in extract_np_arrays(module_data)
        ][:3]

    # Landscapes: keep only necessary degrees and modest resolution
    degrees = list(range(min(expansion_dim + 1, 2)))  # usually H0/H1
    landscapes = []
    for degree in degrees:
        land = bimod.landscapes(
            degree=degree,
            ks=range(KS_MAX),
            plot=False,
            box=bimod.get_box(),
            resolution=(LANDSCAPE_RES, LANDSCAPE_RES),
        )
        landscapes.append(torch.from_numpy(land).to(torch.float16))

    # Ensure fixed stack shape: pad up to 3 degrees with zeros
    while len(landscapes) < 3:
        landscapes.append(
            torch.zeros(KS_MAX, LANDSCAPE_RES, LANDSCAPE_RES, dtype=torch.float16)
        )

    all_lands = torch.stack(landscapes, dim=0).to(torch.float16)
    return module_tensors, all_lands


# ---------------- Worker ----------------


def generate_datapoint(args):
    specific_label, numerical_label = args
    n_points = 100

    # Generate point cloud
    if specific_label == "klein_hybrid":
        X = sample_hybrid_klein_bottle(
            n=n_points,
            scale=np.random.uniform(1, 2),
            hybrid_strength=np.random.uniform(1, 2),
            noise_amp=np.random.uniform(0.1, 0.5),
        )
    elif specific_label == "klein_twist":
        X = sample_twist_klein_bottle(
            n=n_points,
            scale=np.random.uniform(0.5, 2),
            twist=np.random.uniform(0.5, 2),
            warp_freq=np.random.uniform(0.5, 2),
            warp_amp=np.random.uniform(0.5, 2),
        )
    elif specific_label == "klein_standard":
        X = sample_klein_bottle(n=n_points, scaled=np.random.uniform(0.5, 2))
    elif specific_label == "sphere_parametric":
        X = sample_param_sphere(n=n_points, radius=np.random.uniform(0.1, 10))
    elif specific_label == "sphere_bumpy":
        X = sample_bumpy_sphere(
            n=n_points,
            bump_freq=np.random.uniform(0.1, 10),
            bump_amp=np.random.uniform(0.1, 1),
        )
    elif specific_label == "sphere_pack_smooth":
        X = sample_sphere_pack(
            n_spheres=3,
            points_per_sphere=n_points // 3 + 1,
            spread=np.random.uniform(0.1, 10),
            bump=False,
        )
    elif specific_label == "sphere_pack_bumpy":
        X = sample_sphere_pack(
            n_spheres=3,
            points_per_sphere=n_points // 3 + 1,
            spread=np.random.uniform(0.1, 10),
            bump=True,
        )
    elif specific_label == "torus_3d":
        X = sample_3_torus(
            n=n_points,
            r1=np.random.uniform(0.1, 2),
            r2=np.random.uniform(0.1, 2),
            r3=np.random.uniform(0.1, 2),
            noise_std=np.random.uniform(0.1, 1),
        )
    else:
        raise ValueError(f"Unknown label: {specific_label}")

    # Pad to TARGET_DIMENSION
    current_dim = X.shape[1]
    if current_dim < TARGET_DIMENSION:
        padding = np.zeros((X.shape[0], TARGET_DIMENSION - current_dim))
        X = np.hstack([X, padding])

    # Normalize point cloud
    max_abs = np.max(np.abs(X))
    if max_abs > 0:
        X = X / max_abs

    pc = torch.from_numpy(X).to(torch.float16)

    # Expansion dimension
    expansion_dim = 3 if "torus" in specific_label else 2

    modules, landscapes = compute_mods_and_landscapes(X, expansion_dim)

    # Persist to disk and return only metadata
    os.makedirs(OUT_DIR, exist_ok=True)
    uid = str(uuid.uuid4())
    path = os.path.join(OUT_DIR, f"sample_{uid}.pt")
    torch.save(
        {
            "pc": pc,
            "y": int(numerical_label),
            "landscapes": landscapes,
            "modules": modules if KEEP_MODULES else None,
        },
        path,
    )

    return {"path": path, "y": int(numerical_label)}


# ---------------- Dataset ----------------


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


def main():
    torch.set_num_threads(1)  # keep per-process BLAS single-threaded

    print(
        f"\nStarting data generation for {TOTAL_POINTS} points using {NUM_WORKERS} workers."
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

    # Write lightweight index
    index = {"paths": saved_paths, "labels": saved_labels, "label_map": label_map}
    index_file = os.path.join(os.getcwd(), "unified_topological_index.pt")
    torch.save(index, index_file)

    # Also optionally pack all items into one file without loading everything at once
    # (still on-disk streaming)
    print("Packing consolidated dataset file (streaming)...")
    ds = OnDiskTopologicalDataset(index_file)

    # We will not materialize everything; we just keep paths+labels+map.
    torch.save(
        {
            "index_file": index_file,
            "label_map": label_map,
            "paths": saved_paths,
            "labels": saved_labels,
        },
        FINAL_DATASET,
    )

    print(f"\nWrote index → {index_file}")
    print(f"Wrote consolidated metadata → {FINAL_DATASET}")
    print(f"Total items: {len(saved_paths)}")

    # Quick sanity peek
    if saved_paths:
        sample = torch.load(saved_paths[0], map_location="cpu")
        print(
            f"First item - Object shape: {tuple(sample['pc'].shape)}, Label: {sample['y']} ({label_map[sample['y']]})"
        )


if __name__ == "__main__":
    # macOS-safe start method
    mp.set_start_method("spawn", force=True)
    main()
