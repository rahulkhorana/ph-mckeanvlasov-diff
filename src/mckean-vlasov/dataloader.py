import numpy as np
import torch
from typing import Dict, Any, Iterator, Tuple
from dataclasses import dataclass


def load_packed_pt(path: str, require_modules: bool = True) -> Dict[str, Any]:
    pack = torch.load(path, map_location="cpu")
    pcs: torch.Tensor = pack["pcs"]  # (N, P, 6) float16
    labels: torch.Tensor = pack["labels"].long()  # (N,)
    lands: torch.Tensor = pack["landscapes"]  # (N, 3, KS, H, W) float16
    modules = pack["modules"]  # list length N (ragged)
    meta = pack.get("meta", {})
    ks_max = int(meta.get("ks_max", lands.shape[2]))
    res = int(meta.get("landscape_res", lands.shape[-1]))
    degrees = 3

    if require_modules and (modules is None):
        raise ValueError("Dataset has no 'modules' but require_modules=True.")

    lands_np = lands.numpy().astype(np.float32)  # (N,3,KS,H,W)
    vol = np.transpose(lands_np, (0, 3, 4, 2, 0))  # (N,H,W,KS,3)

    mu = float(vol.mean())
    sigma = float(vol.std() + 1e-6)
    vol_norm = (vol - mu) / sigma

    return {
        "vol": vol_norm,
        "vol_mean": mu,
        "vol_std": sigma,
        "labels": labels.numpy().astype(np.int32),
        "modules": modules,
        "KS": ks_max,
        "degrees": degrees,
        "res": res,
        "label_map": pack.get("label_map", {}),
    }


@dataclass
class PackedDataset:
    vol: np.ndarray
    labels: np.ndarray
    modules: Any
    vol_mean: float
    vol_std: float
    KS: int
    degrees: int
    res: int
    label_map: Dict[int, str]


def describe(pack: Dict[str, Any]) -> str:
    v = pack["vol"]
    return (
        f"N={v.shape[0]}  vol=(N,H,W,K,C)={tuple(v.shape)}  "
        f"KS={pack['KS']}  degrees={pack['degrees']}  res={pack['res']}"
    )


def train_val_split(
    pack: Dict[str, Any], val_frac: float = 0.1, seed: int = 0
) -> Tuple[PackedDataset, PackedDataset]:
    rng = np.random.RandomState(seed)
    N = pack["vol"].shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(N * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    def subset(I):
        return PackedDataset(
            vol=pack["vol"][I],
            labels=pack["labels"][I],
            modules=[pack["modules"][i] for i in I],
            vol_mean=pack["vol_mean"],
            vol_std=pack["vol_std"],
            KS=pack["KS"],
            degrees=pack["degrees"],
            res=pack["res"],
            label_map=pack["label_map"],
        )

    return subset(tr_idx), subset(val_idx)


def iterate_batches(
    ds: PackedDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    epochs: int | None = None,
):
    rng = np.random.RandomState(seed)
    N = ds.vol.shape[0]
    order = np.arange(N)
    n_epochs = 10**12 if epochs is None else epochs
    for _ in range(n_epochs):
        if shuffle:
            rng.shuffle(order)
        for i in range(0, N, batch_size):
            idx = order[i : i + batch_size]
            yield {
                "vol": ds.vol[idx],  # (B,H,W,KS,3)
                "labels": ds.labels[idx],  # (B,)
                "modules": [ds.modules[j] for j in idx],
            }


# ---------- class prototypes for bridge guidance ----------
def compute_class_prototypes(
    vol: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    vol: (N,H,W,KS,3) normalized
    labels: (N,)
    returns: (num_classes,H,W,KS,3) normalized prototypes
    """
    H, W, K, C = vol.shape[1:]
    prot = np.zeros((num_classes, H, W, K, C), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int32)
    for i in range(vol.shape[0]):
        y = int(labels[i])
        prot[y] += vol[i]
        counts[y] += 1
    for y in range(num_classes):
        if counts[y] > 0:
            prot[y] /= counts[y]
    return prot
