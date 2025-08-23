import os
from typing import Any, Dict, Iterator, List, Tuple
import numpy as np
import torch

# ---------- load & reshape ----------


def load_packed_pt(path: str, require_modules: bool = True):
    pack = torch.load(path, map_location="cpu")

    lands_t = pack["landscapes"]  # (N, 3, KS, H, W) torch.float16
    lands = lands_t.cpu().numpy().astype(np.float32)

    # correct permutation: (N, H, W, KS, 3)
    vol = np.transpose(lands, (0, 3, 4, 2, 1))

    modules = pack["modules"]
    if require_modules and (modules is None):
        raise ValueError("Dataset has no modules but require_modules=True")

    pcs = pack["pcs"].cpu().numpy().astype(np.float32)  # (N, 100, 6)
    labels = pack["labels"].cpu().numpy().astype(np.int64)  # (N,)
    meta = pack["meta"]
    label_map = pack["label_map"]
    KS = lands.shape[2]  # from (N, 3, KS, H, W)
    degrees = 3  # H0/H1/H2
    res = int(meta.get("landscape_res", lands.shape[-1]))  # H (== W)

    return {
        "vol": vol,  # (N, H, W, KS, 3)
        "pcs": pcs,  # (N, 100, 6)
        "labels": labels,  # (N,)
        "modules": modules,  # list of lists of tensors
        "meta": meta,
        "label_map": label_map,
        "KS": KS,
        "degrees": degrees,
        "res": res,
    }


def describe(pack: Dict[str, Any]) -> str:
    v = pack["vol"]
    s = f"N={v.shape[0]}  vol=(N,H,W,K,C)={tuple(v.shape)}  KS={pack['KS']}  degrees={pack['degrees']}  res={pack['res']}"
    print(s)
    return s


# ---------- tiny dataset & iterator ----------


class PackedDataset:
    def __init__(self, vol: np.ndarray, labels: np.ndarray, modules: List[Any]):
        self.vol = vol  # (N,H,W,K,C) float32
        self.labels = labels  # (N,)
        self.modules = modules

    def __len__(self) -> int:
        return int(self.vol.shape[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Any]:
        return self.vol[idx], int(self.labels[idx]), self.modules[idx]


def train_val_split(
    pack: Dict[str, Any],
    val_frac: float = 0.1,
    seed: int = 0,
    ensure_min_per_class: int = 1,
) -> Tuple[PackedDataset, PackedDataset]:

    vol = pack["vol"]
    labels = pack["labels"]
    modules = pack["modules"]

    N = vol.shape[0]
    assert len(labels) == N and len(modules) == N, "mismatched lengths in pack"

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    train_idx, val_idx = [], []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        # how many for val from this class
        n_val_c = int(round(len(idx_c) * val_frac))
        if ensure_min_per_class > 0:
            n_val_c = max(n_val_c, min(ensure_min_per_class, len(idx_c)))
        # avoid empty train slice for tiny classes
        if n_val_c >= len(idx_c) and len(idx_c) > 1:
            n_val_c = len(idx_c) - 1
        val_idx.append(idx_c[:n_val_c])
        train_idx.append(idx_c[n_val_c:])

    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)
    tr_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    def subset(ii):
        return PackedDataset(
            pack["vol"][ii], pack["labels"][ii], [pack["modules"][i] for i in ii]
        )

    return subset(tr_idx), subset(val_idx)


def iterate_batches(
    ds: PackedDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    epochs: int | None = None,
) -> Iterator[Dict[str, Any]]:
    N = len(ds)
    rng = np.random.RandomState(seed)
    epoch = 0
    while True:
        order = np.arange(N)
        if shuffle:
            rng.shuffle(order)
        for i in range(0, N, batch_size):
            idx = order[i : i + batch_size]
            vol = ds.vol[idx]  # (B,H,W,K,C)
            labels = ds.labels[idx]  # (B,)
            modules = [ds.modules[j] for j in idx]
            yield {"vol": vol, "labels": labels, "modules": modules}
        epoch += 1
        if epochs is not None and epoch >= epochs:
            break
