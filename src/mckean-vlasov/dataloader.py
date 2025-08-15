import os
from typing import Any, Dict, Iterator, List, Tuple
import numpy as np
import torch

# ---------- load & reshape ----------


def load_packed_pt(path: str, require_modules: bool = True) -> Dict[str, Any]:
    pack = torch.load(path, map_location="cpu")
    # landscapes: (N, 3, KS, H, W) float16
    lands = pack["landscapes"].float().numpy()  # (N,3,KS,H,W) -> float32
    N, D, KS, H, W = lands.shape
    # NHWKC with C=3(degrees), K=KS (depth)
    vol = np.transpose(lands, (0, 3, 4, 2, 0))  # (N, H, W, KS, 3)
    labels = np.array(pack["labels"].numpy(), dtype=np.int32)  # (N,)
    modules = pack["modules"]  # list length N; each is list of tensors/arrays
    if require_modules and (modules is None):
        raise ValueError("Dataset has no 'modules'. Regenerate with KEEP_MODULES=1.")
    return {
        "vol": vol.astype(np.float32),  # (N,H,W,K,C)
        "labels": labels,  # (N,)
        "modules": modules,  # ragged python list
        "label_map": pack["label_map"],
        "meta": pack["meta"],
        "KS": KS,
        "degrees": D,
        "res": H,
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
    pack: Dict[str, Any], val_frac: float = 0.1, seed: int = 0
) -> Tuple[PackedDataset, PackedDataset]:
    N = pack["vol"].shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(val_frac * N))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

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
