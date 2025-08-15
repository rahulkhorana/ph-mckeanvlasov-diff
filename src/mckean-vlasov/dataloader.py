import os
from typing import Dict, Any, Iterator, Optional, Tuple
import numpy as np
import torch


class PackedDataset:
    """Simple container to hold numpy arrays for JAX."""

    def __init__(self, lands_vol: np.ndarray, modules, meta: Dict[str, Any]):
        # lands_vol: (N,H,W,K,3) float32
        self.lands = lands_vol
        self.modules = modules
        self.meta = meta

    @property
    def N(self):
        return self.lands.shape[0]

    def __len__(self):
        return int(self.lands.shape[0])

    def __getitem__(self, idx):
        return {
            "lands": self.lands[idx],
            "modules": self.modules[idx],
            "meta": None if self.meta is None else self.meta[idx],
        }


def _to_volume(lands_t: torch.Tensor) -> Tuple[np.ndarray, int]:
    """
    Input torch tensor: (N, 3, KS, H, W) float16 from your pack.
    Returns float32 numpy: (N, H, W, KS, 3) and KS.
    """
    assert (
        lands_t.ndim == 5 and lands_t.shape[1] == 3
    ), f"bad landscapes shape {tuple(lands_t.shape)}"
    N, D, KS, H, W = lands_t.shape
    # -> (N, H, W, 3, KS)
    lands = lands_t.permute(0, 3, 4, 1, 2).contiguous().float().cpu().numpy()
    # -> (N, H, W, KS, 3)
    lands = np.transpose(lands, (0, 1, 2, 4, 3)).astype(np.float32)
    return lands, KS


def load_packed_pt(path: str, require_modules: bool = True) -> Dict[str, Any]:
    pack = torch.load(path, map_location="cpu")
    keys = list(pack.keys())
    assert (
        "landscapes" in keys and "modules" in keys and "meta" in keys
    ), f"missing keys {keys}"
    lands, KS = _to_volume(pack["landscapes"])
    modules = pack["modules"]
    if require_modules:
        assert (
            isinstance(modules, list) and len(modules) == lands.shape[0]
        ), "modules not present per-sample"
    meta = dict(pack["meta"])
    meta["KS"] = KS
    return {"lands_vol": lands, "modules": modules, "meta": meta}


def describe(p) -> str:
    lands = p["lands_vol"]
    meta = p["meta"]
    return (
        f"N={lands.shape[0]}  vol=(N,H,W,K,C)={lands.shape}  "
        f"KS={meta.get('KS')}  degrees=3  res={meta.get('landscape_res')}"
    )


def train_val_split(
    packed: Dict[str, Any], val_frac=0.1, seed=0
) -> Tuple[PackedDataset, PackedDataset]:
    rng = np.random.RandomState(seed)
    N = packed["lands_vol"].shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    nv = max(1, int(val_frac * N))
    val_idx = idx[:nv]
    tr_idx = idx[nv:]

    def sel(arr):
        return arr[tr_idx], arr[val_idx]

    lands_tr, lands_val = sel(packed["lands_vol"])
    mods_tr = [packed["modules"][i] for i in tr_idx]
    mods_val = [packed["modules"][i] for i in val_idx]
    meta = packed["meta"]
    return PackedDataset(lands_tr, mods_tr, meta), PackedDataset(
        lands_val, mods_val, meta
    )


def iterate_batches(
    ds: PackedDataset,
    batch_size: int,
    shuffle=True,
    seed=0,
    epochs: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    N = ds.N
    ep = 0
    while epochs is None or ep < epochs:
        order = np.arange(N)
        if shuffle:
            rng.shuffle(order)
        for s in range(0, N, batch_size):
            idx = order[s : s + batch_size]
            yield {"lands": ds.lands[idx], "modules": [ds.modules[i] for i in idx]}
        ep += 1
