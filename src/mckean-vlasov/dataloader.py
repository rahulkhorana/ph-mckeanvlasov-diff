# dataloader.py — load MPL volumes, split, and (optionally) normalize by train percentiles
import os
from typing import Any, Dict, Iterator, List, Tuple, Optional
import numpy as np
import torch

# ---------------- I/O ----------------


def load_packed_pt(path: str, require_modules: bool = True) -> Dict[str, Any]:
    pack = torch.load(path, map_location="cpu")

    lands_t = pack["landscapes"]  # (N, 3, KS, H, W) float16/float32
    lands = lands_t.cpu().numpy().astype(np.float32)

    # reshape to (N,H,W,KS,C)
    vol = np.transpose(lands, (0, 3, 4, 2, 1))

    modules = pack.get("modules", None)
    if require_modules and (modules is None):
        raise ValueError("Dataset has no modules but require_modules=True")

    pcs = pack["pcs"].cpu().numpy().astype(np.float32) if "pcs" in pack else None
    labels = pack["labels"].cpu().numpy().astype(np.int64) if "labels" in pack else None
    meta = dict(pack.get("meta", {}))
    label_map = pack.get("label_map", None)
    KS = lands.shape[2]
    degrees = 3
    res = int(meta.get("landscape_res", lands.shape[-1]))

    return {
        "vol": vol,  # (N, H, W, KS, 3)
        "pcs": pcs,  # optional
        "labels": labels,  # (N,)
        "modules": modules,  # list of list of tensors
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


# ------------- normalization helpers -------------


def _compute_channel_percentiles(
    vol: np.ndarray, p_lo=1.0, p_hi=99.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    vol: (N,H,W,K,C) in real data scale
    returns per-channel low/high of shape (C,)
    """
    N, H, W, K, C = vol.shape
    x = vol.reshape(N * H * W * K, C)
    lo = np.percentile(x, p_lo, axis=0).astype(np.float32)
    hi = np.percentile(x, p_hi, axis=0).astype(np.float32)
    return lo, hi


def _apply_norm(vol: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Map each channel to roughly [-1, 1] using train percentiles.
    """
    lo = lo[None, None, None, None, :]
    hi = hi[None, None, None, None, :]
    mid = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-6)
    return (vol - mid) / half


def _invert_norm(vol_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lo = lo[None, None, None, None, :]
    hi = hi[None, None, None, None, :]
    mid = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-6)
    return vol_norm * half + mid


# ------------- dataset container -------------


class PackedDataset:
    def __init__(
        self,
        vol: np.ndarray,  # (N,H,W,K,C) float32
        labels: Optional[np.ndarray],
        modules: Optional[List[Any]],
        norm: Optional[Dict[str, Any]] = None,  # {"lo": (C,), "hi": (C,)}
    ):
        self.vol = vol
        self.labels = labels
        self.modules = modules
        self.norm = norm or {}

    def __len__(self) -> int:
        return int(self.vol.shape[0])

    def __getitem__(self, idx: int):
        out = {"vol": self.vol[idx]}
        if self.labels is not None:
            out["labels"] = int(self.labels[idx])
        if self.modules is not None:
            out["modules"] = self.modules[idx]
        return out


# ------------- splitting & iterating -------------


def train_val_split(
    pack: Dict[str, Any],
    val_frac: float = 0.1,
    seed: int = 0,
    normalize: bool = True,
) -> Tuple[PackedDataset, PackedDataset, Dict[str, Any]]:
    """
    Returns train_ds, val_ds, norm_meta
    If normalize=True, compute per-channel percentiles ONLY on train,
    map both train/val to approximately [-1,1], and return {'norm': {...}}.
    """
    vol = pack["vol"]
    labels = pack.get("labels", None)
    modules = pack.get("modules", None)

    N = vol.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(val_frac * N))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    if normalize:
        lo, hi = _compute_channel_percentiles(vol[tr_idx])
        vol_tr = _apply_norm(vol[tr_idx], lo, hi)
        vol_va = _apply_norm(vol[val_idx], lo, hi)
        norm_meta = {
            "lo": lo.tolist(),
            "hi": hi.tolist(),
            "type": "pct",
            "p": [1.0, 99.0],
        }
    else:
        vol_tr, vol_va = vol[tr_idx], vol[val_idx]
        norm_meta = {}

    def subset(ii, vol_arr):
        return PackedDataset(
            vol_arr,
            labels[ii] if labels is not None else None,
            [modules[i] for i in ii] if modules is not None else None,
            norm=norm_meta,
        )

    return subset(tr_idx, vol_tr), subset(val_idx, vol_va), norm_meta


def iterate_batches(
    ds: PackedDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    epochs: Optional[int] = None,
):
    N = len(ds)
    rng = np.random.RandomState(seed)
    epoch = 0
    while True:
        order = np.arange(N)
        if shuffle:
            rng.shuffle(order)
        for i in range(0, N, batch_size):
            idx = order[i : i + batch_size]
            vol = ds.vol[idx]
            batch = {"vol": vol}
            if ds.labels is not None:
                batch["labels"] = ds.labels[idx]
            if ds.modules is not None:
                batch["modules"] = [  # type:ignore
                    ds.modules[int(j)] for j in idx.tolist()
                ]
            yield batch
        epoch += 1
        if epochs is not None and epoch >= epochs:
            break


# expose invert function for saving
invert_norm = _invert_norm
