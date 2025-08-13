import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class PackedDataset:
    pcs: np.ndarray  # (N, 100, 6) float32
    labels: np.ndarray  # (N,) int64
    lands: np.ndarray  # (N, H, W, C) float32  with C = 3*KS
    modules: List[Any]  # length N, ragged per-sample structure (REQUIRED)
    label_map: Dict[int, str]
    meta: Dict[str, Any]
    KS: int  # original KS dimension for landscapes

    @property
    def N(self) -> int:
        return int(self.pcs.shape[0])

    def get_item(self, i: int) -> Dict[str, Any]:
        return {
            "pc": self.pcs[i],  # (100, 6) float32
            "label": int(self.labels[i]),
            "landscapes": self.lands[i],  # (H, W, C) float32
            "modules": self.modules[i],  # ragged
        }


def load_packed_pt(
    path: str,
    require_modules: bool = True,
    cast_float32: bool = True,
) -> PackedDataset:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Packed dataset not found: {path}")

    data = torch.load(path, map_location="cpu")

    # Required keys
    for k in ("pcs", "labels", "landscapes", "modules", "label_map", "meta"):
        if k not in data:
            raise KeyError(f"Packed file missing key: '{k}'")

    pcs = data["pcs"]  # (N, 100, 6) torch.Tensor (fp16)
    labels = data["labels"]  # (N,) torch.Tensor (long)
    lands = data["landscapes"]  # (N, 3, KS, RES, RES) torch.Tensor (fp16)
    modules = data["modules"]  # list or None
    label_map = data["label_map"]
    meta = data["meta"]

    if require_modules and modules is None:
        raise ValueError(
            "modules=None in packed file, but require_modules=True. "
            "Regenerate with KEEP_MODULES=1."
        )

    # Convert to NumPy
    pcs_np = pcs.numpy() if pcs.device.type == "cpu" else pcs.cpu().numpy()
    labels_np = labels.numpy().astype(np.int64)
    lands_np = lands.numpy() if lands.device.type == "cpu" else lands.cpu().numpy()

    # Cast to float32 (safer for JAX & training)
    if cast_float32:
        pcs_np = pcs_np.astype(np.float32, copy=False)
        lands_np = lands_np.astype(np.float32, copy=False)

    # Reshape landscapes to NHWC: (N, H, W, C) with C = 3*KS
    if lands_np.ndim != 5:
        raise ValueError(f"landscapes expected (N,3,KS,H,W); got {lands_np.shape}")
    N, D, KS, H, W = lands_np.shape
    if D != 3:
        raise ValueError(
            f"First dim of landscapes must be 3 (degrees H0,H1,H2); got {D}"
        )
    lands_np = lands_np.reshape(N, D * KS, H, W)  # (N, C, H, W)
    lands_np = np.transpose(lands_np, (0, 2, 3, 1))  # (N, H, W, C)

    # Basic integrity checks
    if pcs_np.shape[0] != N or labels_np.shape[0] != N or len(modules) != N:
        raise ValueError("Inconsistent N across pcs/labels/landscapes/modules")

    return PackedDataset(
        pcs=pcs_np,
        labels=labels_np,
        lands=lands_np,
        modules=list(modules) if modules is not None else [None] * N,
        label_map=label_map,
        meta=meta,
        KS=KS,
    )


def train_val_split(
    ds: PackedDataset,
    val_frac: float = 0.1,
    seed: int = 0,
    stratify: bool = True,
) -> Tuple[PackedDataset, PackedDataset]:
    """
    Deterministic split. If stratify=True, preserve label proportions.
    """
    rng = np.random.RandomState(seed)
    N = ds.N
    idx = np.arange(N)

    if stratify:
        # group by label and split inside each
        labels = ds.labels
        train_idx, val_idx = [], []
        for y in np.unique(labels):
            grp = idx[labels == y]
            rng.shuffle(grp)
            k = int(round(val_frac * len(grp)))
            val_idx.append(grp[:k])
            train_idx.append(grp[k:])
        train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
        val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    else:
        rng.shuffle(idx)
        k = int(round(val_frac * N))
        val_idx, train_idx = idx[:k], idx[k:]

    def _slice(ds: PackedDataset, sel: np.ndarray) -> PackedDataset:
        return PackedDataset(
            pcs=ds.pcs[sel],
            labels=ds.labels[sel],
            lands=ds.lands[sel],
            modules=[ds.modules[i] for i in sel],
            label_map=ds.label_map,
            meta=ds.meta,
            KS=ds.KS,
        )

    return _slice(ds, train_idx), _slice(ds, val_idx)


def iterate_batches(
    ds: PackedDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 0,
    epochs: Optional[int] = None,  # None = infinite
) -> Iterator[Dict[str, Any]]:
    """
    Deterministic epoch-based iterator. Yields dicts with:
      'lands': (B,H,W,C) float32
      'modules': list length B (ragged)
      (you can add 'pcs'/'labels' here if needed)
    """
    N = ds.N
    rng = np.random.RandomState(seed)
    epoch = 0
    while epochs is None or epoch < epochs:
        order = np.arange(N)
        if shuffle:
            rng.shuffle(order)

        for start in range(0, N, batch_size):
            end = start + batch_size
            if end > N and drop_last:
                break
            sel = order[start:end]
            yield {
                "lands": ds.lands[sel],
                "modules": [ds.modules[i] for i in sel],
                # uncomment if you need them in-batch:
                # "pcs": ds.pcs[sel],
                # "labels": ds.labels[sel],
            }
        epoch += 1


def describe(ds: PackedDataset) -> str:
    N, H, W, C = ds.lands.shape
    return (
        f"N={ds.N}  pcs={ds.pcs.shape}  lands={ds.lands.shape}  "
        f"labels={ds.labels.shape}  KS={ds.KS}  C=3*KS={3*ds.KS}  "
        f"modules_len={len(ds.modules)}"
    )
