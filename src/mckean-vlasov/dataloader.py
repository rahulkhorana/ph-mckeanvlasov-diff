# JAX-friendly dataset utilities:
# - loads your packed .pt and exposes vol=(N,H,W,KS,3)
# - simple train/val split
# - host iterator (numpy) and JAX iterator (jnp, device_put, prefetch)
# - optional pmap sharding (B -> [n_local_devices, B//n, ...])
from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Tuple, Optional
import collections

import numpy as np
import torch
import jax
import jax.numpy as jnp


# ---------- load & reshape ----------


def load_packed_pt(path: str, require_modules: bool = True) -> Dict[str, Any]:
    """
    Returns dict with:
      vol: (N,H,W,KS,3) float32    # MPL volume: KS landscapes over 3 degrees
      pcs: (N,P,6) float32
      labels: (N,) int64
      modules: list[Any]           # ragged modules per item
      meta, label_map, KS, degrees, res
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    pack = torch.load(path, map_location="cpu")

    # landscapes: (N, 3, KS, H, W) -> (N, H, W, KS, 3)
    lands_t = pack["landscapes"]
    if (
        not isinstance(lands_t, torch.Tensor)
        or lands_t.ndim != 5
        or lands_t.shape[1] != 3
    ):
        raise ValueError(
            f"Expected pack['landscapes'] torch.Tensor (N,3,KS,H,W), got {type(lands_t)} {getattr(lands_t,'shape',None)}"
        )
    lands_np = lands_t.cpu().numpy().astype(np.float32)
    vol = np.transpose(lands_np, (0, 3, 4, 2, 1))  # (N,H,W,KS,3)

    modules = pack.get("modules", None)
    if require_modules and modules is None:
        raise ValueError("Dataset has no 'modules' but require_modules=True")

    pcs = pack["pcs"].cpu().numpy().astype(np.float32)  # (N,P,6)
    labels = pack["labels"].cpu().numpy().astype(np.int64)  # (N,)
    meta: Dict[str, Any] = pack.get("meta", {})
    label_map: Dict[int, str] = pack.get("label_map", {})

    N, H, W, KS, _ = vol.shape
    degrees = int(lands_np.shape[1])  # =3
    res = int(meta.get("landscape_res", H))

    return {
        "vol": vol,  # (N,H,W,KS,3)
        "pcs": pcs,  # (N,P,6)
        "labels": labels,  # (N,)
        "modules": modules,  # list[Any] (ragged)
        "meta": meta,
        "label_map": label_map,
        "KS": int(KS),
        "degrees": int(degrees),
        "res": res,
    }


def describe(pack: Dict[str, Any]) -> str:
    v = pack["vol"]
    s = (
        f"N={v.shape[0]}  vol=(N,H,W,KS,C)={tuple(v.shape)}  "
        f"KS={pack['KS']}  degrees={pack['degrees']}  res={pack['res']}"
    )
    print(s)
    return s


# ---------- tiny dataset & iterator ----------


class PackedDataset:
    """
    Lightweight wrapper over numpy arrays and ragged modules list.
    """

    def __init__(
        self,
        vol: np.ndarray,  # (N,H,W,KS,3)
        labels: np.ndarray,  # (N,)
        modules: List[Any],  # len N
        class_filter: Optional[List[int]] = None,
    ):
        if class_filter is not None:
            mask = np.isin(labels, np.array(class_filter, dtype=np.int64))
            vol = vol[mask]
            labels = labels[mask]
            modules = [m for m, keep in zip(modules, mask.tolist()) if keep]
        self.vol = vol.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.modules = modules

        # sanity
        assert (
            self.vol.ndim == 5 and self.vol.shape[-1] == 3
        ), f"vol must be (N,H,W,KS,3), got {self.vol.shape}"
        assert (
            len(self.modules) == self.vol.shape[0] == self.labels.shape[0]
        ), "vol, labels, modules length mismatch"

    def __len__(self) -> int:
        return int(self.vol.shape[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Any]:
        return self.vol[idx], int(self.labels[idx]), self.modules[idx]


def train_val_split(
    pack: Dict[str, Any],
    val_frac: float = 0.1,
    seed: int = 0,
    class_filter: Optional[List[int]] = None,
) -> Tuple[PackedDataset, PackedDataset]:
    """
    Returns (train_ds, val_ds).
    Optionally filter to subset of classes first via class_filter=[ids].
    """
    vol = pack["vol"]
    labels = pack["labels"]
    modules: List[Any] = pack["modules"]

    if class_filter is not None:
        mask = np.isin(labels, np.array(class_filter, dtype=np.int64))
        vol = vol[mask]
        labels = labels[mask]
        modules = [m for m, keep in zip(modules, mask.tolist()) if keep]

    N = vol.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(val_frac * N))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def subset(ii: np.ndarray) -> PackedDataset:
        return PackedDataset(vol[ii], labels[ii], [modules[i] for i in ii])

    return subset(tr_idx), subset(val_idx)


def iterate_batches(
    ds: PackedDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    epochs: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Host (numpy) iterator. Yields:
      - "vol":    (B,H,W,KS,3) float32
      - "labels": (B,) int64
      - "modules": list[Any] length B
    """
    N = len(ds)
    rng = np.random.RandomState(seed)
    epoch = 0
    while True:
        order = np.arange(N)
        if shuffle:
            rng.shuffle(order)
        for i in range(0, N, batch_size):
            idx = order[i : i + batch_size]
            vol = ds.vol[idx]  # (B,H,W,KS,3)
            labels = ds.labels[idx]  # (B,)
            modules = [ds.modules[j] for j in idx]  # ragged
            yield {"vol": vol, "labels": labels, "modules": modules}
        epoch += 1
        if epochs is not None and epoch >= epochs:
            break


# ---------- JAX helpers: device_put, sharding, prefetch ----------


def _to_jnp_host(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy->jnp on host. Keep ragged 'modules' as-is (Python)."""
    out = {
        "vol": jnp.asarray(batch["vol"], dtype=jnp.float32),
        "labels": jnp.asarray(batch["labels"], dtype=jnp.int32),
        "modules": batch["modules"],  # leave on host; embed outside jit
    }
    return out


def _device_put_batch(batch: Dict[str, Any], device=None) -> Dict[str, Any]:
    """device_put vol & labels; keep modules on host (ragged)."""
    dev = device or jax.devices()[0]
    return {
        "vol": jax.device_put(batch["vol"], dev),
        "labels": jax.device_put(batch["labels"], dev),
        "modules": batch["modules"],
    }


def shard_for_pmap(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reshape vol/labels as (n_local_devices, B//n, ...) for pmap.
    'modules' stays as python list of length B (handle on host).
    """
    ndev = jax.local_device_count()
    B = batch["vol"].shape[0]
    if B % ndev != 0:
        raise ValueError(f"Batch size {B} not divisible by local_device_count {ndev}.")
    b = B // ndev

    def _shard(x):
        return x.reshape(ndev, b, *x.shape[1:])

    return {
        "vol": _shard(batch["vol"]),
        "labels": _shard(batch["labels"]),
        "modules": batch[
            "modules"
        ],  # keep as list; typically embed on host per sub-batch
    }


def prefetch_to_device(
    it: Iterator[Dict[str, Any]],
    size: int = 2,
    to_jnp: bool = True,
    device_put: bool = True,
    for_pmap: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Wrap any iterator and prefetch a few batches to device.

    to_jnp:      convert numpy -> jnp on host first
    device_put:  place vol/labels on device
    for_pmap:    additionally shard vol/labels as (n_local_devices, B//n, ...)

    Note: 'modules' remains a python list in all cases (embedded outside jit).
    """
    queue: collections.deque = collections.deque()
    it = iter(it)

    # prime buffer
    for _ in range(size):
        try:
            b = next(it)
            if to_jnp:
                b = _to_jnp_host(b)
            if for_pmap:
                # device_put before/after sharding both work in practice;
                # we shard on host jnp then device_put implicitly by pmap.
                b = shard_for_pmap(b)
            if device_put and not for_pmap:
                b = _device_put_batch(b)
            queue.append(b)
        except StopIteration:
            break

    while queue:
        yield queue.popleft()
        try:
            b = next(it)
            if to_jnp:
                b = _to_jnp_host(b)
            if for_pmap:
                b = shard_for_pmap(b)
            if device_put and not for_pmap:
                b = _device_put_batch(b)
            queue.append(b)
        except StopIteration:
            pass


# ---------- convenience: JAX-ready iterator ----------


def iterate_batches_jax(
    ds: PackedDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    epochs: Optional[int] = None,
    prefetch: int = 2,
    for_pmap: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Builds on iterate_batches and returns a JAX-ready stream:
    - jnp arrays
    - device_put (single device) or sharded for pmap
    - prefetch buffer

    Example:
      train_iter = iterate_batches_jax(train_ds, bs, prefetch=4)           # single device
      train_iter = iterate_batches_jax(train_ds, bs, for_pmap=True)        # multi-device

    NOTE: 'modules' remains a python list in both cases; embed them outside jit.
    """
    base = iterate_batches(ds, batch_size, shuffle=shuffle, seed=seed, epochs=epochs)
    return prefetch_to_device(
        base,
        size=max(0, int(prefetch)),
        to_jnp=True,
        device_put=True,
        for_pmap=for_pmap,
    )
