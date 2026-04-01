from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class KVCacheEntry:
    """Cached key/value projections for one attention site."""

    key: Optional[Tensor] = None
    value: Optional[Tensor] = None

    def is_valid(self) -> bool:
        return self.key is not None and self.value is not None

    def __getitem__(self, indices) -> KVCacheEntry:
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key[indices], value=self.value[indices])

    def __setitem__(self, indices, other: KVCacheEntry):
        if self.is_valid() and other.is_valid():
            self.key[indices] = other.key
            self.value[indices] = other.value

    def to(self, device) -> KVCacheEntry:
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key.to(device), value=self.value.to(device))

    @staticmethod
    def concat(entries: List[KVCacheEntry], dim: int = 0) -> KVCacheEntry:
        keys = [entry.key for entry in entries if entry.is_valid()]
        values = [entry.value for entry in entries if entry.is_valid()]
        if not keys:
            return KVCacheEntry()
        return KVCacheEntry(key=torch.cat(keys, dim=dim), value=torch.cat(values, dim=dim))


@dataclass
class KVCache:
    """Per-layer KV cache container."""

    kv: Dict[int, KVCacheEntry] = field(default_factory=dict)

    def is_populated(self) -> bool:
        return any(entry.is_valid() for entry in self.kv.values())

    def __getitem__(self, indices) -> KVCache:
        return self.__class__(kv={idx: entry[indices] for idx, entry in self.kv.items()})

    def __setitem__(self, indices, other: KVCache):
        for idx, other_entry in other.kv.items():
            if idx in self.kv:
                device = self.kv[idx].key.device
                self.kv[idx][indices] = other_entry.to(device)

    def to(self, device) -> KVCache:
        return self.__class__(kv={idx: entry.to(device) for idx, entry in self.kv.items()})

    @staticmethod
    def concat(caches: List[KVCache], dim: int = 0) -> KVCache:
        merged = {}
        all_indices = set()
        for cache in caches:
            all_indices.update(cache.kv.keys())
        for idx in sorted(all_indices):
            merged[idx] = KVCacheEntry.concat([cache.kv[idx] for cache in caches if idx in cache.kv], dim=dim)
        return KVCache(kv=merged)

    def preallocate(self, reference: KVCache, batch_shape: tuple, device="cpu"):
        batch_ndim = len(batch_shape)
        for idx, ref_entry in reference.kv.items():
            if ref_entry.is_valid():
                key_shape = batch_shape + ref_entry.key.shape[batch_ndim:]
                value_shape = batch_shape + ref_entry.value.shape[batch_ndim:]
                self.kv[idx] = KVCacheEntry(
                    key=torch.zeros(key_shape, dtype=ref_entry.key.dtype, device=device),
                    value=torch.zeros(value_shape, dtype=ref_entry.value.dtype, device=device),
                )


@dataclass
class TabICLCache:
    """Top-level cache for the TabICL inference pipeline."""

    col_cache: Optional[KVCache] = None
    icl_cache: Optional[KVCache] = None
    train_shape: Tuple[int, int, int] = (0, 0, 0)
    num_classes: Optional[int] = None

    def __post_init__(self):
        if self.col_cache is None:
            self.col_cache = KVCache()
        if self.icl_cache is None:
            self.icl_cache = KVCache()

    def is_empty(self) -> bool:
        col_empty = self.col_cache is None or not self.col_cache.kv
        icl_empty = self.icl_cache is None or not self.icl_cache.kv
        return col_empty and icl_empty

    def slice_batch(self, start: int, end: int) -> TabICLCache:
        return self[slice(start, end)]

    def __getitem__(self, indices) -> TabICLCache:
        if isinstance(indices, slice):
            batch_size = max(0, indices.stop - (indices.start or 0))
        else:
            batch_size = self.train_shape[0]

        return TabICLCache(
            col_cache=self.col_cache[indices] if self.col_cache else KVCache(),
            icl_cache=self.icl_cache[indices] if self.icl_cache else KVCache(),
            train_shape=(batch_size, *self.train_shape[1:]),
            num_classes=self.num_classes,
        )

    def to(self, device) -> TabICLCache:
        return TabICLCache(
            col_cache=self.col_cache.to(device) if self.col_cache else KVCache(),
            icl_cache=self.icl_cache.to(device) if self.icl_cache else KVCache(),
            train_shape=self.train_shape,
            num_classes=self.num_classes,
        )

    @staticmethod
    def concat(caches: List[TabICLCache], dim: int = 0) -> TabICLCache:
        if not caches:
            return TabICLCache()

        train_shape = caches[0].train_shape
        if len(train_shape) == 3:
            train_shape = (
                sum(cache.train_shape[0] for cache in caches),
                train_shape[1],
                train_shape[2],
            )

        return TabICLCache(
            col_cache=KVCache.concat([cache.col_cache for cache in caches], dim=dim),
            icl_cache=KVCache.concat([cache.icl_cache for cache in caches], dim=dim),
            train_shape=train_shape,
            num_classes=caches[0].num_classes,
        )
