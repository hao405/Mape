from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..model.inference_config import InferenceConfig
from ..model.kv_cache import TabICLCache
from ..model.tabicl import _forward_with_explicit_cache, _initialize_tabicl_cache

if TYPE_CHECKING:
    from .preprocessing import EnsembleGenerator
    from ..model.tabicl import TabICL


class TabICLKVCachePlugin:
    """Runtime-side KV cache orchestrator for sklearn inference."""

    def __init__(
        self,
        model: "TabICL",
        device: torch.device,
        inference_config: InferenceConfig,
        batch_size: int | None,
    ) -> None:
        self.model = model
        self.device = device
        self.inference_config = inference_config
        self.batch_size = batch_size

    def build_cache(self, ensemble_generator: "EnsembleGenerator") -> OrderedDict[str, TabICLCache]:
        # X=None is passed explicitly because sklearn's set_output wrapper expects
        # transform() to receive X positionally even when it is unused.
        train_data = ensemble_generator.transform(X=None, mode="train")
        model_kv_cache = OrderedDict()

        for norm_method, (Xs, ys) in train_data.items():
            batch_size = self.batch_size or Xs.shape[0]
            n_batches = int(np.ceil(Xs.shape[0] / batch_size))
            Xs_split = np.array_split(Xs, n_batches)
            ys_split = np.array_split(ys, n_batches)

            caches = []
            for X_batch, y_batch in zip(Xs_split, ys_split):
                X_batch_t = torch.from_numpy(X_batch).float().to(self.device)
                y_batch_t = torch.from_numpy(y_batch).float().to(self.device)
                cache = _initialize_tabicl_cache(X_batch_t, y_batch_t)
                with torch.no_grad():
                    _forward_with_explicit_cache(
                        self.model,
                        cache=cache,
                        X_train=X_batch_t,
                        y_train=y_batch_t,
                        use_cache=False,
                        store_cache=True,
                        inference_config=self.inference_config,
                    )
                caches.append(cache)

            model_kv_cache[norm_method] = TabICLCache.concat(caches)

        return model_kv_cache

    def predict_with_cache(
        self,
        X_test_views,
        model_kv_cache: OrderedDict[str, TabICLCache],
        *,
        average_logits: bool,
        softmax_temperature: float,
    ) -> np.ndarray:
        outputs = []
        for norm_method, (Xs_test,) in X_test_views.items():
            outputs.append(
                self._predict_single_view(
                    Xs_test,
                    model_kv_cache[norm_method],
                    average_logits=average_logits,
                    softmax_temperature=softmax_temperature,
                )
            )
        return np.concatenate(outputs, axis=0)

    def predict_view_with_cache(
        self,
        Xs: np.ndarray,
        kv_cache: TabICLCache,
        *,
        average_logits: bool,
        softmax_temperature: float,
    ) -> np.ndarray:
        return self._predict_single_view(
            Xs,
            kv_cache,
            average_logits=average_logits,
            softmax_temperature=softmax_temperature,
        )

    def _predict_single_view(
        self,
        Xs: np.ndarray,
        kv_cache: TabICLCache,
        *,
        average_logits: bool,
        softmax_temperature: float,
    ) -> np.ndarray:
        batch_size = self.batch_size or Xs.shape[0]
        n_batches = int(np.ceil(Xs.shape[0] / batch_size))
        Xs_split = np.array_split(Xs, n_batches)

        outputs = []
        offset = 0
        for X_batch in Xs_split:
            bs = X_batch.shape[0]
            cache_subset = kv_cache.slice_batch(offset, offset + bs)
            offset += bs

            X_batch_t = torch.from_numpy(X_batch).float().to(self.device)
            with torch.no_grad():
                out = _forward_with_explicit_cache(
                    self.model,
                    cache=cache_subset,
                    X_test=X_batch_t,
                    return_logits=average_logits,
                    softmax_temperature=softmax_temperature,
                    use_cache=True,
                    store_cache=False,
                    inference_config=self.inference_config,
                )
            outputs.append(out.float().cpu().numpy())

        return np.concatenate(outputs, axis=0)
