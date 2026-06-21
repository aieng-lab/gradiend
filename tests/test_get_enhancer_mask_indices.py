"""Tests for get_enhancer_mask index-space correctness."""

from __future__ import annotations

import pytest
import torch

from gradiend.model import ParamMappedGradiendModel
from gradiend.model.model_with_gradiend import ModelWithGradiend


def test_get_enhancer_mask_uses_base_global_indices_in_local_sized_mask() -> None:
    """
    Document current behavior: get_enhancer_mask() calls get_topk_weights(), which
    returns base-global indices, but applies them to a mask sized gradiend.input_dim
    (local GRADIEND space). That is only valid before any pruning; after pre-prune
    the indices generally do not fit local positions.

    This does NOT affect top-k overlap heatmaps (they intersect base-global sets directly).
    It can break mask_and_encode(..., topk=...).
    """
    gradiend = ParamMappedGradiendModel(input_dim=4, latent_dim=1, param_map={
        "w": {"shape": (2, 2), "repr": "all"},
    }, lazy_init=False)

    class _Stub(ModelWithGradiend):
        @classmethod
        def _load_model(cls, *args, **kwargs):
            raise NotImplementedError

        def _save_model(self, *args, **kwargs):
            raise NotImplementedError

        def create_gradients(self, *args, **kwargs):
            raise NotImplementedError

    stub = _Stub.__new__(_Stub)
    torch.nn.Module.__init__(stub)
    stub.base_model = torch.nn.Linear(1, 1)
    stub.gradiend = gradiend
    stub._source = "factual"
    stub._target = "diff"
    stub._enhancer_mask_cache = {}

    # Simulate post-pre-prune: local dim 4, but top-k reported in original base-global coords.
    stub.get_topk_weights = lambda part="decoder-weight", topk=3: [10, 20, 30]

    with pytest.raises(IndexError):
        stub.get_enhancer_mask(topk=3, part="decoder-weight")
