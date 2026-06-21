from types import SimpleNamespace

import pytest
import torch

from gradiend.comparison.similarity import (
    compute_grouped_similarity_matrices,
    compute_similarity_matrix,
)


class _TopKModel:
    def __init__(self, indices):
        self.indices = list(indices)

    def get_topk_weights(self, part="decoder-weight", topk=100):
        return self.indices[: int(topk)]


class _TinyGradiend:
    def __init__(self, decoder_rows, *, param_names=None, with_param_map=True):
        decoder_weight = torch.tensor(decoder_rows, dtype=torch.float32)
        self.input_dim = int(decoder_weight.shape[0])
        self.decoder = [SimpleNamespace(linear=SimpleNamespace(weight=decoder_weight))]
        self.encoder = [SimpleNamespace(linear=SimpleNamespace(weight=decoder_weight.t().contiguous()))]
        self._base_map = torch.arange(self.input_dim, dtype=torch.long)
        self._param_names = param_names or {
            idx: f"encoder.layer.{idx // 2}.weight"
            for idx in range(self.input_dim)
        }
        if with_param_map:
            self.param_map = {
                f"p{idx}": {"shape": (1,), "repr": "all"}
                for idx in range(self.input_dim)
            }

    def _require_built(self):
        return None

    def _get_base_global_index_map(self):
        return self._base_map

    def decode_base_global_index(self, idx):
        return {"param_name": self._param_names[int(idx)]}

    def get_topk_weights(self, part="decoder-weight", topk=None):
        if topk is None:
            return list(range(self.input_dim))
        return list(range(min(int(topk), self.input_dim)))

    def get_weight_importance(self, part="decoder-weight"):
        return self.decoder[0].linear.weight.abs().sum(dim=1)

    def get_update_vector(self, part="decoder-sum"):
        return self.get_weight_importance()


class _TinyModel:
    def __init__(self, decoder_rows, **kwargs):
        self.gradiend = _TinyGradiend(decoder_rows, **kwargs)

    def get_topk_weights(self, part="decoder-weight", topk=100):
        return self.gradiend.get_topk_weights(part=part, topk=topk)


def test_compute_similarity_matrix_topk_overlap_direct_payload():
    models = {
        "a": _TopKModel([1, 2, 3]),
        "b": _TopKModel([2, 3, 4]),
    }

    result = compute_similarity_matrix(models, measure="topk_overlap", topk=3)

    assert result["measure"] == "topk_overlap"
    assert result["part"] == "decoder-weight"
    assert result["model_ids"] == ["a", "b"]
    assert result["matrix"][0][0] == pytest.approx(1.0)
    assert result["matrix"][0][1] == pytest.approx(2.0 / 3.0)
    assert result["resolved_topk"] == {"a": 3, "b": 3}


def test_compute_similarity_matrix_multi_seed_reports_cell_counts_and_stats():
    models = {
        "a": [_TopKModel([1, 2]), _TopKModel([1, 3])],
        "b": [_TopKModel([2, 3]), _TopKModel([3, 4])],
    }

    result = compute_similarity_matrix(
        models,
        measure="topk_overlap",
        topk=2,
        dispersion="std",
    )

    assert result["multi_seed"] is True
    assert result["n_matrix"] == [[4, 4], [4, 4]]
    assert result["matrix"][0][1] == pytest.approx(0.375)
    assert result["cell_stats"][0][1]["scores"] == pytest.approx([0.5, 0.0, 0.5, 0.5])
    assert "std" in result["cell_stats"][0][1]


def test_grouped_similarity_matrix_groups_by_decoded_layer_names():
    models = {
        "a": _TinyModel([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        "b": _TinyModel([[0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]]),
    }

    grouped = compute_grouped_similarity_matrices(
        models,
        measure="cosine",
        part="decoder-weight",
        group_by="layer",
    )

    assert set(grouped) == {"layer_0", "layer_1"}
    assert grouped["layer_0"]["model_ids"] == ["a", "b"]
    assert grouped["layer_0"]["matrix"][0][1] == pytest.approx(1.0)
    assert grouped["layer_1"]["matrix"][0][1] == pytest.approx(1.0)


def test_spearman_metadata_distinguishes_dense_fallback_from_sparse_exact():
    models = {
        "a": _TinyModel([[1.0], [2.0], [3.0]], with_param_map=False),
        "b": _TinyModel([[3.0], [2.0], [1.0]], with_param_map=False),
    }

    result = compute_similarity_matrix(
        models,
        measure="spearman_signed",
        part="decoder-weight",
    )

    assert result["fallback_dense"] is True
    assert result["full_zero_filled"] is False
    assert result["sparse_exact"] is False
    assert result["matrix"][0][1] == pytest.approx(-1.0)
