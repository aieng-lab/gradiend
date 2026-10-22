"""Top-k overlap must compare base-parameter indices, also for bare (differently pruned) GRADIEND models."""

import torch

from gradiend.comparison import compute_similarity_matrix
from gradiend.comparison.common import topk_base_indices
from gradiend.model import ParamMappedGradiendModel
from gradiend.visualizer.topk.venn_ import _topk_weights_for_venn_set


def _pruned_model(selected, important_local):
    """A model over base parameter ``w`` (6 entries), pruned to ``selected``; ``important_local`` carry the top weights."""
    model = ParamMappedGradiendModel(
        input_dim=len(selected), latent_dim=1,
        param_map={"w": {"shape": (6,), "repr": "indices", "indices": torch.tensor(selected)}},
        lazy_init=False, device=torch.device("cpu"),
    )
    weights = torch.full((len(selected), 1), 0.01)
    weights[important_local] = 1.0
    with torch.no_grad():
        model.decoder[0].linear.weight.copy_(weights)
    return model


def _models():
    # both models consider base entries 2 and 3 most important, but at different local positions
    return _pruned_model([0, 1, 2, 3], [2, 3]), _pruned_model([2, 3, 4, 5], [0, 1])


def test_bare_models_are_compared_in_base_parameter_space():
    a, b = _models()
    assert sorted(topk_base_indices(a, "decoder-weight", 2)) == [2, 3]
    assert sorted(topk_base_indices(b, "decoder-weight", 2)) == [2, 3]
    matrix = compute_similarity_matrix({"a": a, "b": b}, measure="topk_overlap", part="decoder-weight", topk=2, value="intersection_frac")
    assert matrix["matrix"][0][1] == 1.0  # identical base entries; local indices {2, 3} vs {0, 1} would give 0


def test_venn_sets_use_base_parameter_indices():
    a, b = _models()
    assert set(_topk_weights_for_venn_set(a, part="decoder-weight", topk=2)) == {2, 3}
    assert set(_topk_weights_for_venn_set(b, part="decoder-weight", topk=2)) == {2, 3}
