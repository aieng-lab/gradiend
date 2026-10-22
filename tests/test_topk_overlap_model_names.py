import math

import pytest

from gradiend.visualizer.topk import as_named_models, plot_topk_overlap
from gradiend.visualizer.topk.naming import checkpoint_label


class _Gradiend:
    def __init__(self, name_or_path):
        self.name_or_path = name_or_path


class _DummyModel:
    """Minimal stand-in for ModelWithGradiend: top-k weights plus a checkpoint path."""

    def __init__(self, weights, path=None):
        self._weights = list(weights)
        self.gradiend = _Gradiend(path)

    def get_topk_weights(self, part="decoder-weight", topk=100):
        k = int(math.ceil(topk * len(self._weights))) if isinstance(topk, float) else int(topk)
        return self._weights[:k]


def test_dict_is_returned_unchanged():
    models = {"A": _DummyModel([1]), "B": _DummyModel([2])}
    assert as_named_models(models) is models


def test_trainer_layout_is_labelled_by_run_folder():
    models = [
        _DummyModel([1], "runs/3SG_3PL/model"),
        _DummyModel([2], "runs\\he_she\\model"),
    ]
    assert list(as_named_models(models)) == ["3SG_3PL", "he_she"]


def test_seed_layout_skips_generic_folders():
    assert checkpoint_label(_DummyModel([1], "out/gender_de/seeds/seed_42/model")) == "gender_de"


def test_real_multi_seed_layout_is_labelled_by_run_id():
    # layout written by multi-seed training: <experiment>/<run_id>/seeds/seed_N (no trailing "model")
    assert checkpoint_label(_DummyModel([1], "runs/pronouns_multi/seeds/seed_1")) == "pronouns_multi"


def test_seed_group_uses_first_member():
    from gradiend.trainer.core.seed_models import SeedModelGroup

    group = SeedModelGroup(
        [_DummyModel([1], "runs/white_black/seeds/seed_1/model"), _DummyModel([1], "runs/white_black/seeds/seed_2/model")],
        seed_values=[1, 2],
    )
    assert list(as_named_models([group])) == ["white_black"]


def test_missing_checkpoint_path_raises_instead_of_inventing_labels():
    with pytest.raises(ValueError, match="Pass a dict"):
        as_named_models([_DummyModel([1], None)])


def test_duplicate_labels_raise():
    models = [_DummyModel([1], "a/same/model"), _DummyModel([2], "b/same/model")]
    with pytest.raises(ValueError, match="not unique"):
        as_named_models(models)


def _models(names):
    return [_DummyModel(list(range(i, i + 4)), f"runs/{n}/model") for i, n in enumerate(names)]


def test_plot_topk_overlap_list_heatmap_is_labelled_by_checkpoint_folders():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        # more than 6 models -> heatmap path (same call as in the paper)
        result = plot_topk_overlap(_models("ABCDEFG"), topk=4, show=False, return_data=True)
        assert list(result["model_ids"]) == list("ABCDEFG")
    finally:
        plt.close("all")


def test_plot_topk_overlap_list_venn_is_labelled_by_checkpoint_folders():
    pytest.importorskip("matplotlib")
    pytest.importorskip("matplotlib_venn")
    import matplotlib.pyplot as plt

    try:
        # 2-6 models -> Venn path
        result = plot_topk_overlap(_models("XY"), topk=4, show=False)
        assert list(result["per_model"]) == ["X", "Y"]
    finally:
        plt.close("all")
