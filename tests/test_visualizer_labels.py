"""Tests for non-convergence plot label helpers."""

import pytest

from gradiend import TrainingArguments
from gradiend.visualizer.labels import (
    NON_CONVERGENCE_MARKER,
    NON_CONVERGENCE_MARKER_TEX,
    converged_from_run_info,
    format_plotly_label,
    format_label_with_convergence,
    format_model_labels_with_convergence,
    non_convergence_marker_for_matplotlib,
    resolve_axis_convergence_for_comparison_heatmap,
    resolve_highlight_non_convergence,
    resolve_plot_title_with_convergence,
)


class _TrainerStub:
    def __init__(self, *, run_id="demo_run", highlight=True, converged=False):
        self.run_id = run_id
        self.training_args = TrainingArguments(highlight_non_convergence=highlight)
        self._converged = converged

    def get_training_stats(self):
        return {
            "convergence_info": {"converged": self._converged},
            "training_stats": {},
        }


def test_non_convergence_marker_is_latin_cross():
    assert NON_CONVERGENCE_MARKER == "✝"
    assert NON_CONVERGENCE_MARKER != "✗"


def test_format_label_with_convergence_marker():
    marker = non_convergence_marker_for_matplotlib()
    assert format_label_with_convergence("run_a", converged=False) == f"run_a {marker}"
    assert format_label_with_convergence("run_a", converged=False, highlight_non_convergence=False) == "run_a"
    assert format_label_with_convergence("run_a", converged=True) == "run_a"
    assert format_label_with_convergence("run_a", converged=None) == "run_a"
    already = f"run_a {marker}"
    assert format_label_with_convergence(already, converged=False) == already


def test_format_label_with_convergence_uses_tex_safe_marker(monkeypatch):
    import matplotlib as mpl

    monkeypatch.setitem(mpl.rcParams, "text.usetex", True)
    assert format_label_with_convergence("run_a", converged=False) == f"run_a {NON_CONVERGENCE_MARKER_TEX}"


def test_converged_from_run_info():
    assert converged_from_run_info(None) is None
    assert converged_from_run_info({}) is None
    assert converged_from_run_info({"convergence_info": {"converged": True}}) is True
    assert converged_from_run_info({"convergence_info": {"converged": False}}) is False


def test_resolve_highlight_non_convergence():
    trainer = _TrainerStub(highlight=False)
    assert resolve_highlight_non_convergence(None, trainer=trainer) is False
    assert resolve_highlight_non_convergence(True, trainer=trainer) is True
    assert resolve_highlight_non_convergence(False, trainer=trainer) is False
    assert resolve_highlight_non_convergence(None) is True


def test_resolve_highlight_non_convergence():
    trainer = _TrainerStub(highlight=False)
    assert resolve_highlight_non_convergence(None, trainer=trainer) is False
    assert resolve_highlight_non_convergence(True, trainer=trainer) is True
    assert resolve_highlight_non_convergence(False, trainer=trainer) is False
    assert resolve_highlight_non_convergence(None) is True


def test_resolve_plot_title_with_convergence():
    marker = non_convergence_marker_for_matplotlib()
    trainer = _TrainerStub(run_id="my_run", converged=False)
    assert resolve_plot_title_with_convergence(True, trainer=trainer) == f"my_run {marker}"
    assert resolve_plot_title_with_convergence(True, trainer=trainer, highlight_non_convergence=False) == "my_run"
    assert resolve_plot_title_with_convergence(False, trainer=trainer) is False
    assert resolve_plot_title_with_convergence(None, trainer=trainer) is False
    trainer_ok = _TrainerStub(run_id="ok", converged=True)
    assert resolve_plot_title_with_convergence(True, trainer=trainer_ok) == "ok"


def test_resolve_axis_convergence_from_aligned_rows():
    import pandas as pd

    trainers = {
        "ab": _TrainerStub(converged=True),
        "ac": _TrainerStub(converged=False),
    }
    aligned_rows = pd.DataFrame(
        [
            {"trainer_id": "ab", "anchor_class": "A", "eval_class": "A"},
            {"trainer_id": "ab", "anchor_class": "B", "eval_class": "B"},
            {"trainer_id": "ac", "anchor_class": "A", "eval_class": "C"},
            {"trainer_id": "ac", "anchor_class": "C", "eval_class": "C"},
        ]
    )
    row_status, col_status = resolve_axis_convergence_for_comparison_heatmap(
        {"aligned_rows": aligned_rows},
        models=trainers,
    )
    assert row_status["A"] is False
    assert row_status["B"] is True
    assert row_status["C"] is False
    assert col_status["A"] is True
    assert col_status["B"] is True
    assert col_status["C"] is False


def test_resolve_axis_convergence_from_gradiend_feature_n_matrix():
    trainers = {
        "good": _TrainerStub(converged=True),
        "bad": _TrainerStub(converged=False),
    }
    comparison_data = {
        "measure": "gradiend_feature_cross_encoding_mean",
        "model_ids": ["good", "bad"],
        "column_ids": ["A", "B"],
        "n_matrix": [[3, 0], [2, 4]],
    }
    row_status, col_status = resolve_axis_convergence_for_comparison_heatmap(
        comparison_data,
        models=trainers,
        column_ids=["A", "B"],
    )
    assert row_status["good"] is True
    assert row_status["bad"] is False
    assert col_status["A"] is False
    assert col_status["B"] is False


def test_format_model_labels_with_convergence():
    marker = non_convergence_marker_for_matplotlib()
    class _Model:
        name_or_path = "/tmp/model"

    labels = format_model_labels_with_convergence(
        ["a", "b"],
        models={"a": _Model(), "b": _Model()},
        converged_by_id={"a": True, "b": False},
        highlight_non_convergence=True,
    )
    assert labels["a"] == "a"
    assert labels["b"] == f"b {marker}"


def test_format_plotly_label_hides_hover_helper_names():
    assert format_plotly_label("text") == "Text"
    assert format_plotly_label("text_hover") == "Text"
    assert format_plotly_label("text_:hover") == "Text"
    assert format_plotly_label("data_split") == "Split"


def test_plot_functions_expose_highlight_non_convergence_param():
    """Plot entry points must accept highlight_non_convergence for API consistency."""
    import inspect

    from gradiend.visualizer.heatmaps import (
        plot_comparison_heatmap,
        plot_cross_encoding_heatmap,
        plot_similarity_heatmap,
    )
    from gradiend.visualizer.convergence import plot_training_convergence
    from gradiend.visualizer.encoder_distributions import plot_encoder_distributions
    from gradiend.visualizer.encoder_scatter import plot_encoder_scatter
    from gradiend.visualizer.encoder_by_target import plot_encoder_by_target
    from gradiend.visualizer.encoder_strip_split import plot_encoder_strip_by_split
    from gradiend.visualizer.probability_shifts import plot_probability_shifts
    from gradiend.visualizer.topk.pairwise_heatmap import plot_topk_overlap_heatmap
    from gradiend.visualizer.topk.venn_ import plot_topk_overlap_venn

    for fn in (
        plot_training_convergence,
        plot_encoder_distributions,
        plot_encoder_scatter,
        plot_encoder_by_target,
        plot_encoder_strip_by_split,
        plot_probability_shifts,
        plot_comparison_heatmap,
        plot_similarity_heatmap,
        plot_cross_encoding_heatmap,
        plot_topk_overlap_heatmap,
        plot_topk_overlap_venn,
    ):
        assert "highlight_non_convergence" in inspect.signature(fn).parameters, fn.__name__


def test_plot_encoder_scatter_title_none_disables_title():
    pytest.importorskip("plotly")

    import pandas as pd

    from gradiend.visualizer.encoder_scatter import plot_encoder_scatter

    encoder_df = pd.DataFrame(
        {
            "encoded": [0.1, -0.2],
            "label": ["3SG", "3PL"],
            "source_id": ["3SG", "3PL"],
            "factual_token": ["he", "they"],
            "type": ["training", "training"],
        }
    )

    fig = plot_encoder_scatter(encoder_df=encoder_df, title=None, show=False)

    assert fig.layout.title.text in (None, "")


def test_plot_encoder_strip_by_split_title_none_disables_title():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    import pandas as pd
    from unittest.mock import patch

    from gradiend.visualizer.encoder_strip_split import plot_encoder_strip_by_split

    encoder_df = pd.DataFrame(
        {
            "encoded": [0.1, -0.2, 0.2, -0.1],
            "source_id": ["3SG", "3PL", "3SG", "3PL"],
            "factual_token": ["he", "they", "him", "them"],
            "data_split": ["train", "train", "test", "test"],
            "type": ["training"] * 4,
        }
    )

    with patch("matplotlib.axes.Axes.set_title") as set_title:
        plot_encoder_strip_by_split(encoder_df=encoder_df, title=None, show=False)

    set_title.assert_not_called()
