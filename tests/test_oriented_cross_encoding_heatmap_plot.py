"""Tests for oriented cross-encoding heatmap axis labels and plot wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gradiend.visualizer.heatmaps.encoding import (
    ORIENTED_CROSS_ENCODING_XLABEL_COUNTERFACTUAL,
    ORIENTED_CROSS_ENCODING_XLABEL_FACTUAL,
    ORIENTED_CROSS_ENCODING_XLABEL_TRANSITION,
    ORIENTED_CROSS_ENCODING_YLABEL,
    _oriented_cross_encoding_axis_labels,
    plot_cross_encoding_heatmap,
)


def _oriented_comparison_payload():
    return {
        "measure": "anchor_aligned_encoding_factual_mean",
        "model_ids": ["A", "B"],
        "column_ids": ["A", "B"],
        "rows": ["A", "B"],
        "columns": ["A", "B"],
        "matrix": [[1.0, -0.2], [0.1, 1.0]],
    }


def _dummy_trainers() -> dict[str, object]:
    return {"ab": SimpleNamespace(target_classes=["A", "B"])}


@pytest.mark.parametrize(
    ("alignment", "expected_xlabel"),
    [
        ("factual", ORIENTED_CROSS_ENCODING_XLABEL_FACTUAL),
        ("counterfactual", ORIENTED_CROSS_ENCODING_XLABEL_COUNTERFACTUAL),
        ("transition", ORIENTED_CROSS_ENCODING_XLABEL_TRANSITION),
        ("cf", ORIENTED_CROSS_ENCODING_XLABEL_COUNTERFACTUAL),
        ("alternatives", ORIENTED_CROSS_ENCODING_XLABEL_COUNTERFACTUAL),
    ],
)
def test_oriented_cross_encoding_axis_label_helper(alignment, expected_xlabel):
    ylabel, xlabel = _oriented_cross_encoding_axis_labels(alignment)
    assert ylabel == ORIENTED_CROSS_ENCODING_YLABEL
    assert xlabel == expected_xlabel


def test_plot_cross_encoding_heatmap_sets_encoding_cbar_label(monkeypatch):
    captured: dict[str, object] = {}

    def fake_plot(comparison_data, **kwargs):
        captured["cbar_label"] = kwargs.get("cbar_label")
        return {"path": None}

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_anchor_aligned_encoding_matrix",
        lambda **kwargs: _oriented_comparison_payload(),
    )
    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.plot_comparison_heatmap",
        fake_plot,
    )

    plot_cross_encoding_heatmap(
        _dummy_trainers(),
        ["A", "B"],
        encoder_summary={"ab": {"encoder_df": []}},
        show=False,
    )
    assert captured["cbar_label"] == "Encoding"


def test_plot_cross_encoding_oriented_normalize_scales_rows_by_diagonal(monkeypatch):
    captured: dict[str, object] = {}

    def fake_compute(**kwargs):
        return {
            "measure": "anchor_aligned_encoding_factual_mean",
            "model_ids": ["A", "B"],
            "column_ids": ["A", "B"],
            "rows": ["A", "B"],
            "columns": ["A", "B"],
            "matrix": [[0.8, 0.4], [0.2, 0.6]],
        }

    def fake_plot(comparison_data, **kwargs):
        captured["matrix"] = comparison_data["matrix"]
        captured["measure"] = comparison_data.get("measure")
        captured["cbar_label"] = kwargs.get("cbar_label")
        return {"path": None}

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_anchor_aligned_encoding_matrix",
        fake_compute,
    )
    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.plot_comparison_heatmap",
        fake_plot,
    )

    plot_cross_encoding_heatmap(
        _dummy_trainers(),
        ["A", "B"],
        encoder_summary={"ab": {"encoder_df": []}},
        normalize=True,
        show=False,
    )
    matrix = captured["matrix"]
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(0.5)
    assert matrix[1][0] == pytest.approx(1.0 / 3.0)
    assert matrix[1][1] == pytest.approx(1.0)
    assert captured["measure"] == "anchor_aligned_encoding_factual_mean_row_normalized"
    assert captured["cbar_label"] == "Relative encoding"


def test_plot_cross_encoding_heatmap_resolves_multi_seed_for_stable_trainers(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    captured: dict[str, object] = {}

    def fake_dense(*args, **kwargs):
        captured["seed_selection"] = kwargs.get("seed_selection")
        captured["dispersion"] = kwargs.get("dispersion")
        return _oriented_comparison_payload()

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_dense_anchor_aligned_encoding_matrix",
        fake_dense,
    )
    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.plot_comparison_heatmap",
        lambda *args, **kwargs: {"path": None},
    )

    from gradiend.trainer.core.arguments import TrainingArguments
    from tests.test_multi_seed_view import _local_temp
    from tests.test_trainer_model import MockTrainerForTest

    temp_dir = _local_temp("heatmap_seed_resolve")
    try:
        trainer = MockTrainerForTest(
            model=f"{temp_dir}/model",
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        plot_cross_encoding_heatmap(
            {"ab": trainer},
            ["A", "B"],
            cross_task_eval=True,
            show=False,
        )
        assert captured["seed_selection"] == "all_convergent"
        assert captured["dispersion"] == "std"
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_plot_cross_encoding_heatmap_oriented_sets_default_axis_labels(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_anchor_aligned_encoding_matrix",
        lambda **kwargs: _oriented_comparison_payload(),
    )

    try:
        _, _, ax = plot_cross_encoding_heatmap(
            _dummy_trainers(),
            ["A", "B"],
            encoder_summary={"ab": {"encoder_df": []}},
            show=False,
            return_fig_ax=True,
        )
        assert ax.get_xlabel() == ORIENTED_CROSS_ENCODING_XLABEL_FACTUAL
        assert ax.get_ylabel() == ORIENTED_CROSS_ENCODING_YLABEL
    finally:
        plt.close("all")


def test_plot_cross_encoding_heatmap_oriented_respects_alignment_axis_labels(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    payload = _oriented_comparison_payload()
    payload["measure"] = "anchor_aligned_encoding_counterfactual_mean"

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_anchor_aligned_encoding_matrix",
        lambda **kwargs: payload,
    )

    try:
        _, _, ax = plot_cross_encoding_heatmap(
            _dummy_trainers(),
            ["A", "B"],
            alignment="counterfactual",
            encoder_summary={"ab": {"encoder_df": []}},
            show=False,
            return_fig_ax=True,
        )
        assert ax.get_xlabel() == ORIENTED_CROSS_ENCODING_XLABEL_COUNTERFACTUAL
        assert ax.get_ylabel() == ORIENTED_CROSS_ENCODING_YLABEL
    finally:
        plt.close("all")


def test_plot_cross_encoding_heatmap_oriented_axis_labels_can_be_overridden(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_anchor_aligned_encoding_matrix",
        lambda **kwargs: _oriented_comparison_payload(),
    )

    try:
        _, _, ax = plot_cross_encoding_heatmap(
            _dummy_trainers(),
            ["A", "B"],
            encoder_summary={"ab": {"encoder_df": []}},
            xlabel="Custom probe",
            ylabel="Custom orienting",
            show=False,
            return_fig_ax=True,
        )
        assert ax.get_xlabel() == "Custom probe"
        assert ax.get_ylabel() == "Custom orienting"
    finally:
        plt.close("all")


def test_plot_cross_encoding_heatmap_directed_mode_has_no_default_axis_labels(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_cross_encoding_matrix",
        lambda *args, **kwargs: {
            "measure": "cross_encoding_positive_mean",
            "model_ids": ["a", "b"],
            "matrix": [[1.0, 0.5], [0.4, 1.0]],
        },
    )

    try:
        _, _, ax = plot_cross_encoding_heatmap(
            _dummy_trainers(),
            feature_classes=None,
            run_evaluation=False,
            show=False,
            return_fig_ax=True,
        )
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
    finally:
        plt.close("all")


def test_plot_cross_encoding_heatmap_oriented_marks_non_converged_feature_labels(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt
    import pandas as pd

    from gradiend.visualizer.heatmaps.base import plot_comparison_heatmap
    from gradiend.visualizer.labels import non_convergence_marker_for_matplotlib

    marker = non_convergence_marker_for_matplotlib()

    payload = {
        "measure": "anchor_aligned_encoding_factual_mean",
        "model_ids": ["A", "B", "C"],
        "column_ids": ["A", "B", "C"],
        "rows": ["A", "B", "C"],
        "columns": ["A", "B", "C"],
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    payload["aligned_rows"] = pd.DataFrame(
        [
            {"trainer_id": "ab", "anchor_class": "A", "eval_class": "A"},
            {"trainer_id": "ab", "anchor_class": "B", "eval_class": "B"},
            {"trainer_id": "ac", "anchor_class": "A", "eval_class": "C"},
            {"trainer_id": "ac", "anchor_class": "C", "eval_class": "C"},
        ]
    )

    class _TrainerStub:
        def __init__(self, converged: bool):
            self._converged = converged

        def get_training_stats(self):
            return {"convergence_info": {"converged": self._converged}}

    trainers = {
        "ab": _TrainerStub(True),
        "ac": _TrainerStub(False),
    }

    try:
        _, ax = plot_comparison_heatmap(
            payload,
            models=trainers,
            show=False,
            return_fig_ax=True,
            return_data=False,
        )
        ylabels = [label.get_text() for label in ax.get_yticklabels()]
        xlabels = [label.get_text() for label in ax.get_xticklabels()]
        assert marker in ylabels[0]
        assert marker not in ylabels[1]
        assert marker in ylabels[2]
        assert marker not in xlabels[0]
        assert marker not in xlabels[1]
        assert marker in xlabels[2]
    finally:
        plt.close("all")


def test_plot_cross_encoding_heatmap_oriented_forwards_plot_kwargs(monkeypatch):
    captured: dict = {}

    def _capture_plot_comparison_heatmap(comparison_data, **kwargs):
        captured.update(kwargs)
        return comparison_data

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_anchor_aligned_encoding_matrix",
        lambda **kwargs: _oriented_comparison_payload(),
    )
    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.plot_comparison_heatmap",
        _capture_plot_comparison_heatmap,
    )

    plot_cross_encoding_heatmap(
        _dummy_trainers(),
        ["A", "B"],
        encoder_summary={"ab": {"encoder_df": []}},
        output_path="oriented_cross_encoding.pdf",
        row_label_mapping={"A": "Row A"},
        column_label_mapping={"B": "Col B"},
        xlabel=None,
        show=False,
    )

    assert captured["output_path"] == "oriented_cross_encoding.pdf"
    assert captured["row_label_mapping"] == {"A": "Row A"}
    assert captured["column_label_mapping"] == {"B": "Col B"}
    assert captured["xlabel"] == ORIENTED_CROSS_ENCODING_XLABEL_FACTUAL
    assert captured["ylabel"] == ORIENTED_CROSS_ENCODING_YLABEL
    assert "seed_aggregate" not in captured
    assert "dispersion" not in captured


def test_plot_cross_encoding_heatmap_directed_forwards_only_valid_plot_kwargs(monkeypatch):
    captured: dict = {}

    def _capture_plot_comparison_heatmap(comparison_data, **kwargs):
        captured.update(kwargs)
        return comparison_data

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_cross_encoding_matrix",
        lambda *args, **kwargs: {
            "measure": "cross_encoding_positive_mean",
            "model_ids": ["a", "b"],
            "matrix": [[1.0, 0.5], [0.4, 1.0]],
            "dispersion": kwargs.get("dispersion", "none"),
        },
    )
    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.plot_comparison_heatmap",
        _capture_plot_comparison_heatmap,
    )

    plot_cross_encoding_heatmap(
        _dummy_trainers(),
        feature_classes=None,
        run_evaluation=False,
        seed_aggregate="median",
        dispersion="std",
        cmap="plasma",
        show=False,
    )

    assert captured["cmap"] == "plasma"
    assert "seed_aggregate" not in captured
    assert "dispersion" not in captured
