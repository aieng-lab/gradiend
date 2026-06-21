import math

import pytest

from gradiend.visualizer.topk.pairwise_heatmap import (
    plot_topk_overlap_heatmap,
    plot_topk_overlap_heatmap_with_correlation,
)


class _DummyTopKModel:
    def __init__(self, weights, total_count=None):
        self._weights = list(weights)
        self._total_count = int(total_count) if total_count is not None else len(self._weights)

    def get_topk_weights(self, part="decoder-weight", topk=100):
        if isinstance(topk, float):
            k = max(1, int(math.ceil(topk * self._total_count)))
        else:
            k = int(topk)
        return self._weights[:k]


def _make_models():
    return {
        "A": _DummyTopKModel([1, 2, 3, 4]),
        "B": _DummyTopKModel([3, 4, 5, 6]),
    }


def _make_fractional_models():
    return {
        "A": _DummyTopKModel([1, 2], total_count=200),
        "B": _DummyTopKModel([2, 3, 4], total_count=300),
    }


def test_overlap_heatmap_percentage_fraction_bounds_match_colorbar():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        result = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            value="intersection_frac",
            percentages=True,
            vmin=0.0,
            vmax=1.0,
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]

        assert result["matrix"][0][1] == 50.0
        assert mesh.norm.vmin == 0.0
        assert mesh.norm.vmax == 100.0
    finally:
        plt.close("all")


def test_overlap_heatmap_can_use_existing_axis_and_return_fig_ax():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        data, returned_fig, returned_ax = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            show=False,
            return_data=True,
            return_fig_ax=True,
            ax=ax,
        )

        assert data["matrix"][0][1] == 50.0
        assert returned_fig is fig
        assert returned_ax is ax
        ax.set_title("Custom heatmap")
        assert ax.get_title() == "Custom heatmap"
    finally:
        plt.close("all")


def test_overlap_heatmap_percentage_count_bounds_match_colorbar():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        result = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            value="intersection",
            percentages=True,
            vmin=0.0,
            vmax=4.0,
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]

        assert result["matrix"][0][1] == 50.0
        assert mesh.norm.vmin == 0.0
        assert mesh.norm.vmax == 100.0
    finally:
        plt.close("all")


def test_overlap_heatmap_with_correlation_accepts_annot_fmt_when_metrics_missing(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        "gradiend.visualizer.topk.pairwise_heatmap._extract_best_correlation_for_models",
        lambda models: {},
    )

    try:
        result = plot_topk_overlap_heatmap_with_correlation(
            _make_models(),
            topk=4,
            annot_fmt=".0f",
            show=False,
            return_data=True,
        )

        assert result["matrix"][0][1] == pytest.approx(0.5)
    finally:
        plt.close("all")


def test_overlap_heatmap_fractional_topk_uses_actual_resolved_set_sizes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    result = plot_topk_overlap_heatmap(
        _make_fractional_models(),
        topk=0.01,
        value="intersection_frac",
        percentages=False,
        show=False,
        return_data=True,
    )

    assert result["resolved_topk"] == {"A": 2, "B": 3}
    assert result["matrix"][0][1] == pytest.approx(0.5)


def test_overlap_heatmap_fractional_topk_percentages_use_pairwise_smaller_set():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    result = plot_topk_overlap_heatmap(
        _make_fractional_models(),
        topk=0.01,
        value="intersection",
        percentages=True,
        show=False,
        return_data=True,
    )

    assert result["matrix"][0][1] == pytest.approx(50.0)


def test_overlap_heatmap_rejects_custom_count_bounds_for_nonuniform_percentage_sets():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    with pytest.raises(ValueError, match="Custom vmin/vmax"):
        plot_topk_overlap_heatmap(
            _make_fractional_models(),
            topk=0.01,
            value="intersection",
            percentages=True,
            vmin=0.0,
            vmax=2.0,
            show=False,
            return_data=True,
        )
