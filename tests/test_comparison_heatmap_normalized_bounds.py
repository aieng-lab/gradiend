import pytest

from gradiend.visualizer.heatmaps import plot_comparison_heatmap


def test_normalized_cross_encoding_positive_expands_default_vmax():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "cross_encoding_positive_mean",
                "row_normalized_by_diagonal": True,
                "model_ids": ["a", "b"],
                "matrix": [[1.0, 1.4], [0.6, 1.0]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(0.0)
        assert mesh.norm.vmax == pytest.approx(1.4)
    finally:
        plt.close("all")


def test_normalized_cross_encoding_difference_uses_symmetric_signed_bounds():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "cross_encoding_positive_minus_negative",
                "row_normalized_by_diagonal": True,
                "model_ids": ["a", "b"],
                "matrix": [[1.0, -1.8], [0.25, 1.0]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(-1.8)
        assert mesh.norm.vmax == pytest.approx(1.8)
    finally:
        plt.close("all")


def test_cross_encoding_difference_uses_data_driven_signed_bounds_without_normalization():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "cross_encoding_positive_minus_negative",
                "model_ids": ["a", "b"],
                "matrix": [[0.9, -1.8], [0.25, 1.2]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(-1.8)
        assert mesh.norm.vmax == pytest.approx(1.8)
    finally:
        plt.close("all")


def test_anchor_aligned_encoding_uses_signed_bounds_and_coolwarm():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "anchor_aligned_encoding_factual_mean",
                "model_ids": ["A", "B"],
                "column_ids": ["A", "B"],
                "matrix": [[0.74, -0.23], [0.50, 0.86]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(-1.0)
        assert mesh.norm.vmax == pytest.approx(1.0)
        assert mesh.cmap.name == "coolwarm"
    finally:
        plt.close("all")
