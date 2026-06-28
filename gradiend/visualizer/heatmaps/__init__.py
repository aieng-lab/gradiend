"""Heatmap visualization APIs."""

from gradiend.visualizer.heatmaps.base import plot_comparison_heatmap
from gradiend.visualizer.heatmaps.encoding import (
    plot_cross_encoding_heatmap,
    plot_gradiend_feature_cross_encoding_heatmap,
    plot_gradiend_transition_cross_encoding_heatmap,
)
from gradiend.visualizer.heatmaps.similarity import (
    plot_similarity_heatmap,
    plot_similarity_heatmap_with_correlation,
)

__all__ = [
    "plot_comparison_heatmap",
    "plot_cross_encoding_heatmap",
    "plot_gradiend_feature_cross_encoding_heatmap",
    "plot_gradiend_transition_cross_encoding_heatmap",
    "plot_similarity_heatmap",
    "plot_similarity_heatmap_with_correlation",
]
