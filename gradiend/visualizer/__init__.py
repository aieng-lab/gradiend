"""
Visualizer package: evaluation-related plots.

- Visualizer(trainer): holds trainer; exposes single-model plots (encoder distributions, scatter, convergence).
- encoder_distributions: plot_encoder_distributions(trainer, encoder_df=None, ...)
- topk_venn: compute_topk_sets, plot_topk_venn, plot_topk_neuron_intersection
"""
from gradiend.visualizer.visualizer import Visualizer
from gradiend.visualizer.encoder_distributions import plot_encoder_distributions
from gradiend.visualizer.convergence import plot_training_convergence
from gradiend.visualizer.encoder_scatter import plot_encoder_scatter
from gradiend.visualizer.encoder_strip_split import plot_encoder_strip_by_split
from gradiend.visualizer.encoder_by_target import plot_encoder_by_target
from gradiend.visualizer.heatmaps import (
    plot_anchor_aligned_encoding_heatmap,
    plot_comparison_heatmap,
    plot_cross_encoding_heatmap,
    plot_gradiend_feature_cross_encoding_heatmap,
    plot_similarity_heatmap,
    plot_similarity_heatmap_with_correlation,
)
from gradiend.visualizer.topk import (
    plot_topk_overlap_heatmap,
    plot_topk_overlap_venn,
)
from gradiend.visualizer.topk.venn_ import (
    compute_topk_sets,
    plot_topk_venn,
)

__all__ = [
    "Visualizer",
    "plot_encoder_distributions",
    "plot_training_convergence",
    "plot_encoder_scatter",
    "plot_encoder_strip_by_split",
    "plot_encoder_by_target",
    "plot_anchor_aligned_encoding_heatmap",
    "plot_comparison_heatmap",
    "plot_cross_encoding_heatmap",
    "plot_gradiend_feature_cross_encoding_heatmap",
    "plot_similarity_heatmap",
    "plot_similarity_heatmap_with_correlation",
    "compute_topk_sets",
    "plot_topk_venn",
    "plot_topk_overlap_heatmap",
    "plot_topk_overlap_venn",
]
