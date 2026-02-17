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
from gradiend.visualizer.topk.venn_ import (
    compute_topk_sets,
    plot_topk_venn,
    plot_topk_overlap_venn,
)

__all__ = [
    "Visualizer",
    "plot_encoder_distributions",
    "plot_training_convergence",
    "plot_encoder_scatter",
    "compute_topk_sets",
    "plot_topk_venn",
    "plot_topk_overlap_venn",
]
