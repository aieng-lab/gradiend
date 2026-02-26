from typing import Optional, Dict

from .pairwise_heatmap import plot_topk_overlap_heatmap
from .venn_ import plot_topk_overlap_venn


def plot_topk_overlap(
    models: Dict[str, object],
    topk: int = 100,
    part: str = "decoder-weight",
    output_path: Optional[str] = None,
    show: bool = False,
    **kwargs
):
    """Plot top-k neuron overlap between models. Uses Venn diagram for 2-6 models, heatmap for more."""
    if len(models) <= 6:
        venn_kw = {k: v for k, v in kwargs.items() if k in ("figsize", "circle_names_fontsize", "region_counts_fontsize", "patch_linewidth", "alpha", "title")}
        return plot_topk_overlap_venn(models, topk=topk, part=part, output_path=output_path, show=show, **venn_kw)
    return plot_topk_overlap_heatmap(models, topk=topk, part=part, output_path=output_path, show=show, **kwargs)


__all__ = [
    "plot_topk_overlap",
    "plot_topk_overlap_venn",
    "plot_topk_overlap_heatmap",
]
