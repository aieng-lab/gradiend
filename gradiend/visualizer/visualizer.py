"""
Visualizer(trainer): evaluation-related plots (encoder distributions, scatter, convergence).

Holds a reference to the trainer; exposes single-model visualization helpers.
Option A: Evaluator holds a Visualizer for evaluation-related plots.
"""

from typing import Any, Dict, Optional

from gradiend.visualizer.encoder_distributions import plot_encoder_distributions as _plot_encoder_distributions
from gradiend.visualizer.convergence import plot_training_convergence as _plot_training_convergence
from gradiend.visualizer.encoder_scatter import plot_encoder_scatter as _plot_encoder_scatter
from gradiend.visualizer.topk.venn_ import (
    compute_topk_sets,
    plot_topk_overlap_venn,
)


class Visualizer:
    """
    Visualizer bound to a trainer. Exposes single-model plots.
    User can subclass to customize plotting behavior.
    """

    def __init__(self, trainer: Any):
        self._trainer = trainer

    @property
    def trainer(self):
        return self._trainer

    def plot_encoder_distributions(self, encoder_df: Optional[Any] = None, **kwargs: Any) -> str:
        """Plot encoder distributions (grouped split violins). Pass encoder_df for self-managed data."""
        return _plot_encoder_distributions(self._trainer, encoder_df=encoder_df, **kwargs)

    def plot_topk_neuron_intersection(
        self,
        models: Optional[Dict[str, Any]] = None,
        topk: int = 100,
        part: str = "decoder-weight",
        **kwargs: Any,
    ) -> Any:
        """Plot top-k neuron intersection. If models is None, uses trainer.get_model()."""
        if models is None:
            model = self._trainer.get_model()
            models = {getattr(model, "name_or_path", "model"): model} if model is not None else {}
        return plot_topk_overlap_venn(models, topk=topk, part=part, **kwargs)

    def plot_training_convergence(self, **kwargs: Any) -> str:
        """Plot training convergence (means by class/feature_class and correlation). Uses trainer for stats."""
        return _plot_training_convergence(trainer=self._trainer, **kwargs)

    def plot_encoder_scatter(
        self,
        encoder_df: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Interactive 1D encoder scatter (jitter x, encoded y), colored by label, with hover. For Jupyter."""
        return _plot_encoder_scatter(trainer=self._trainer, encoder_df=encoder_df, **kwargs)

    @staticmethod
    def compute_topk_sets(models: Dict[str, Any], topk: int = 100, part: str = "decoder-weight"):
        """Compute top-k weight sets for multiple models (intersection/union)."""
        return compute_topk_sets(models, topk=topk, part=part)
