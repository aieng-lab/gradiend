"""Tests for non-convergence plot label helpers."""

from gradiend import TrainingArguments
from gradiend.visualizer.labels import (
    NON_CONVERGENCE_MARKER,
    converged_from_run_info,
    format_label_with_convergence,
    format_model_labels_with_convergence,
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
    assert format_label_with_convergence("run_a", converged=False) == f"run_a {NON_CONVERGENCE_MARKER}"
    assert format_label_with_convergence("run_a", converged=False, highlight_non_convergence=False) == "run_a"
    assert format_label_with_convergence("run_a", converged=True) == "run_a"
    assert format_label_with_convergence("run_a", converged=None) == "run_a"
    already = f"run_a {NON_CONVERGENCE_MARKER}"
    assert format_label_with_convergence(already, converged=False) == already


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


def test_resolve_plot_title_with_convergence():
    trainer = _TrainerStub(run_id="my_run", converged=False)
    assert resolve_plot_title_with_convergence(True, trainer=trainer) == f"my_run {NON_CONVERGENCE_MARKER}"
    assert resolve_plot_title_with_convergence(True, trainer=trainer, highlight_non_convergence=False) == "my_run"
    assert resolve_plot_title_with_convergence(False, trainer=trainer) is False
    trainer_ok = _TrainerStub(run_id="ok", converged=True)
    assert resolve_plot_title_with_convergence(True, trainer=trainer_ok) == "ok"


def test_format_model_labels_with_convergence():
    class _Model:
        name_or_path = "/tmp/model"

    labels = format_model_labels_with_convergence(
        ["a", "b"],
        models={"a": _Model(), "b": _Model()},
        converged_by_id={"a": True, "b": False},
        highlight_non_convergence=True,
    )
    assert labels["a"] == "a"
    assert labels["b"] == f"b {NON_CONVERGENCE_MARKER}"


def test_plot_functions_expose_highlight_non_convergence_param():
    """Plot entry points must accept highlight_non_convergence for API consistency."""
    import inspect

    from gradiend.visualizer.heatmaps import (
        plot_anchor_aligned_encoding_heatmap,
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
        plot_anchor_aligned_encoding_heatmap,
        plot_topk_overlap_heatmap,
        plot_topk_overlap_venn,
    ):
        assert "highlight_non_convergence" in inspect.signature(fn).parameters, fn.__name__
