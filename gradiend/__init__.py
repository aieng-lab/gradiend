"""
GRADIEND: Feature Learning within Neural Networks (https://arxiv.org/abs/2502.01406)

GRADIEND is a method for learning features within neural networks
by training an encoder-decoder architecture on gradients.

Public API (from gradiend):

    - Model: GradiendModel, ParamMappedGradiendModel, ModelWithGradiend
    - Data: TextFilterConfig, TextPredictionDataCreator, DataCreator, TextPreprocessConfig,

      SpacyTagSpec, preprocess_texts, resolve_base_data

    - Trainer: TextPredictionTrainer, TextPredictionConfig, TrainingArguments,

      load_training_stats, GradientTrainingDataset, TextGradientTrainingDataset,
      create_model_with_gradiend

    - Comparison: compute_similarity_matrix, compute_trainer_pair_encoding_matrix,

      compute_anchor_aligned_encoding_matrix, compute_gradiend_feature_cross_encoding_matrix,
      compute_gradiend_transition_cross_encoding_matrix,
      build_cross_task_encoder_summary

    - Logging: setup_logging, get_logger

Sub-packages (use when you need modality-specific or internal APIs):

    - gradiend.trainer: Trainer, PrePruneConfig, PostPruneConfig, callbacks, etc.
    - gradiend.trainer.text: TextModelWithGradiend, TextBatchedDataset, etc.
    - gradiend.evaluator: EncoderEvaluator, DecoderEvaluator, Evaluator
    - gradiend.visualizer: Visualizer, plot_encoder_distributions, plot_topk_overlap_heatmap, plot_topk_overlap_venn, etc.
    - gradiend.data: same as top-level data API (alternative import path)

For experimental features (analysis, plotting, LaTeX export), install with:
    pip install gradiend[recommended]

    or

    pip install gradiend # minimal requirements
"""

from importlib.metadata import PackageNotFoundError, version

from gradiend.util.hf_env import configure_hf_download_env
from gradiend.util.tqdm_utils import patch_sys_stderr_for_tqdm

configure_hf_download_env()
patch_sys_stderr_for_tqdm()

try:
    __version__ = version("gradiend")
except PackageNotFoundError:
    __version__ = "0.2.3"  # editable install before package is installed

__all__ = [
    # Core model
    "GradiendModel",
    "ParamMappedGradiendModel",
    "ModelWithGradiend",
    # Data (high-level)
    "TextFilterConfig",
    "PairedTextFilterConfig",
    "TextPredictionDataCreator",
    "DataCreator",
    "TextPreprocessConfig",
    "SpacyTagSpec",
    "preprocess_texts",
    "resolve_base_data",
    # Trainers and configs
    "TextPredictionTrainer",
    "TextPredictionConfig",
    "TrainerConfig",
    "PrePruneConfig",
    "PostPruneConfig",
    "LRSearchConfig",
    # Logging
    "setup_logging",
    "get_logger",
    "compute_similarity_matrix",
    "compute_grouped_similarity_matrices",
    "compute_trainer_pair_encoding_matrix",
    "compute_anchor_aligned_encoding_matrix",
    "build_cross_task_encoder_summary",
    "compute_gradiend_feature_cross_encoding_matrix",
    "compute_gradiend_transition_cross_encoding_matrix",
    # Training
    "load_training_stats",
    "set_seed",
    "TrainerSuite",
    "TrainerCollection",
    "PositiveTrainerSuite",
    "SymmetricTrainerSuite",
    "SuitePairDefinition",
    "PositiveFeatureDefinition",
    "TrainingArguments",
    "TransitionSpec",
    "pair",
    "identity",
    "GradientTrainingDataset",
    "TextGradientTrainingDataset",
    "create_model_with_gradiend",
    # Visualization
    "plot_comparison_heatmap",
    "plot_gradiend_feature_cross_encoding_heatmap",
    "plot_gradiend_transition_cross_encoding_heatmap",
    "plot_cross_encoding_heatmap",
    "plot_similarity_heatmap",
    "plot_topk_overlap_heatmap",
    "plot_topk_overlap_venn",
    "check_plot_environment",
    "configure_plot_style",
    "PlotStyleConfig",
    "PlotStyleStatus",
    "format_transition_label",
    "transition_bidi_arrow",
    "transition_directed_arrow",
]

_LAZY_IMPORTS = {
    # Core model classes
    "GradiendModel": ("gradiend.model", "GradiendModel"),
    "ParamMappedGradiendModel": ("gradiend.model", "ParamMappedGradiendModel"),
    "ModelWithGradiend": ("gradiend.model", "ModelWithGradiend"),
    # High-level data API
    "TextFilterConfig": ("gradiend.data", "TextFilterConfig"),
    "PairedTextFilterConfig": ("gradiend.data", "PairedTextFilterConfig"),
    "TextPredictionDataCreator": ("gradiend.data", "TextPredictionDataCreator"),
    "DataCreator": ("gradiend.data", "DataCreator"),
    "TextPreprocessConfig": ("gradiend.data", "TextPreprocessConfig"),
    "SpacyTagSpec": ("gradiend.data", "SpacyTagSpec"),
    "preprocess_texts": ("gradiend.data", "preprocess_texts"),
    "resolve_base_data": ("gradiend.data", "resolve_base_data"),
    # Text prediction trainer
    "TextPredictionTrainer": ("gradiend.trainer.text.prediction.trainer", "TextPredictionTrainer"),
    "TextPredictionConfig": ("gradiend.trainer.text.prediction.trainer", "TextPredictionConfig"),
    # Logging
    "setup_logging": ("gradiend.util.logging", "setup_logging"),
    "get_logger": ("gradiend.util.logging", "get_logger"),
    # Comparison
    "compute_similarity_matrix": ("gradiend.comparison", "compute_similarity_matrix"),
    "compute_grouped_similarity_matrices": ("gradiend.comparison", "compute_grouped_similarity_matrices"),
    "compute_trainer_pair_encoding_matrix": ("gradiend.comparison", "compute_trainer_pair_encoding_matrix"),
    "compute_anchor_aligned_encoding_matrix": ("gradiend.comparison", "compute_anchor_aligned_encoding_matrix"),
    "build_cross_task_encoder_summary": ("gradiend.comparison", "build_cross_task_encoder_summary"),
    "compute_gradiend_feature_cross_encoding_matrix": (
        "gradiend.comparison",
        "compute_gradiend_feature_cross_encoding_matrix",
    ),
    "compute_gradiend_transition_cross_encoding_matrix": (
        "gradiend.comparison",
        "compute_gradiend_transition_cross_encoding_matrix",
    ),
    # Visualization
    "plot_gradiend_feature_cross_encoding_heatmap": (
        "gradiend.visualizer",
        "plot_gradiend_feature_cross_encoding_heatmap",
    ),
    "plot_gradiend_transition_cross_encoding_heatmap": (
        "gradiend.visualizer",
        "plot_gradiend_transition_cross_encoding_heatmap",
    ),
    "plot_comparison_heatmap": ("gradiend.visualizer", "plot_comparison_heatmap"),
    "plot_cross_encoding_heatmap": ("gradiend.visualizer", "plot_cross_encoding_heatmap"),
    "plot_similarity_heatmap": ("gradiend.visualizer", "plot_similarity_heatmap"),
    "plot_topk_overlap_heatmap": ("gradiend.visualizer", "plot_topk_overlap_heatmap"),
    "plot_topk_overlap_venn": ("gradiend.visualizer", "plot_topk_overlap_venn"),
    "check_plot_environment": ("gradiend.visualizer", "check_plot_environment"),
    "configure_plot_style": ("gradiend.visualizer", "configure_plot_style"),
    "PlotStyleConfig": ("gradiend.visualizer", "PlotStyleConfig"),
    "PlotStyleStatus": ("gradiend.visualizer", "PlotStyleStatus"),
    "format_transition_label": ("gradiend.visualizer", "format_transition_label"),
    "transition_bidi_arrow": ("gradiend.visualizer", "transition_bidi_arrow"),
    "transition_directed_arrow": ("gradiend.visualizer", "transition_directed_arrow"),
    # Training
    "load_training_stats": ("gradiend.trainer", "load_training_stats"),
    "set_seed": ("gradiend.trainer", "set_seed"),
    "TrainerSuite": ("gradiend.trainer", "TrainerSuite"),
    "TrainerCollection": ("gradiend.trainer", "TrainerCollection"),
    "PositiveTrainerSuite": ("gradiend.trainer", "PositiveTrainerSuite"),
    "SymmetricTrainerSuite": ("gradiend.trainer", "SymmetricTrainerSuite"),
    "SuitePairDefinition": ("gradiend.trainer", "SuitePairDefinition"),
    "PositiveFeatureDefinition": ("gradiend.trainer", "PositiveFeatureDefinition"),
    "TrainingArguments": ("gradiend.trainer", "TrainingArguments"),
    "TransitionSpec": ("gradiend.trainer", "TransitionSpec"),
    "pair": ("gradiend.trainer", "pair"),
    "identity": ("gradiend.trainer", "identity"),
    "TrainerConfig": ("gradiend.trainer", "TrainerConfig"),
    "GradientTrainingDataset": ("gradiend.trainer", "GradientTrainingDataset"),
    "TextGradientTrainingDataset": ("gradiend.trainer", "TextGradientTrainingDataset"),
    "create_model_with_gradiend": ("gradiend.trainer", "create_model_with_gradiend"),
    "PrePruneConfig": ("gradiend.trainer", "PrePruneConfig"),
    "LRSearchConfig": ("gradiend.trainer", "LRSearchConfig"),
    "PostPruneConfig": ("gradiend.trainer", "PostPruneConfig"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr_name = _LAZY_IMPORTS[name]
        try:
            value = getattr(importlib.import_module(module_name), attr_name)
        except Exception as exc:
            raise ImportError(
                f"Importing gradiend.{name} failed. "
                "This usually means an optional runtime dependency is unavailable. "
                "Import a narrower submodule directly or fix the environment."
            ) from exc
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
