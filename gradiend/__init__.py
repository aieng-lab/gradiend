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
    - Comparison: compute_similarity_matrix, compute_cross_encoding_matrix,
      compute_anchor_aligned_encoding_matrix, compute_gradiend_feature_cross_encoding_matrix,
      compute_gradiend_transition_cross_encoding_matrix
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
    __version__ = "0.1.0"  # editable install before package is installed

__all__ = [
    # Core model
    "GradiendModel",
    "ParamMappedGradiendModel",
    "ModelWithGradiend",
    # Data (high-level)
    "TextFilterConfig",
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
    # Logging
    "setup_logging",
    "get_logger",
    "compute_similarity_matrix",
    "compute_grouped_similarity_matrices",
    "compute_cross_encoding_matrix",
    "compute_anchor_aligned_encoding_matrix",
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
]

_IMPORT_ERROR = None

try:
    # Core model classes
    from gradiend.model import GradiendModel, ParamMappedGradiendModel, ModelWithGradiend

    # High-level data API (filter config, data creators, preprocess)
    from gradiend.data import (
        TextFilterConfig,
        TextPredictionDataCreator,
        DataCreator,
        TextPreprocessConfig,
        SpacyTagSpec,
        preprocess_texts,
        resolve_base_data,
    )

    # Text prediction trainer (high-level)
    from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer, TextPredictionConfig

    # Logging
    from gradiend.util.logging import setup_logging, get_logger
    from gradiend.comparison import (
        compute_similarity_matrix,
        compute_grouped_similarity_matrices,
        compute_cross_encoding_matrix,
        compute_anchor_aligned_encoding_matrix,
        compute_gradiend_feature_cross_encoding_matrix,
        compute_gradiend_transition_cross_encoding_matrix,
    )

    # Visualization functions
    from gradiend.visualizer import (
        plot_gradiend_feature_cross_encoding_heatmap,
        plot_gradiend_transition_cross_encoding_heatmap,
        plot_comparison_heatmap,
        plot_cross_encoding_heatmap,
        plot_similarity_heatmap,
        plot_topk_overlap_heatmap,
        plot_topk_overlap_venn,
        check_plot_environment,
    )

    # Training
    from gradiend.trainer import (
        load_training_stats,
        set_seed,
        TrainerSuite,
        TrainerCollection,
        PositiveTrainerSuite,
        SymmetricTrainerSuite,
        SuitePairDefinition,
        PositiveFeatureDefinition,
        TrainingArguments,
        TransitionSpec,
        pair,
        identity,
        TrainerConfig,
        GradientTrainingDataset,
        TextGradientTrainingDataset,
        create_model_with_gradiend,
        PrePruneConfig,
        PostPruneConfig,
    )
except Exception as exc:
    _IMPORT_ERROR = exc


def __getattr__(name):
    if name in __all__ and _IMPORT_ERROR is not None:
        raise ImportError(
            "Importing gradiend's full public API failed. "
            "This usually means optional runtime dependencies such as torch are unavailable. "
            "Import the needed submodule directly or fix the environment."
        ) from _IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
