"""
GRADIEND: Feature Learning within Neural Networks

GRADIEND is a method for learning features within neural networks
by training an encoder-decoder architecture on gradients.

Public API (from gradiend):
    - Model: GradiendModel, ParamMappedGradiendModel, ModelWithGradiend
    - Data: TextFilterConfig, TextPredictionDataCreator, TextPreprocessConfig,
      SpacyTagSpec, preprocess_texts, resolve_base_data
    - Trainer: TextPredictionTrainer, TextPredictionConfig, TrainingArguments,
      load_training_stats, GradientTrainingDataset, TextGradientTrainingDataset,
      create_model_with_gradiend
    - Logging: setup_logging, get_logger

Sub-packages (use when you need modality-specific or internal APIs):
    - gradiend.trainer: Trainer, PrePruneConfig, PostPruneConfig, callbacks, etc.
    - gradiend.trainer.text: TextModelWithGradiend, TextBatchedDataset, etc.
    - gradiend.evaluator: EncoderEvaluator, DecoderEvaluator, Evaluator
    - gradiend.visualizer: Visualizer, plot_* functions
    - gradiend.data: same as top-level data API (alternative import path)

For experimental features (analysis, plotting, LaTeX export), install with:
    pip install gradiend[recommended]
"""

__version__ = "1.0.0"

# Core model classes
from gradiend.model import GradiendModel, ParamMappedGradiendModel, ModelWithGradiend

# High-level data API (filter config, data creator, preprocess)
from gradiend.data import (
    TextFilterConfig,
    TextPredictionDataCreator,
    TextPreprocessConfig,
    SpacyTagSpec,
    preprocess_texts,
    resolve_base_data,
)

# Text prediction trainer (high-level)
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer, TextPredictionConfig

# Logging
from gradiend.util.logging import setup_logging, get_logger

# Training
from gradiend.trainer import (
    load_training_stats,
    TrainingArguments,
    TrainerConfig,
    GradientTrainingDataset,
    TextGradientTrainingDataset,
    create_model_with_gradiend,
    PrePruneConfig,
    PostPruneConfig
)

__all__ = [
    # Core model
    "GradiendModel",
    "ParamMappedGradiendModel",
    "ModelWithGradiend",
    # Data (high-level)
    "TextFilterConfig",
    "TextPredictionDataCreator",
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
    # Training
    "load_training_stats",
    "TrainingArguments",
    "GradientTrainingDataset",
    "TextGradientTrainingDataset",
    "create_model_with_gradiend",
]
