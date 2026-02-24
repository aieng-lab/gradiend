"""
GRADIEND Training Module

This module provides classes and utilities for training GRADIEND models.

Main API:
    - Trainer: HF-like trainer with model at creation time and lazy Evaluator
    - TrainingArguments: Configuration for training
    - load_training_stats: Load correlation, config, best checkpoint from a saved model dir

Core components:
    - train (train_core): Core training loop
    - TextGradientTrainingDataset: Dataset for training
    - create_model_with_gradiend: Factory for model creation
"""

# Trainer
from .trainer import Trainer

# Load training stats from saved model directory
from .core.stats import load_training_stats

# Core training components
from .core.training import train as train_core
from .core.callbacks import (
    TrainingCallback,
    EvaluationCallback,
    CheckpointCallback,
    NormalizationCallback,
    LoggingCallback,
    get_default_callbacks,
)

# Training Arguments (HF-like)
from .core.arguments import TrainingArguments

# Base config (modality-agnostic)
from .config import TrainerConfig

# Pre-prune and post-prune configs and helpers
from .core.pruning import PostPruneConfig, PrePruneConfig, post_prune, pre_prune

# Dataset: modality-agnostic in core; text wrapper in trainer.text.common
from .core.dataset import GradientTrainingDataset
from .text.common.dataset import TextGradientTrainingDataset

# Factory
from .factory import create_model_with_gradiend

__all__ = [
    "Trainer",
    "TrainerConfig",
    "load_training_stats",
    "train_core",
    "TrainingArguments",
    "PostPruneConfig",
    "PrePruneConfig",
    "post_prune",
    "pre_prune",
    "GradientTrainingDataset",
    "TextGradientTrainingDataset",
    "create_model_with_gradiend",
    "TrainingCallback",
    "EvaluationCallback",
    "CheckpointCallback",
    "NormalizationCallback",
    "LoggingCallback",
    "get_default_callbacks",
]