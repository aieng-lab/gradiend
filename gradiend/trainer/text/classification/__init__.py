"""
Text classification trainer and data for GRADIEND.

Provides TextClassificationTrainer using AutoModelForSequenceClassification,
unified (factual, alternative) data, and label_fn-based data creation.
"""

from gradiend.trainer.text.classification.config import TextClassificationConfig
from gradiend.trainer.text.classification.trainer import TextClassificationTrainer
from gradiend.trainer.text.classification.model_with_gradiend import TextClassificationModelWithGradiend
from gradiend.trainer.text.classification.evaluator import (
    ClassificationEvaluator,
    ClassificationEncoderEvaluator,
)

__all__ = [
    "TextClassificationConfig",
    "TextClassificationTrainer",
    "TextClassificationModelWithGradiend",
    "ClassificationEvaluator",
    "ClassificationEncoderEvaluator",
]
