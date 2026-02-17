"""
Common text utilities shared by prediction and classification.

Exports: TextModelWithGradiend (base), TextBatchedDatasetBase, loading, lm_eval.
"""

from gradiend.trainer.text.common.model_base import TextModelWithGradiend
from gradiend.trainer.text.common.dataset_base import TextBatchedDatasetBase
from gradiend.trainer.text.common.loading import (
    AutoModelForLM,
    AutoTokenizerForLM,
    InstructTokenizerWrapper,
)

__all__ = [
    "TextModelWithGradiend",
    "TextBatchedDatasetBase",
    "AutoModelForLM",
    "AutoTokenizerForLM",
    "InstructTokenizerWrapper",
]
