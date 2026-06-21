"""
Common text utilities shared by prediction and classification.

Exports: TextModelWithGradiend (base), TextBatchedDatasetBase, loading, lm_eval, dual_head_model.
"""

from gradiend.trainer.text.common.model_base import TextModelWithGradiend
from gradiend.trainer.text.common.dataset_base import TextBatchedDatasetBase
from gradiend.trainer.text.common.loading import (
    AutoModelForLM,
    AutoTokenizerForLM,
    InstructTokenizerWrapper,
)
from gradiend.trainer.text.common.dual_head_model import (
    attach_lm_head_to_classification_model,
    build_dual_head_sequence_model,
    try_build_dual_head_from_base_path,
)

__all__ = [
    "TextModelWithGradiend",
    "TextBatchedDatasetBase",
    "AutoModelForLM",
    "AutoTokenizerForLM",
    "InstructTokenizerWrapper",
    "attach_lm_head_to_classification_model",
    "build_dual_head_sequence_model",
    "try_build_dual_head_from_base_path",
]
