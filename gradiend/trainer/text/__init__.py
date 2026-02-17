"""
Text modality: prediction (MLM/CLM) and classification trainers for GRADIEND.
"""

from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer, TextPredictionConfig
from gradiend.trainer.text.prediction import (
    TextBatchedDataset,
    TextTrainingDataset,
    TextPredictionModelWithGradiend,
    DecoderModelWithMLMHead,
    train_mlm_head,
    create_masked_pair_from_text,
)
from gradiend.trainer.text.common import (
    TextModelWithGradiend,
    TextBatchedDatasetBase,
    AutoModelForLM,
    AutoTokenizerForLM,
    InstructTokenizerWrapper,
)

__all__ = [
    "TextPredictionTrainer",
    "TextPredictionConfig",
    "TextBatchedDataset",
    "TextTrainingDataset",
    "TextPredictionModelWithGradiend",
    "DecoderModelWithMLMHead",
    "train_mlm_head",
    "create_masked_pair_from_text",
    "TextModelWithGradiend",
    "TextBatchedDatasetBase",
    "AutoModelForLM",
    "AutoTokenizerForLM",
    "InstructTokenizerWrapper",
]
