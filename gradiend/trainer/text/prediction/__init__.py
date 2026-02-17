"""
Text prediction modality: MLM/CLM trainers and models.
"""

from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer, TextPredictionConfig
from gradiend.trainer.text.prediction.dataset import TextBatchedDataset, TextTrainingDataset, create_masked_pair_from_text
from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
from gradiend.trainer.text.prediction.decoder_only_mlm import DecoderModelWithMLMHead, train_mlm_head

__all__ = [
    "TextPredictionTrainer",
    "TextPredictionConfig",
    "TextBatchedDataset",
    "TextTrainingDataset",
    "TextPredictionModelWithGradiend",
    "DecoderModelWithMLMHead",
    "train_mlm_head",
    "create_masked_pair_from_text",
]
