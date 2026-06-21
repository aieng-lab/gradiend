"""
Text modality utilities shared across prediction and classification trainers.

Trainers are not re-exported at this package level to keep imports lightweight.
Use ``gradiend`` top-level exports or import from the modality subpackage, e.g.::

    from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
    from gradiend.trainer.text.classification.trainer import TextClassificationTrainer
"""

from gradiend.trainer.text.common import (
    TextModelWithGradiend,
    TextBatchedDatasetBase,
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
