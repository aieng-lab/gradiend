"""
FeatureLearningDefinition protocols: interfaces for definition components.

This module defines protocols for the key components of a FeatureLearningDefinition (data provider,
evaluator, feature analyzer), allowing for better type hints and extensibility.
"""

from typing import Protocol, Optional, Dict, Any, Union
from torch.utils.data import Dataset


class DataProvider(Protocol):
    """
    Protocol for providing training data.

    Implementations should create datasets suitable for GRADIEND training.
    """

    def create_training_data(
        self,
        tokenizer: Any,
        split: str = 'train',
        **kwargs
    ) -> Dataset:
        """
        Create training dataset.

        Args:
            tokenizer: Tokenizer for the model
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments

        Returns:
            Dataset suitable for GRADIEND training
        """
        ...


class Evaluator(Protocol):
    """
    Protocol for evaluating GRADIEND models.

    Implementations should evaluate GRAIDEND encoder performance and return metrics.
    """

    def evaluate_encoder(
        self,
        model_with_gradiend: Any,
        eval_data: Any,
        eval_batch_size: int = 32,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate encoder on eval data and return metrics.

        Args:
            model_with_gradiend: ModelWithGradiend instance to evaluate
            eval_data: GradientTrainingDataset (gradients + labels)
            eval_batch_size: Batch size for evaluation
            **kwargs: Additional arguments

        Returns:
            Dictionary with evaluation metrics and results
        """
        ...
