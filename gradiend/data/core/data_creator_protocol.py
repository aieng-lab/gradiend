"""
Protocol for data creators: shared interface for TextPredictionDataCreator and TextClassificationDataCreator.

Both implement generate_training_data and generate_neutral_data with consistent parameters.
Common parameters (e.g. splitting) are part of the protocol so implementations stay aligned.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import pandas as pd


@runtime_checkable
class DataCreator(Protocol):
    """Protocol for data creators that produce training and neutral datasets.

    Implementations such as ``TextPredictionDataCreator`` and
    ``TextClassificationDataCreator`` may return either one DataFrame or a
    per-class mapping, but they share the high-level generation methods below.
    """

    def generate_training_data(
        self,
        *,
        output: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_col: str = "split",
        min_rows_for_split: int = 0,
        **kwargs: Any,
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Generate labeled training data.

        Args:
            output: Optional path where generated data should be written.
            train_ratio: Fraction assigned to the train split.
            val_ratio: Fraction assigned to the validation split.
            test_ratio: Fraction assigned to the test split.
            split_col: Column name used for split labels.
            min_rows_for_split: Minimum row count required before split
                assignment is attempted.
            **kwargs: Implementation-specific generation options.

        Returns:
            Either a single training DataFrame or a mapping from class id to
            class-specific training DataFrame, depending on the creator.

        Raises:
            ValueError: Implementations may raise when inputs are malformed,
                split ratios are invalid, or too little data exists.
        """
        ...

    def generate_neutral_data(
        self,
        *,
        base_data: Optional[Union[str, pd.DataFrame, list]] = None,
        max_size: Optional[int] = None,
        output: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate neutral data.

        Args:
            base_data: Optional source override for neutral examples.
            max_size: Optional maximum number of neutral examples.
            output: Optional path where generated neutral data should be written.
            **kwargs: Implementation-specific generation options.

        Returns:
            A DataFrame containing neutral examples. Text creators should include
            at least a ``"text"`` column.

        Raises:
            ValueError: Implementations may raise when the source is malformed
                or no suitable neutral rows can be generated.
        """
        ...
