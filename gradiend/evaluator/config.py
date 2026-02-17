"""
Encoder Evaluation Configuration.

This module defines EncoderEvalConfig for configuring encoder evaluation modes.
"""

from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd

@dataclass
class EncoderEvalConfig:
    """
    Configuration for encoder evaluation.

    This config supports different evaluation modes:
    - neutral_data: Completely neutral data (no feature-related tokens)
    - training_data_neutral_gradients: Training data but with neutral gradients (masking non-feature-related tokens)
    """

    # Neutral data evaluation
    neutral_data: Optional[Union[pd.DataFrame, str]] = None  # DataFrame or path to CSV

    # Training data with neutral masks
    training_data_neutral_gradients: bool = False
    ignore_tokens: Optional[list] = None  # Tokens to ignore when masking

    # General settings
    max_size: Optional[int] = None  # Limit evaluation data size
    split: str = 'test'  # Dataset split to use
