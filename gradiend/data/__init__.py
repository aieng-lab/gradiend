"""
Data module: create training and neutral datasets from base corpora.

Use TextPredictionDataCreator to filter and mask base texts based on TextFilterConfig.
"""

from gradiend.data.core import resolve_base_data
from gradiend.data.text import (
    SpacyTagSpec,
    TextFilterConfig,
    TextPreprocessConfig,
    preprocess_texts,
)
from gradiend.data.text.prediction import TextPredictionDataCreator

__all__ = [
    "SpacyTagSpec",
    "TextFilterConfig",
    "TextPreprocessConfig",
    "TextPredictionDataCreator",
    "preprocess_texts",
    "resolve_base_data",
]
