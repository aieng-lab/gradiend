"""
Data module: create training and neutral datasets from base corpora.

Use TextPredictionDataCreator for the public text-prediction workflow.
Experimental data creators live in their own subpackages.
"""

from gradiend.data.core import DataCreator, resolve_base_data
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
    "DataCreator",
    "preprocess_texts",
    "resolve_base_data",
]
