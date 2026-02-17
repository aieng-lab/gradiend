"""Text-specific data modules."""

from gradiend.data.text.filter_config import SpacyTagSpec, TextFilterConfig
from gradiend.data.text.preprocess import (
    TextPreprocessConfig,
    iter_sentences_from_texts,
    preprocess_texts,
)

__all__ = [
    "SpacyTagSpec",
    "TextFilterConfig",
    "TextPreprocessConfig",
    "iter_sentences_from_texts",
    "preprocess_texts",
]
