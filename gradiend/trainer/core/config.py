"""
Core training configuration constants for GRADIEND.

This module defines shared constants used across training components,
including source/target keywords for gradient computation.
"""

from gradiend.model._source_target import SOURCE_TARGET_KEYWORDS, validate_source_target

# Keywords that require factual gradient computation
factual_computation_required_keywords = {'factual', 'diff'}

# Keywords that require alternative gradient computation
alternative_computation_required_keywords = {'alternative', 'diff'}

# All valid source/target keywords (including None for optional computation)
source_target_keywords = {None} | SOURCE_TARGET_KEYWORDS

__all__ = [
    "SOURCE_TARGET_KEYWORDS",
    "validate_source_target",
    "factual_computation_required_keywords",
    "alternative_computation_required_keywords",
    "source_target_keywords",
]
