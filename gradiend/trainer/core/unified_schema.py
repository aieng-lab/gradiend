"""
Unified schema constants for text-prediction training data.

Canonical location for the unified dataframe contract used by trainers,
suites, and data utilities.
"""

from typing import Optional

UNIFIED_MASKED = "masked"
UNIFIED_SPLIT = "split"
UNIFIED_FACTUAL_CLASS = "factual_class"
UNIFIED_ALTERNATIVE_CLASS = "alternative_class"
UNIFIED_FACTUAL = "factual"
UNIFIED_ALTERNATIVE = "alternative"
UNIFIED_TRANSITION = "transition"

REQUIRED_UNIFIED = {
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_ALTERNATIVE,
    UNIFIED_TRANSITION,
}


def transition_id(factual_class: str, alternative_class: str) -> str:
    """Canonical transition string for (factual_class, alternative_class)."""
    return f"{factual_class}→{alternative_class}"


def normalize_transition_id(value: object) -> Optional[str]:
    """Normalize legacy ASCII ``->`` ids to the canonical Unicode arrow form."""
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "→" in text:
        return text
    if "->" in text:
        left, _, right = text.partition("->")
        if left and right:
            return transition_id(left, right)
    return text
