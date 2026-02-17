"""
Unified schema constants for text prediction.
"""

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
