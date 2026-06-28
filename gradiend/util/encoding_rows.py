"""Dependency-light helpers for encoding gradient datasets into row dicts."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from gradiend.util.logging import get_logger
from gradiend.util.tqdm_utils import gradiend_tqdm

logger = get_logger(__name__)


def gradient_entry_to_encoder_row(
    entry: Dict[str, Any],
    *,
    encoded: float,
    input_type: Optional[str] = "factual",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one encoder-analysis row from a gradient-dataset entry."""
    if input_type == "alternative":
        source_id = entry.get("alternative_id")
        source_token = entry.get("alternative_token")
    elif input_type == "diff":
        source_id = entry.get("feature_class_id")
        source_token = None
    else:
        source_id = entry.get("factual_id")
        source_token = entry.get("factual_token")
    row: Dict[str, Any] = {
        "encoded": encoded,
        "label": float(entry.get("label", 0.0)),
        "source_id": source_id,
        "target_id": entry.get("alternative_id"),
        "factual_id": entry.get("factual_id"),
        "counterfactual_id": entry.get("alternative_id"),
        "transition_id": entry.get("feature_class_id"),
        "feature_class_id": entry.get("feature_class_id"),
        "input_type": input_type,
    }
    if source_token is not None:
        row["source_token"] = source_token
    for token_key in ("factual_token", "alternative_token"):
        if entry.get(token_key) is not None:
            row[token_key] = entry[token_key]
    for text_key in ("text", "template", "input_text", "display_text"):
        if entry.get(text_key) is not None:
            row[text_key] = entry[text_key]
    if entry.get("template") is not None and row.get("masked") is None:
        row["masked"] = entry["template"]
    if entry.get("data_split") is not None:
        row["data_split"] = entry["data_split"]
    if entry.get("neutral_variant") is not None:
        row["neutral_variant"] = entry["neutral_variant"]
    if overrides:
        row.update(overrides)
    return row


def encode_dataset_to_rows(
    model_with_gradiend: Any,
    dataset: Any,
    row_extractor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Encode a GradientTrainingDataset and return per-row dicts for building DataFrames.

    Each row has: encoded, label, source_id, target_id, plus optional fields
    provided by row_extractor (modality-specific, e.g. text).
    Used by EncoderEvaluator and by callers (e.g. _analyze_encoder) when training_rows
    are not available from cache.
    """
    rows: List[Dict[str, Any]] = []
    try:
        total = len(dataset)
    except (TypeError, AttributeError):
        total = None
    for entry in gradiend_tqdm(
        dataset,
        desc="Encoding",
        total=total,
        leave=False,
        ncols=80,
        position=0,
    ):
        grad = entry["source"]
        label = entry["label"]
        encoded_val = model_with_gradiend.encode(grad, return_float=True)
        input_type = getattr(dataset, "source", None)
        row = gradient_entry_to_encoder_row(
            entry,
            encoded=encoded_val,
            input_type=input_type,
        )
        row["eval_group"] = entry.get("eval_group")
        if row_extractor is not None:
            try:
                extra = row_extractor(entry)
                if isinstance(extra, dict) and extra:
                    row.update(extra)
            except Exception as e:
                logger.warning("Row extractor failed: %s", e)
        rows.append(row)
    return rows


__all__ = ["encode_dataset_to_rows", "gradient_entry_to_encoder_row"]
