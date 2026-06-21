"""Dependency-light helpers for encoding gradient datasets into row dicts."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from gradiend.util.logging import get_logger

logger = get_logger(__name__)


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
    _tqdm_kw = dict(
        desc="Encoding",
        total=total,
        leave=False,
        ncols=80,
        dynamic_ncols=False,
        ascii=True,
        mininterval=0.5,
        position=0,
        disable=not sys.stderr.isatty(),
    )
    for entry in tqdm(dataset, **_tqdm_kw):
        grad = entry["source"]
        label = entry["label"]
        encoded_val = model_with_gradiend.encode(grad, return_float=True)
        input_type = getattr(dataset, "source", None)
        factual_id = entry.get("factual_id")
        counterfactual_id = entry.get("alternative_id")
        transition_id = entry.get("feature_class_id")
        if input_type == "alternative":
            source_id = counterfactual_id
        elif input_type == "diff":
            source_id = transition_id
        else:
            source_id = factual_id
        row: Dict[str, Any] = {
            "encoded": encoded_val,
            "label": float(label),
            "source_id": source_id,
            "target_id": counterfactual_id,
            "factual_id": factual_id,
            "counterfactual_id": counterfactual_id,
            "transition_id": transition_id,
            "feature_class_id": transition_id,
            "input_type": input_type,
            "eval_group": entry.get("eval_group"),
        }
        for token_key in ("factual_token", "alternative_token"):
            if entry.get(token_key) is not None:
                row[token_key] = entry[token_key]
        if entry.get("data_split") is not None:
            row["data_split"] = entry["data_split"]
        if entry.get("neutral_variant") is not None:
            row["neutral_variant"] = entry["neutral_variant"]
        if row_extractor is not None:
            try:
                extra = row_extractor(entry)
                if isinstance(extra, dict) and extra:
                    row.update(extra)
            except Exception as e:
                logger.warning("Row extractor failed: %s", e)
        rows.append(row)
    return rows


__all__ = ["encode_dataset_to_rows"]
