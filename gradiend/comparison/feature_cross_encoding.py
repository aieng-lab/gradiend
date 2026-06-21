"""Dense GRADIEND by feature-class cross-encoding matrices."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from gradiend.comparison.common import _safe_float
from gradiend.comparison.cross_encoding import _load_eval_model_for_trainer
from gradiend.util.encoding_rows import encode_dataset_to_rows

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


def _normalize_split_name(value: object) -> str:
    value = str(value).strip().lower()
    aliases = {
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
    }
    return aliases.get(value, value)


def collect_unified_test_rows_by_feature_class(
    trainers: Dict[str, object],
    *,
    split: str = "test",
) -> Dict[str, pd.DataFrame]:
    """
    Merge test-split unified rows from all trainers, grouped by ``factual_class``.

    Used to build a dense GRADIEND × feature-class cross-encoding matrix: every
    trainer is evaluated on the same per-class eval snippets.
    """
    resolved_split = _normalize_split_name(split)
    frames_by_class: Dict[str, List[pd.DataFrame]] = {}
    for trainer_id, trainer in trainers.items():
        ensure = getattr(trainer, "_ensure_data", None)
        if callable(ensure):
            ensure()
        combined = getattr(trainer, "combined_data", None)
        if not isinstance(combined, pd.DataFrame) or combined.empty:
            continue
        missing = REQUIRED_UNIFIED - set(combined.columns)
        if missing:
            raise ValueError(
                f"Trainer {trainer_id!r} combined_data missing unified columns: {sorted(missing)}"
            )
        split_mask = combined[UNIFIED_SPLIT].astype(str).map(_normalize_split_name) == resolved_split
        test_df = combined[split_mask].copy()
        if test_df.empty:
            continue
        for class_id, group in test_df.groupby(test_df[UNIFIED_FACTUAL_CLASS].astype(str)):
            frames_by_class.setdefault(str(class_id), []).append(group)
    out: Dict[str, pd.DataFrame] = {}
    for class_id, frames in frames_by_class.items():
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(
            subset=[c for c in merged.columns if c in REQUIRED_UNIFIED],
            keep="first",
        ).reset_index(drop=True)
        out[class_id] = merged
    return out


def _build_probe_pairs_from_unified_df(
    trainer: object,
    unified_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    target_classes = getattr(trainer, "target_classes", None) or []
    if len(target_classes) != 2:
        raise ValueError(
            f"Trainer {getattr(trainer, 'run_id', trainer)!r} must have exactly 2 target_classes"
        )
    class_pair = (str(target_classes[0]), str(target_classes[1]))
    pairs: List[Dict[str, Any]] = []
    for _, row in unified_df.iterrows():
        src = str(row[UNIFIED_FACTUAL_CLASS])
        tgt = str(row[UNIFIED_ALTERNATIVE_CLASS])
        if src == class_pair[0]:
            label = 1.0
        elif src == class_pair[1]:
            label = -1.0
        else:
            label = 0.0
        entry: Dict[str, Any] = {
            "masked": row[UNIFIED_MASKED],
            "factual": row[UNIFIED_FACTUAL],
            "alternative": row[UNIFIED_ALTERNATIVE],
            "factual_id": src,
            "alternative_id": tgt,
            "label": label,
            "feature_class_id": f"{src}->{tgt}",
        }
        if UNIFIED_SPLIT in row.index and pd.notna(row[UNIFIED_SPLIT]):
            entry[UNIFIED_SPLIT] = row[UNIFIED_SPLIT]
        pairs.append(entry)
    return pairs


def _mean_encoded_for_feature_class(
    trainer: object,
    model: object,
    class_df: pd.DataFrame,
    *,
    max_size: Optional[int] = None,
) -> Optional[float]:
    if class_df is None or class_df.empty:
        return None
    df = class_df
    if max_size is not None and len(df) > max_size:
        df = df.sample(n=max_size, random_state=42).reset_index(drop=True)
    pairs = _build_probe_pairs_from_unified_df(trainer, df)
    if not pairs:
        return None
    grad_ds = trainer.create_gradient_training_dataset(pairs, model)
    rows = encode_dataset_to_rows(model, grad_ds)
    if not rows:
        return None
    values = [_safe_float(r.get("encoded")) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def compute_gradiend_feature_cross_encoding_matrix(
    trainers: Dict[str, object],
    feature_classes: Sequence[str],
    *,
    trainer_order: Optional[Sequence[str]] = None,
    eval_by_class: Optional[Dict[str, pd.DataFrame]] = None,
    split: str = "test",
    max_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute a dense GRADIEND by feature-class cross-encoding matrix.

    Row *i* is a trained GRADIEND; column *j* is a feature class. Cell ``(i, j)``
    is the mean encoded value when GRADIEND *i* encodes shared eval snippets for
    class *j*. When ``eval_by_class`` is omitted, snippets are collected from the
    trainers' unified data for the requested split.

    Args:
        trainers: Mapping from trainer id to trainer object. Trainers must be
            able to load a model and create a gradient training dataset for the
            generated probe pairs.
        feature_classes: Column order for feature classes to evaluate.
        trainer_order: Optional row order. Unknown ids are ignored; at least one
            valid id must remain.
        eval_by_class: Optional precomputed mapping from feature class to a
            unified-data DataFrame. If omitted, it is built from ``trainers`` via
            :func:`collect_unified_test_rows_by_feature_class`.
        split: Split used when collecting unified eval rows.
        max_size: Optional maximum examples per feature class. If set, rows are
            sampled with a fixed random seed before encoding.

    Returns:
        A payload with ``measure``, ``model_ids``, ``column_ids``, ``rows``,
        ``columns``, ``matrix``, ``n_matrix``, ``split``, ``max_size``, and
        ``eval_classes_found``. Missing classes produce ``NaN`` cells and count
        ``0``.

    Raises:
        ValueError: If no trainers/classes are provided, ``trainer_order`` has
            no valid trainer ids, trainer unified data is malformed, or a
            trainer is not binary where probe-pair construction requires it.
    """
    if not trainers:
        raise ValueError("trainers must be a non-empty dict")
    classes = [str(c) for c in feature_classes]
    if not classes:
        raise ValueError("feature_classes must be non-empty")
    order = [str(t) for t in (trainer_order or list(trainers.keys())) if str(t) in trainers]
    if not order:
        raise ValueError("trainer_order produced no valid trainer ids")
    if eval_by_class is None:
        eval_by_class = collect_unified_test_rows_by_feature_class(trainers, split=split)

    matrix: List[List[float]] = []
    n_matrix: List[List[int]] = []
    for trainer_id in order:
        trainer = trainers[trainer_id]
        model = _load_eval_model_for_trainer(trainer)
        row_values: List[float] = []
        row_counts: List[int] = []
        try:
            for class_id in classes:
                class_df = eval_by_class.get(class_id)
                if class_df is None or class_df.empty:
                    row_values.append(float("nan"))
                    row_counts.append(0)
                    continue
                sample_n = min(len(class_df), max_size) if max_size is not None else len(class_df)
                mean_val = _mean_encoded_for_feature_class(
                    trainer,
                    model,
                    class_df,
                    max_size=max_size,
                )
                row_values.append(float(mean_val) if mean_val is not None else float("nan"))
                row_counts.append(int(sample_n) if mean_val is not None else 0)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        matrix.append(row_values)
        n_matrix.append(row_counts)

    return {
        "measure": "gradiend_feature_cross_encoding_mean",
        "model_ids": order,
        "column_ids": classes,
        "rows": order,
        "columns": classes,
        "matrix": matrix,
        "n_matrix": n_matrix,
        "split": split,
        "max_size": max_size,
        "eval_classes_found": sorted(eval_by_class.keys()),
    }
