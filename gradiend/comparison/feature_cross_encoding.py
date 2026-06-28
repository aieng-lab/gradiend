"""Dense GRADIEND by feature-class cross-encoding matrices."""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from gradiend.comparison.anchor_aligned import _transition_id_from_row
from gradiend.comparison.common import (
    _aggregate_seed_scores,
    _safe_float,
    _validate_aggregate_dispersion_combo,
    encoder_dataframes_from_summary,
    orient_encoder_df_by_label_correlation,
)
from gradiend.comparison.cross_encoding import _load_eval_model_for_trainer
from gradiend.comparison.encoder_aggregation import (
    aggregate_encoder_dataframes,
    encoder_probe_key_from_row,
    encoder_probe_keys,
)
from gradiend.comparison.seed_policy import (
    analysis_seed_entries,
    comparison_seed_metadata,
    unwrap_trainer,
)
from gradiend.model.utils import is_seq2seq_model
from gradiend.trainer.core.cache_policy import coerce_artifact_use_cache
from gradiend.trainer.core.unified_schema import normalize_transition_id, transition_id
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.util.encoding_rows import encode_dataset_to_rows
from gradiend.util.logging import get_logger
from gradiend.util.paths import invalidate_encoder_metrics_cache, resolve_encoder_analysis_path

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

logger = get_logger(__name__)


def _normalize_split_name(value: object) -> str:
    value = str(value).strip().lower()
    aliases = {
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
    }
    return aliases.get(value, value)


def _iter_trainer_unified_test_frames(
    trainers: Dict[str, object],
    *,
    split: str = "test",
) -> List[pd.DataFrame]:
    resolved_split = _normalize_split_name(split)
    frames: List[pd.DataFrame] = []
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
        if not test_df.empty:
            frames.append(test_df)
    return frames


def collect_unified_test_rows(
    trainers: Dict[str, object],
    *,
    split: str = "test",
    probe_trainers: Optional[Dict[str, object]] = None,
    required_factual_classes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Merge deduplicated test-split unified rows from all trainers.

    Each row is one directed transition (``factual_class`` -> ``alternative_class``).
    Used for cross-task encoder evaluation: every GRADIEND encodes the same full
    transition pool (he->she, jewish->christian, etc.).

    When ``required_factual_classes`` is set, ``probe_trainers`` are consulted to
    materialize transitions for factual classes missing from the trained-trainer
    pool (e.g. ``1PL`` / ``2SGPL`` when only a subset of pronoun GRADIENDs trained).
    """
    frames = _iter_trainer_unified_test_frames(trainers, split=split)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if probe_trainers and required_factual_classes:
        merged = supplement_unified_test_rows(
            merged,
            probe_trainers,
            required_factual_classes,
            split=split,
        )
    if merged.empty:
        return merged
    return merged.drop_duplicates(
        subset=[c for c in merged.columns if c in REQUIRED_UNIFIED],
        keep="first",
    ).reset_index(drop=True)


def materialize_trainer_test_transitions(
    trainer: object,
    *,
    split: str = "test",
) -> pd.DataFrame:
    """Return unified test-split rows with all class transitions materialized."""
    ensure = getattr(trainer, "_ensure_data", None)
    if callable(ensure):
        try:
            ensure(materialize_all_class_transitions=True)
        except TypeError:
            ensure()
    combined = getattr(trainer, "combined_data", None)
    if not isinstance(combined, pd.DataFrame) or combined.empty:
        return pd.DataFrame()
    missing = REQUIRED_UNIFIED - set(combined.columns)
    if missing:
        raise ValueError(
            f"Trainer {getattr(trainer, 'run_id', trainer)!r} combined_data missing unified columns: {sorted(missing)}"
        )
    resolved_split = _normalize_split_name(split)
    split_mask = combined[UNIFIED_SPLIT].astype(str).map(_normalize_split_name) == resolved_split
    return combined[split_mask].copy().reset_index(drop=True)


def supplement_unified_test_rows(
    eval_rows: pd.DataFrame,
    probe_trainers: Dict[str, object],
    required_factual_classes: Sequence[str],
    *,
    split: str = "test",
) -> pd.DataFrame:
    """Add test transitions for factual classes absent from ``eval_rows``."""
    required = {str(value) for value in required_factual_classes}
    if not required or not probe_trainers:
        return eval_rows
    present: set[str] = set()
    if isinstance(eval_rows, pd.DataFrame) and not eval_rows.empty and UNIFIED_FACTUAL_CLASS in eval_rows.columns:
        present = set(eval_rows[UNIFIED_FACTUAL_CLASS].astype(str).tolist())
    missing = sorted(required - present)
    if not missing:
        return eval_rows
    supplemental_frames: List[pd.DataFrame] = []
    for trainer in probe_trainers.values():
        try:
            probe_df = materialize_trainer_test_transitions(trainer, split=split)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Skipping probe trainer %s while supplementing cross-task eval rows: %s",
                getattr(trainer, "run_id", trainer),
                exc,
            )
            continue
        if probe_df.empty:
            continue
        supplemental_frames.append(
            probe_df[probe_df[UNIFIED_FACTUAL_CLASS].astype(str).isin(missing)].copy()
        )
        present.update(probe_df[UNIFIED_FACTUAL_CLASS].astype(str).tolist())
        if not (required - present):
            break
    if not supplemental_frames:
        logger.warning(
            "Cross-task eval pool is missing factual classes %s and no probe trainer supplied rows.",
            missing,
        )
        return eval_rows
    frames = [eval_rows] if isinstance(eval_rows, pd.DataFrame) and not eval_rows.empty else []
    frames.extend(supplemental_frames)
    return pd.concat(frames, ignore_index=True)


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
    merged = collect_unified_test_rows(trainers, split=split)
    if merged.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for class_id, group in merged.groupby(merged[UNIFIED_FACTUAL_CLASS].astype(str)):
        out[str(class_id)] = group.reset_index(drop=True)
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
            "feature_class_id": transition_id(src, tgt),
        }
        if UNIFIED_SPLIT in row.index and pd.notna(row[UNIFIED_SPLIT]):
            entry[UNIFIED_SPLIT] = row[UNIFIED_SPLIT]
        pairs.append(entry)
    return pairs


def _unified_df_for_text_training(unified_df: pd.DataFrame, pairs: List[Dict[str, Any]]) -> pd.DataFrame:
    out = unified_df.copy().reset_index(drop=True)
    if "label" not in out.columns:
        out["label"] = [pair["label"] for pair in pairs]
    if "feature_class_id" not in out.columns:
        out["feature_class_id"] = [pair["feature_class_id"] for pair in pairs]
    if "factual_id" not in out.columns:
        out["factual_id"] = out[UNIFIED_FACTUAL_CLASS].astype(str)
    if "alternative_id" not in out.columns:
        out["alternative_id"] = out[UNIFIED_ALTERNATIVE_CLASS].astype(str)
    return out


def _gradient_dataset_for_unified_df(
    trainer: object,
    model: object,
    unified_df: pd.DataFrame,
) -> object:
    """Build a gradient dataset from unified rows (preserves factual/alternative ids)."""
    tokenizer = model.tokenizer
    prediction_objective = getattr(trainer, "_prediction_objective", None)
    if callable(prediction_objective):
        objective_name = prediction_objective(tokenizer).name
    else:
        objective_name = "mlm"
    is_seq2seq = is_seq2seq_model(tokenizer)
    is_decoder_only_model = False if is_seq2seq else tokenizer.mask_token_id is None
    if objective_name == "clm_sequence_cloze":
        is_decoder_only_model = True
    elif objective_name == "seq2seq_decoder_sequence_cloze":
        is_decoder_only_model = False
    text_ds = TextTrainingDataset(
        unified_df.reset_index(drop=True),
        tokenizer,
        batch_size=1,
        is_decoder_only_model=is_decoder_only_model,
        is_seq2seq_model=is_seq2seq,
        target_key="label",
        balance_column="feature_class_id",
        seed=42,
        prediction_objective=objective_name,
        rhs_window=getattr(getattr(trainer, "_training_args", None), "decoder_sequence_cloze_rhs_window", -1),
    )
    return trainer.create_gradient_training_dataset(text_ds, model)


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
    grad_ds = _gradient_dataset_for_unified_df(trainer, model, _unified_df_for_text_training(df, pairs))
    rows = encode_dataset_to_rows(model, grad_ds)
    if not rows:
        return None
    values = [_safe_float(r.get("encoded")) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_cross_task_eval_rows(
    eval_rows: pd.DataFrame,
    max_size: Optional[int],
) -> pd.DataFrame:
    """Cap eval rows per factual class so rare classes are not dropped."""
    if max_size is None or eval_rows.empty:
        return eval_rows
    parts: List[pd.DataFrame] = []
    for _, group in eval_rows.groupby(eval_rows[UNIFIED_FACTUAL_CLASS].astype(str)):
        part = group
        if len(part) > max_size:
            part = part.sample(n=max_size, random_state=42)
        parts.append(part)
    if not parts:
        return eval_rows
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def _artifact_use_cache_from_trainers(
    trainers: Dict[str, object],
    use_cache: Optional[bool],
) -> bool:
    if use_cache is not None:
        return coerce_artifact_use_cache(use_cache)
    for trainer in trainers.values():
        args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
        if args is not None:
            return coerce_artifact_use_cache(getattr(args, "use_cache", False))
    return False


def _resolve_unified_encoder_cache_path(
    trainer: object,
    *,
    split: str,
    max_size: Optional[int],
) -> Optional[str]:
    """Same CSV path as ``evaluate_encoder`` / ``_analyze_encoder`` (split + max_size only)."""
    cache_path_fn = getattr(trainer, "_encoder_cache_path", None)
    if callable(cache_path_fn):
        model_path = getattr(trainer, "model_path", None) or ""
        return cache_path_fn(model_path, split=split, max_size=max_size)
    experiment_dir = getattr(trainer, "experiment_dir", None)
    if callable(experiment_dir):
        experiment_dir = experiment_dir()
    if not experiment_dir:
        return None
    key_kwargs: Dict[str, Any] = {}
    if split is not None:
        available = None
        combined = getattr(trainer, "combined_data", None)
        if isinstance(combined, pd.DataFrame) and UNIFIED_SPLIT in combined.columns:
            available = combined[UNIFIED_SPLIT].dropna().astype(str).tolist()
        from gradiend.util.encoder_splits import encoder_split_cache_key

        key_kwargs["split"] = encoder_split_cache_key(split, available=available)
    if max_size is not None:
        key_kwargs["max_size"] = max_size
    return resolve_encoder_analysis_path(experiment_dir, None, **key_kwargs)


def _probe_key_from_unified_row(row: pd.Series) -> Tuple[str, str, str]:
    """Stable probe identity for a unified eval snippet."""
    return (
        str(row[UNIFIED_MASKED]),
        str(row[UNIFIED_FACTUAL]),
        str(row[UNIFIED_ALTERNATIVE]),
    )


def _probe_key_from_encoder_row(row: pd.Series) -> Optional[Tuple[str, ...]]:
    """Match a cached encoder row to a unified probe snippet when possible."""
    return encoder_probe_key_from_row(row)


def _encoder_probe_keys(encoder_df: pd.DataFrame) -> set[Tuple[str, ...]]:
    return encoder_probe_keys(encoder_df)


def _filter_unified_rows_for_missing_probes(
    unified_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return unified rows whose probe snippets are not already present in ``cached_df``."""
    if unified_df.empty:
        return unified_df
    cached_keys = _encoder_probe_keys(cached_df)
    if not cached_keys:
        return unified_df
    missing_indices: List[Any] = []
    for index, row in unified_df.iterrows():
        if _probe_key_from_unified_row(row) not in cached_keys:
            missing_indices.append(index)
    if not missing_indices:
        return unified_df.iloc[0:0].copy()
    return unified_df.loc[missing_indices].reset_index(drop=True)


def _merge_encoder_row_dfs(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge encoder outputs, preferring newly encoded rows on probe-key collisions."""
    if existing_df.empty:
        return new_df.copy()
    if new_df.empty:
        return existing_df.copy()
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    keep_indices: List[int] = []
    seen: set[Tuple[str, ...]] = set()
    for index in range(len(combined) - 1, -1, -1):
        row = combined.iloc[index]
        key = _probe_key_from_encoder_row(row)
        if key is None:
            keep_indices.append(index)
            continue
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(index)
    keep_indices.reverse()
    return combined.iloc[keep_indices].reset_index(drop=True)


def _read_encoder_cache_csv(cache_path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(cache_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning(
            "Cached encoder analysis at %s is unreadable (%s). Recomputing.",
            cache_path,
            exc,
        )
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None


def _observed_transition_ids(encoder_df: pd.DataFrame) -> set[str]:
    if "transition_id" in encoder_df.columns:
        values = encoder_df["transition_id"].dropna().astype(str).tolist()
        return {
            tid
            for value in values
            if (tid := normalize_transition_id(value) or str(value))
        }
    observed: set[str] = set()
    for _, row in encoder_df.iterrows():
        tid = _transition_id_from_row(row)
        if tid:
            normalized = normalize_transition_id(tid) or str(tid)
            observed.add(normalized)
    return observed


def _encoder_df_has_directed_transitions(cached: pd.DataFrame) -> bool:
    """True when rows encode directed transitions, not identity class probes only."""
    if "factual_id" in cached.columns and (
        "counterfactual_id" in cached.columns or "alternative_id" in cached.columns
    ):
        factual = cached["factual_id"].astype(str)
        counter_col = (
            "counterfactual_id"
            if "counterfactual_id" in cached.columns
            else "alternative_id"
        )
        counter = cached[counter_col].astype(str)
        return bool((factual != counter).any())
    if "source_id" in cached.columns and "target_id" in cached.columns:
        source = cached["source_id"].astype(str)
        target = cached["target_id"].astype(str)
        return bool((source != target).any())
    return False


def _cached_cross_task_encoder_df_matches(
    cached: pd.DataFrame,
    trainer: object,
    expected_transitions: set[str],
) -> bool:
    """True when the unified cache already holds the full cross-task eval pool."""
    if not isinstance(cached, pd.DataFrame) or cached.empty or "encoded" not in cached.columns:
        return False
    if not _encoder_df_has_directed_transitions(cached):
        return False
    matcher = getattr(trainer, "_cached_encoder_df_matches_request", None)
    if callable(matcher):
        if not matcher(cached, include_other_classes=True, use_all_transitions=True):
            return False
    if expected_transitions:
        observed = _observed_transition_ids(cached)
        if not expected_transitions.issubset(observed):
            return False
    return True


def _encode_cross_task_rows_for_trainer(
    trainer: object,
    *,
    model: object,
    rows_to_encode: pd.DataFrame,
) -> pd.DataFrame:
    """Encode unified cross-task probe rows with one loaded model."""
    if rows_to_encode.empty:
        return pd.DataFrame()
    pairs = _build_probe_pairs_from_unified_df(trainer, rows_to_encode)
    if not pairs:
        return pd.DataFrame()
    grad_ds = _gradient_dataset_for_unified_df(
        trainer,
        model,
        _unified_df_for_text_training(rows_to_encode, pairs),
    )
    encoded_rows = encode_dataset_to_rows(model, grad_ds)
    all_rows: List[Dict[str, Any]] = []
    for row in encoded_rows:
        payload = dict(row)
        payload["type"] = "training"
        all_rows.append(payload)
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def _build_cross_task_encoder_df_for_seed(
    trainer: object,
    *,
    trainer_id: str,
    df: pd.DataFrame,
    split: str,
    max_size: Optional[int],
    use_cache_effective: bool,
    expected_transitions: set[str],
    load_directory: Optional[str] = None,
    allow_disk_cache: bool = True,
) -> pd.DataFrame:
    """Build one cross-task encoder DataFrame for a trainer checkpoint."""
    cache_path = _resolve_unified_encoder_cache_path(trainer, split=split, max_size=max_size)
    encoder_df: Optional[pd.DataFrame] = None
    partial_cache: Optional[pd.DataFrame] = None
    if allow_disk_cache and use_cache_effective and cache_path and os.path.isfile(cache_path):
        cached = _read_encoder_cache_csv(str(cache_path))
        if cached is not None:
            if _cached_cross_task_encoder_df_matches(cached, trainer, expected_transitions):
                encoder_df = cached
                logger.info(
                    "Using unified encoder cache for %s from %s",
                    trainer_id,
                    cache_path,
                )
            elif not cached.empty and "encoded" in cached.columns:
                partial_cache = cached

    if encoder_df is None:
        rows_to_encode = df
        if partial_cache is not None:
            rows_to_encode = _filter_unified_rows_for_missing_probes(df, partial_cache)
            if partial_cache is not None and rows_to_encode.empty:
                encoder_df = partial_cache
            elif partial_cache is not None and not rows_to_encode.empty:
                logger.info(
                    "Reusing partial unified encoder cache for %s; encoding %d new probes.",
                    trainer_id,
                    len(rows_to_encode),
                )

        if encoder_df is None:
            model = _load_eval_model_for_trainer(trainer, load_directory=load_directory)
            try:
                fresh_df = _encode_cross_task_rows_for_trainer(
                    trainer,
                    model=model,
                    rows_to_encode=rows_to_encode,
                )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if partial_cache is not None and not partial_cache.empty:
                encoder_df = _merge_encoder_row_dfs(partial_cache, fresh_df)
            else:
                encoder_df = fresh_df

        if (
            allow_disk_cache
            and use_cache_effective
            and cache_path
            and isinstance(encoder_df, pd.DataFrame)
            and not encoder_df.empty
        ):
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            if os.path.isfile(cache_path):
                invalidate_encoder_metrics_cache(cache_path)
            encoder_df.to_csv(cache_path, index=False)
            logger.info("Wrote unified encoder cache for %s to %s", trainer_id, cache_path)

    if isinstance(encoder_df, pd.DataFrame) and not encoder_df.empty:
        return orient_encoder_df_by_label_correlation(encoder_df)
    if isinstance(encoder_df, pd.DataFrame):
        return encoder_df
    return pd.DataFrame()


def _load_unified_encoder_cache(
    cache_path: str,
    *,
    trainer: object,
    expected_transitions: set[str],
) -> Optional[pd.DataFrame]:
    cached = _read_encoder_cache_csv(cache_path)
    if cached is None:
        return None
    if not _cached_cross_task_encoder_df_matches(cached, trainer, expected_transitions):
        return None
    return cached


def build_cross_task_encoder_summary(
    trainers: Dict[str, object],
    feature_classes: Sequence[str],
    *,
    eval_rows: Optional[pd.DataFrame] = None,
    eval_by_class: Optional[Dict[str, pd.DataFrame]] = None,
    split: str = "test",
    max_size: Optional[int] = None,
    use_cache: Optional[bool] = None,
    seed_selection: Optional[str] = None,
    seed_aggregate: str = "mean",
    dispersion: Optional[str] = None,
) -> Dict[str, Any]:
    """Build per-trainer encoder summaries on the shared cross-task test pool.

    Each binary GRADIEND encodes **every** test transition in the merged pool
    (e.g. he->she, she->he, jewish->christian, ...) using that trainer's model.
    Rows with ``label == 0`` are kept: they are cross-domain probes where the
    snippet's factual class is outside the trainer's pair.

    When ``seed_selection`` is omitted, trainers with
    ``analyze_seed_stability=True`` automatically use all convergent seed
    checkpoints and aggregate encoder responses (same rule as suite plots).
    """
    meta = comparison_seed_metadata(
        trainers,
        seed_selection=seed_selection,
        seed_aggregate=seed_aggregate,
        dispersion=dispersion,
    )
    resolved_selection = meta["seed_selection"]
    resolved_dispersion = meta["dispersion"]
    seed_aggregate = meta["seed_aggregate"]
    multi_seed = meta["multi_seed"]
    _validate_aggregate_dispersion_combo(seed_aggregate, resolved_dispersion)

    if eval_rows is None:
        if eval_by_class is not None:
            eval_rows = pd.concat(
                [df for df in eval_by_class.values() if isinstance(df, pd.DataFrame) and not df.empty],
                ignore_index=True,
            )
            eval_rows = eval_rows.drop_duplicates(
                subset=[c for c in eval_rows.columns if c in REQUIRED_UNIFIED],
                keep="first",
            ).reset_index(drop=True)
        else:
            eval_rows = collect_unified_test_rows(trainers, split=split)

    if eval_rows is None or eval_rows.empty:
        empty = {str(trainer_id): {"encoder_df": pd.DataFrame()} for trainer_id in trainers}
        if multi_seed:
            for payload in empty.values():
                payload["encoder_dfs"] = []
                payload["multi_seed"] = True
        return empty

    df = _sample_cross_task_eval_rows(eval_rows, max_size)
    expected_transitions: set[str] = set()
    if UNIFIED_TRANSITION in df.columns:
        expected_transitions = {
            normalize_transition_id(value) or str(value)
            for value in df[UNIFIED_TRANSITION].astype(str).unique().tolist()
        }
    use_cache_effective = _artifact_use_cache_from_trainers(trainers, use_cache)

    summary: Dict[str, Any] = {}
    for trainer_id, trainer in trainers.items():
        trainer_key = str(trainer_id)
        base = unwrap_trainer(trainer)
        seed_entries = analysis_seed_entries(trainer, seed_selection=resolved_selection)
        if resolved_selection == "best" or len(seed_entries) <= 1:
            encoder_df = _build_cross_task_encoder_df_for_seed(
                base,
                trainer_id=trainer_key,
                df=df,
                split=split,
                max_size=max_size,
                use_cache_effective=use_cache_effective,
                expected_transitions=expected_transitions,
                allow_disk_cache=True,
            )
            summary[trainer_key] = {
                "encoder_df": encoder_df,
                **meta,
                "multi_seed": False,
            }
            continue

        best_seed_path = base.get_best_seed_run_path() if hasattr(base, "get_best_seed_run_path") else None
        encoder_dfs: List[pd.DataFrame] = []
        for seed_val, seed_path in seed_entries:
            allow_disk_cache = (
                best_seed_path is None
                or os.path.normcase(seed_path) == os.path.normcase(str(best_seed_path))
            )
            encoder_df = _build_cross_task_encoder_df_for_seed(
                base,
                trainer_id=trainer_key,
                df=df,
                split=split,
                max_size=max_size,
                use_cache_effective=use_cache_effective and allow_disk_cache,
                expected_transitions=expected_transitions,
                load_directory=seed_path,
                allow_disk_cache=allow_disk_cache,
            )
            if not encoder_df.empty:
                encoder_dfs.append(encoder_df)
            logger.info(
                "Cross-task encoder summary for %s seed %s (%d/%d)",
                trainer_key,
                seed_val,
                len(encoder_dfs),
                len(seed_entries),
            )

        aggregated_df = aggregate_encoder_dataframes(encoder_dfs)
        summary[trainer_key] = {
            "encoder_df": aggregated_df,
            "encoder_dfs": encoder_dfs,
            **meta,
            "multi_seed": True,
            "n_seeds": len(encoder_dfs),
        }
    return summary


def collect_unified_test_transitions(
    trainers: Dict[str, object],
    *,
    split: str = "test",
    eval_rows: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Return sorted directed transition labels observed in the shared test pool."""
    merged = eval_rows if eval_rows is not None else collect_unified_test_rows(trainers, split=split)
    if merged.empty or UNIFIED_TRANSITION not in merged.columns:
        return []
    return sorted(
        normalize_transition_id(value) or str(value)
        for value in merged[UNIFIED_TRANSITION].astype(str).unique().tolist()
    )


def _mean_encoded_by_transition(encoder_df: pd.DataFrame) -> Dict[str, Tuple[float, int]]:
    """Within one GRADIEND: mean encoded value per directed input transition."""
    if not isinstance(encoder_df, pd.DataFrame) or encoder_df.empty or "encoded" not in encoder_df.columns:
        return {}
    work = encoder_df.copy()
    if "type" in work.columns:
        work = work[~work["type"].astype(str).str.contains("neutral", case=False, na=False)].copy()
    if work.empty:
        return {}
    work["_transition_id"] = work.apply(_transition_id_from_row, axis=1)
    work = work[work["_transition_id"].notna()].copy()
    if work.empty:
        return {}
    grouped = (
        work.groupby("_transition_id", dropna=False)["encoded"]
        .agg(["mean", "count"])
        .reset_index()
    )
    out: Dict[str, Tuple[float, int]] = {}
    for _, row in grouped.iterrows():
        transition_id_value = normalize_transition_id(row["_transition_id"])
        if not transition_id_value:
            continue
        mean_val = _safe_float(row["mean"])
        if mean_val is None:
            continue
        out[str(transition_id_value)] = (float(mean_val), int(row["count"]))
    return out


def _transition_stats_from_encoder_summary(
    payload: Any,
) -> Tuple[Dict[str, Tuple[float, int]], Dict[str, float]]:
    """Mean and across-seed std of per-transition encoded means."""
    encoder_dfs = encoder_dataframes_from_summary(payload)
    if not encoder_dfs:
        return {}, {}
    if len(encoder_dfs) == 1:
        return _mean_encoded_by_transition(encoder_dfs[0]), {}

    per_seed = [_mean_encoded_by_transition(df) for df in encoder_dfs]
    keys = {key for mapping in per_seed for key in mapping}
    means: Dict[str, Tuple[float, int]] = {}
    stds: Dict[str, float] = {}
    for key in keys:
        values = [float(mapping[key][0]) for mapping in per_seed if key in mapping]
        if not values:
            continue
        mean_value = float(sum(values) / len(values))
        count = int(next(mapping[key][1] for mapping in per_seed if key in mapping))
        means[key] = (mean_value, count)
        if len(values) > 1:
            stds[key] = float(
                math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
            )
    return means, stds


def _mean_encoded_by_transition_from_summary(
    payload: Any,
    *,
    seed_aggregate: str = "mean",
    dispersion: str = "none",
) -> Dict[str, Tuple[float, int]]:
    """Mean encoded value per transition, aggregating across seed checkpoints when present."""
    encoder_dfs = encoder_dataframes_from_summary(payload)
    if not encoder_dfs:
        return {}
    if len(encoder_dfs) == 1:
        return _mean_encoded_by_transition(encoder_dfs[0])

    aggregated = aggregate_encoder_dataframes(encoder_dfs)
    return _mean_encoded_by_transition(aggregated)


def compute_gradiend_transition_cross_encoding_matrix(
    trainers: Dict[str, object],
    *,
    trainer_order: Optional[Sequence[str]] = None,
    transition_order: Optional[Sequence[str]] = None,
    encoder_summary: Optional[Dict[str, Any]] = None,
    split: str = "test",
    max_size: Optional[int] = None,
    use_cache: Optional[bool] = None,
    seed_selection: Optional[str] = None,
    seed_aggregate: str = "mean",
    dispersion: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a GRADIEND × directed-transition cross-encoding matrix.

    Each row is one trained GRADIEND; each column is a directed input transition
    ``factual_class->alternative_class`` from the shared cross-task test pool.
    Cell ``(i, j)`` is the mean encoded response of GRADIEND *i* on snippets with
    transition *j* (within-model average only).

    This is the standard rectangular matrix **before** anchor sign alignment and
    anchor-class aggregation used by :func:`compute_anchor_aligned_encoding_matrix`.
    """
    if not trainers:
        raise ValueError("trainers must be a non-empty dict")
    order = [str(t) for t in (trainer_order or list(trainers.keys())) if str(t) in trainers]
    if not order:
        raise ValueError("trainer_order produced no valid trainer ids")

    if encoder_summary is None:
        encoder_summary = build_cross_task_encoder_summary(
            trainers,
            [],
            split=split,
            max_size=max_size,
            use_cache=use_cache,
            seed_selection=seed_selection,
            seed_aggregate=seed_aggregate,
            dispersion=dispersion,
        )

    meta = comparison_seed_metadata(
        trainers,
        seed_selection=seed_selection,
        seed_aggregate=seed_aggregate,
        dispersion=dispersion,
    )

    per_trainer_means: Dict[str, Dict[str, Tuple[float, int]]] = {}
    per_trainer_stds: Dict[str, Dict[str, float]] = {}
    for trainer_id, payload in encoder_summary.items():
        if str(trainer_id) not in order:
            continue
        entry = payload if isinstance(payload, dict) else {}
        means, stds = _transition_stats_from_encoder_summary(entry)
        per_trainer_means[str(trainer_id)] = means
        per_trainer_stds[str(trainer_id)] = stds

    if transition_order is None:
        observed: set[str] = set()
        for values in per_trainer_means.values():
            observed.update(values.keys())
        columns = sorted(observed)
    else:
        columns = [
            normalize_transition_id(value) or str(value) for value in transition_order
        ]

    matrix: List[List[float]] = []
    n_matrix: List[List[int]] = []
    cell_stats: List[List[Dict[str, Any]]] = []
    has_std = any(stds for stds in per_trainer_stds.values())
    for trainer_id in order:
        by_transition = per_trainer_means.get(trainer_id, {})
        by_std = per_trainer_stds.get(trainer_id, {})
        row_values: List[float] = []
        row_counts: List[int] = []
        stats_row: List[Dict[str, Any]] = []
        for transition_id in columns:
            cell = by_transition.get(transition_id)
            if cell is None:
                row_values.append(float("nan"))
                row_counts.append(0)
                stats_row.append({})
            else:
                row_values.append(float(cell[0]))
                row_counts.append(int(cell[1]))
                stat: Dict[str, Any] = {"aggregate": float(cell[0]), "n": int(cell[1])}
                std_value = by_std.get(transition_id)
                if std_value is not None:
                    stat["std"] = float(std_value)
                stats_row.append(stat)
        matrix.append(row_values)
        n_matrix.append(row_counts)
        if has_std:
            cell_stats.append(stats_row)

    payload: Dict[str, Any] = {
        "measure": "gradiend_transition_cross_encoding_mean",
        "model_ids": order,
        "column_ids": columns,
        "rows": order,
        "columns": columns,
        "matrix": matrix,
        "n_matrix": n_matrix,
        "split": split,
        "max_size": max_size,
        "transitions_found": sorted(
            {t for values in per_trainer_means.values() for t in values}
        ),
        **meta,
    }
    if has_std:
        payload["cell_stats"] = cell_stats
        payload["multi_seed"] = True
    return payload


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
