"""Cross-encoding matrices for binary trainer comparisons."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from gradiend.comparison.common import _aggregate_seed_scores, _validate_aggregate_dispersion_combo
from gradiend.trainer.core.multi_seed import (
    resolve_dispersion_for_trainers,
    resolve_seed_run_entries,
    resolve_seed_selection_for_trainers,
)
from gradiend.util.paths import (
    ARTIFACT_MODEL,
    has_saved_model,
    resolve_encoder_analysis_path,
    resolve_output_path,
)


def _infer_positive_class_from_pair(target_classes: Sequence[str], explicit_positive_class: Optional[str] = None) -> str:
    classes = [str(c) for c in target_classes]
    if len(classes) != 2:
        raise ValueError("Canonical cross-encoding requires exactly 2 target classes.")
    if explicit_positive_class is not None:
        if explicit_positive_class not in classes:
            raise ValueError(f"positive_class {explicit_positive_class!r} is not in target_classes {classes!r}")
        return explicit_positive_class
    a, b = classes
    if a == f"non_{b}" or a == f"non-{b}":
        return b
    if b == f"non_{a}" or b == f"non-{a}":
        return a
    raise ValueError(
        "Canonical cross-encoding requires a binary target pair with an inferable positive class "
        "(supported heuristic: one class is the other's non_/non- prefixed form), or set "
        "TrainingArguments.positive_class explicitly."
    )


def _extract_positive_mean(eval_result: Dict[str, Any], positive_class: str) -> float:
    mean_by_class = eval_result.get("mean_by_class") or {}
    label_map = eval_result.get("label_value_to_class_name") or {}
    for lbl, class_name in label_map.items():
        if str(class_name) == str(positive_class):
            for candidate in (float(lbl), str(lbl), lbl):
                if candidate in mean_by_class:
                    return float(mean_by_class[candidate])
    if positive_class in mean_by_class:
        return float(mean_by_class[positive_class])
    raise ValueError(f"Could not resolve positive class {positive_class!r} in mean_by_class result.")


def _extract_positive_mean_from_df(encoder_df: Any, positive_class: str) -> float:
    if not hasattr(encoder_df, "columns"):
        raise TypeError("positive_mean requires a pandas DataFrame-like encoder_df")
    df = encoder_df
    if "type" in df.columns:
        df = df[~df["type"].astype(str).str.contains("neutral", case=False, na=False)]
    if "source_id" in df.columns:
        df = df[df["source_id"].astype(str) == str(positive_class)]
    elif "label" in df.columns:
        df = df[df["label"].astype(float) > 0]
    if len(df) == 0:
        raise ValueError(f"No positive-class rows found for {positive_class!r} in encoder subset.")
    return float(df["encoded"].astype(float).mean())


def _extract_negative_mean_from_df(encoder_df: Any, negative_class: str) -> float:
    if not hasattr(encoder_df, "columns"):
        raise TypeError("negative_mean requires a pandas DataFrame-like encoder_df")
    df = encoder_df
    if "type" in df.columns:
        df = df[~df["type"].astype(str).str.contains("neutral", case=False, na=False)]
    if "source_id" in df.columns:
        df = df[df["source_id"].astype(str) == str(negative_class)]
    elif "label" in df.columns:
        df = df[df["label"].astype(float) < 0]
    if len(df) == 0:
        raise ValueError(f"No negative-class rows found for {negative_class!r} in encoder subset.")
    return float(df["encoded"].astype(float).mean())


def _is_gradiend_checkpoint(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "gradiend_context.json")):
        return True
    return has_saved_model(path)


def _resolve_trainer_load_directory(trainer: object, *, load_directory: Optional[str] = None) -> str:
    """Resolve a GRADIEND checkpoint directory for cross-encoding evaluation."""
    candidates: List[str] = []
    if load_directory:
        candidates.append(str(load_directory))
    experiment_dir = getattr(trainer, "experiment_dir", None)
    if callable(experiment_dir):
        experiment_dir = experiment_dir()
    if experiment_dir:
        artifact_path = resolve_output_path(str(experiment_dir), None, ARTIFACT_MODEL)
        if artifact_path:
            candidates.append(artifact_path)
    model_path = getattr(trainer, "model_path", None)
    if model_path:
        candidates.append(str(model_path))

    seen: set[str] = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if _is_gradiend_checkpoint(norm):
            return norm

    if candidates:
        return os.path.normpath(candidates[0])
    run_id = getattr(trainer, "run_id", None)
    raise ValueError(
        f"Trainer {run_id!r} has no saved GRADIEND checkpoint. "
        "Train the model first or pass a checkpoint path via load_directory."
    )


def _load_eval_model_for_trainer(trainer: object, *, load_directory: Optional[str] = None) -> Any:
    from gradiend.comparison.seed_policy import unwrap_trainer

    trainer = unwrap_trainer(trainer)
    training_args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
    kwargs: Dict[str, Any] = {"definition": trainer}
    if training_args is not None:
        kwargs["training_args"] = training_args
        kwargs["trust_remote_code"] = getattr(training_args, "trust_remote_code", False)
    checkpoint_path = _resolve_trainer_load_directory(trainer, load_directory=load_directory)
    return trainer.load_model(checkpoint_path, **kwargs)


def _load_cached_encoder_df(trainer: object, *, split: str, max_size: Optional[int]) -> Optional[pd.DataFrame]:
    experiment_dir = getattr(trainer, "experiment_dir", None)
    if callable(experiment_dir):
        experiment_dir = experiment_dir()
    if not experiment_dir:
        return None
    key_kwargs: Dict[str, Any] = {"split": split}
    if max_size is not None:
        key_kwargs["max_size"] = max_size
    cache_path = resolve_encoder_analysis_path(experiment_dir, None, **key_kwargs)
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        return pd.read_csv(cache_path)
    except Exception:
        return None


def _subset_encoder_df_for_target_classes(encoder_df: Any, target_classes: Sequence[str]) -> Any:
    classes = {str(c) for c in target_classes}
    if len(classes) != 2:
        raise ValueError(f"Cross-encoding currently requires exactly 2 target classes, got {list(target_classes)!r}")
    if not hasattr(encoder_df, "columns"):
        raise TypeError("cross-encoding expects encoder_df to be a pandas DataFrame-like object")
    if "source_id" in encoder_df.columns and "target_id" in encoder_df.columns:
        source_vals = encoder_df["source_id"].astype(str)
        target_vals = encoder_df["target_id"].astype(str)
        mask = source_vals.isin(classes) & target_vals.isin(classes)
        return encoder_df[mask].copy()
    if "source_id" in encoder_df.columns:
        source_vals = encoder_df["source_id"].astype(str)
        return encoder_df[source_vals.isin(classes)].copy()
    raise ValueError(
        "Cross-encoding requires encoder outputs with source_id/target_id information so column-pair subsets can be extracted."
    )


def _resolve_full_eval(full_eval: Optional[bool], split: str) -> bool:
    """Map suite-style full_eval to include_other_classes for encoder evaluation."""
    if full_eval is None:
        return str(split).lower() == "test"
    return bool(full_eval)


def _matrix_row_ids(comparison_data: Dict[str, Any]) -> Optional[List[Any]]:
    row_ids = comparison_data.get("model_ids")
    if row_ids is None:
        row_ids = comparison_data.get("rows")
    return list(row_ids) if isinstance(row_ids, list) else None


def _matrix_column_ids(comparison_data: Dict[str, Any]) -> Optional[List[Any]]:
    col_ids = comparison_data.get("column_ids")
    if col_ids is None:
        col_ids = comparison_data.get("columns")
    if col_ids is None:
        return _matrix_row_ids(comparison_data)
    return list(col_ids) if isinstance(col_ids, list) else None


def can_normalize_cross_encoding_by_diagonal(comparison_data: Dict[str, Any]) -> bool:
    """True when row ``i`` and column ``i`` refer to the same id (square self-axis)."""
    row_ids = _matrix_row_ids(comparison_data)
    col_ids = _matrix_column_ids(comparison_data)
    matrix = comparison_data.get("matrix")
    if row_ids is None or col_ids is None or not isinstance(matrix, list):
        return False
    if len(row_ids) != len(col_ids) or len(matrix) != len(row_ids):
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(row_ids):
            return False
    return row_ids == col_ids


def normalize_cross_encoding_rows_by_diagonal(comparison_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize each row of cross-encoding data by its diagonal value.

    Args:
        comparison_data: Payload with list-valued ``matrix`` and ``model_ids``.
            For oriented matrices, ``column_ids`` must match ``model_ids`` so
            ``matrix[i][i]`` is the anchor self-encoding. Use
            :func:`can_normalize_cross_encoding_by_diagonal` to check first.
            Optional ``cell_stats`` entries are scaled consistently for numeric
            aggregate fields.

    Returns:
        A shallow copy of ``comparison_data`` with ``matrix`` row ``i`` divided
        by ``matrix[i][i]`` and ``row_normalized_by_diagonal=True``. When
        present, numeric ``cell_stats`` fields such as ``aggregate``, ``std``,
        ``min``, ``max``, ``range_half_width``, ``scores``, and ``minmax`` are
        scaled by the same row factor; counts are preserved.

    Raises:
        TypeError: If the payload does not contain list-valued ``matrix`` and
            ``model_ids`` entries.
        ValueError: If row/column ids are not aligned for a square diagonal, or
            any diagonal value is zero.
    """
    if not can_normalize_cross_encoding_by_diagonal(comparison_data):
        row_ids = _matrix_row_ids(comparison_data)
        col_ids = _matrix_column_ids(comparison_data)
        raise ValueError(
            "Cannot normalize by diagonal: matrix must be square with "
            f"matching row and column ids (got rows={row_ids!r}, columns={col_ids!r})."
        )
    matrix = comparison_data.get("matrix")
    model_ids = _matrix_row_ids(comparison_data)
    if not isinstance(matrix, list) or model_ids is None:
        raise TypeError("comparison_data must contain list-valued 'matrix' and 'model_ids'")
    normalized = dict(comparison_data)
    normalized_matrix: List[List[float]] = []
    normalized_cell_stats: List[List[Dict[str, Any]]] = []
    cell_stats = comparison_data.get("cell_stats")
    for i, row in enumerate(matrix):
        diag = float(row[i])
        if diag == 0.0:
            raise ValueError(
                f"Cannot normalize cross-encoding row {i} ({model_ids[i]!r}) because its diagonal value is 0."
            )
        factor = 1.0 / diag
        normalized_matrix.append([float(value) * factor for value in row])
        if isinstance(cell_stats, list) and i < len(cell_stats) and isinstance(cell_stats[i], list):
            stats_row: List[Dict[str, Any]] = []
            for stat in cell_stats[i]:
                if not isinstance(stat, dict):
                    stats_row.append(stat)
                    continue
                scaled_stat: Dict[str, Any] = {}
                for key, value in stat.items():
                    if key == "n":
                        scaled_stat[key] = value
                    elif key in {"aggregate", "min", "max", "std", "range_half_width"} and isinstance(value, (int, float)):
                        scaled_stat[key] = float(value) * factor
                    elif key in {"scores", "minmax"} and isinstance(value, list):
                        scaled_stat[key] = [float(v) * factor if isinstance(v, (int, float)) else v for v in value]
                    else:
                        scaled_stat[key] = value
                stats_row.append(scaled_stat)
            normalized_cell_stats.append(stats_row)
    normalized["matrix"] = normalized_matrix
    if normalized_cell_stats:
        normalized["cell_stats"] = normalized_cell_stats
    normalized["row_normalized_by_diagonal"] = True
    measure = normalized.get("measure")
    if isinstance(measure, str) and not measure.endswith("_row_normalized"):
        normalized["measure"] = f"{measure}_row_normalized"
    return normalized


def compute_cross_encoding_matrix(
    trainers: Dict[str, object],
    *,
    split: str = "test",
    max_size: Optional[int] = None,
    use_cache: bool = True,
    metric: str = "positive_mean",
    full_eval: Optional[bool] = None,
    run_evaluation: bool = True,
    allow_incomplete: bool = False,
    seed_selection: Optional[str] = None,
    seed_aggregate: str = "mean",
    dispersion: Optional[str] = None,
    positive_class_by_trainer: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Compare how trainers encode each other's binary target-pair data.

    Rows are source trainers. Columns are target class pairs. For every row
    trainer, the function obtains encoder outputs on a full evaluation set,
    subsets those outputs to the column trainer's two target classes, and
    aggregates the requested metric. Binary pairs must either use a
    ``non_``/``non-`` negative class naming convention or set
    ``TrainingArguments.positive_class`` explicitly.

    Args:
        trainers: Mapping from display/trainer id to trainer object. Each
            trainer must expose exactly two ``target_classes``.
        split: Data split for encoder evaluation and cache lookup.
        max_size: Optional maximum number of examples used by encoder
            evaluation/cache keys.
        use_cache: Whether cached encoder outputs may be loaded. With
            ``run_evaluation=True``, this is also passed through to
            ``trainer.evaluate_encoder`` for best-seed evaluation.
        metric: Cell metric. Supported values are ``"positive_mean"``,
            ``"negative_mean"``, and ``"positive_minus_negative"``.
        full_eval: Controls whether encoder evaluation includes other classes.
            ``None`` defaults to ``True`` for ``split="test"`` and ``False``
            otherwise. This becomes ``include_other_classes`` in
            ``trainer.evaluate_encoder``.
        run_evaluation: If ``True``, missing caches are computed by loading the
            relevant model and calling ``trainer.evaluate_encoder``. If
            ``False``, missing caches raise unless ``allow_incomplete=True``.
        allow_incomplete: If ``True``, missing row data or empty row/column
            subsets produce ``NaN`` cells instead of raising.
        seed_selection: ``"best"`` compares the selected best model/cache.
            ``"all_convergent"`` evaluates saved convergent seed runs and
            aggregates their per-seed scores.
        seed_aggregate: Aggregation for multi-seed cells. Supported values are
            ``"mean"``, ``"median"``, ``"min"``, and ``"max"``.
        dispersion: Optional dispersion metadata for multi-seed cells.
            Supported values are ``"none"``, ``"std"``, ``"range"``, and
            ``"minmax"``.
        positive_class_by_trainer: Optional explicit positive class by trainer
            id. This is used by positive trainer suites whose positive/negative
            pairs do not follow the ``non_``/``non-`` naming heuristic.

    Returns:
        A comparison payload with ``measure``, ``model_ids``, ``rows``,
        ``columns``, ``matrix``, ``split``, ``max_size``, ``metric``,
        ``positive_class_by_column``, ``negative_class_by_column``,
        ``available_mask``, ``full_eval``, ``run_evaluation``,
        ``allow_incomplete``, ``seed_selection``, ``seed_aggregate``,
        ``dispersion``, ``n_matrix``, ``cell_stats``, and ``multi_seed``.
        ``global_n`` or ``global_n_range`` is included when seed counts are
        available.

    Raises:
        ValueError: If fewer than two trainers are passed, a trainer is not a
            binary target-pair trainer, the positive class cannot be inferred,
            an argument value is unsupported, required encoder data is missing,
            or a row/column subset is empty while ``allow_incomplete=False``.
    """
    if not isinstance(trainers, dict) or len(trainers) < 2:
        raise ValueError("trainers must be a dict with at least 2 trainers")
    if metric not in {"positive_mean", "negative_mean", "positive_minus_negative"}:
        raise ValueError(
            "Currently supported cross-encoding metrics are "
            "'positive_mean', 'negative_mean', and 'positive_minus_negative'"
        )
    seed_selection = resolve_seed_selection_for_trainers(trainers, seed_selection)
    if dispersion is None:
        dispersion = resolve_dispersion_for_trainers(trainers, None)
    _validate_aggregate_dispersion_combo(seed_aggregate, dispersion)
    include_other_classes = _resolve_full_eval(full_eval, split)
    ids = list(trainers.keys())
    explicit_positive_by_trainer = {
        str(key): str(value)
        for key, value in (positive_class_by_trainer or {}).items()
        if value is not None
    }
    positive_by_col: Dict[str, str] = {}
    negative_by_col: Dict[str, str] = {}
    pair_by_col: Dict[str, List[str]] = {}
    row_encoder_dfs: Dict[str, Any] = {}
    for trainer_id in ids:
        trainer = trainers[trainer_id]
        target_classes = [str(c) for c in (trainer.target_classes or [])]
        training_args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
        explicit_positive = explicit_positive_by_trainer.get(str(trainer_id))
        if explicit_positive is None and training_args is not None:
            explicit_positive = getattr(training_args, "positive_class", None)
        positive_by_col[trainer_id] = _infer_positive_class_from_pair(target_classes, explicit_positive)
        negative_by_col[trainer_id] = next(
            cls for cls in target_classes if str(cls) != str(positive_by_col[trainer_id])
        )
        pair_by_col[trainer_id] = target_classes
        if seed_selection == "best":
            encoder_df = _load_cached_encoder_df(trainer, split=split, max_size=max_size) if use_cache else None
            if encoder_df is None and run_evaluation:
                eval_model = _load_eval_model_for_trainer(trainer)
                try:
                    eval_result = trainer.evaluate_encoder(
                        model_with_gradiend=eval_model,
                        split=split,
                        max_size=max_size,
                        use_cache=use_cache,
                        return_df=True,
                        plot=False,
                        include_other_classes=include_other_classes,
                    )
                finally:
                    del eval_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                encoder_df = eval_result.get("encoder_df") if isinstance(eval_result, dict) else None
            row_encoder_dfs[trainer_id] = encoder_df
        else:
            seed_paths = [
                seed_path
                for _, seed_path in resolve_seed_run_entries(trainer, seed_selection)
            ]
            if not seed_paths:
                seed_paths = [trainer.model_path]
            encoder_dfs: List[Any] = []
            best_seed_path = trainer.get_best_seed_run_path() if hasattr(trainer, "get_best_seed_run_path") else None
            reused_best_cache = False
            if use_cache and best_seed_path is not None:
                cached_best_df = _load_cached_encoder_df(trainer, split=split, max_size=max_size)
                if cached_best_df is not None:
                    encoder_dfs.append(cached_best_df)
                    reused_best_cache = True
            for seed_path in seed_paths:
                if not isinstance(seed_path, str) or not os.path.isdir(seed_path):
                    continue
                if reused_best_cache and best_seed_path is not None and os.path.normcase(seed_path) == os.path.normcase(best_seed_path):
                    continue
                eval_model = _load_eval_model_for_trainer(trainer, load_directory=seed_path)
                training_args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
                original_experiment_dir = getattr(training_args, "experiment_dir", None) if training_args is not None else None
                try:
                    if training_args is not None:
                        training_args.experiment_dir = None
                    eval_result = trainer.evaluate_encoder(
                        model_with_gradiend=eval_model,
                        split=split,
                        max_size=max_size,
                        use_cache=False,
                        return_df=True,
                        plot=False,
                        include_other_classes=include_other_classes,
                    )
                finally:
                    if training_args is not None:
                        training_args.experiment_dir = original_experiment_dir
                    del eval_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                encoder_df = eval_result.get("encoder_df") if isinstance(eval_result, dict) else None
                if encoder_df is not None:
                    encoder_dfs.append(encoder_df)
            row_encoder_dfs[trainer_id] = encoder_dfs
        value = row_encoder_dfs[trainer_id]
        if value is None or (isinstance(value, list) and len(value) == 0):
            if allow_incomplete:
                row_encoder_dfs[trainer_id] = None
                continue
            raise ValueError(
                f"Trainer {trainer_id!r} has no full encoder data for cross-encoding. "
                "Run suite.evaluate_encoder(split='test', full_eval=True, return_df=True) first, or pass run_evaluation=True."
            )
    matrix = [[float("nan")] * len(ids) for _ in range(len(ids))]
    available_mask = [[False] * len(ids) for _ in range(len(ids))]
    n_matrix = [[0] * len(ids) for _ in range(len(ids))]
    cell_stats: List[List[Dict[str, Any]]] = []
    all_n: List[int] = []
    for i, row_id in enumerate(ids):
        row_df_value = row_encoder_dfs[row_id]
        row_dfs = row_df_value if isinstance(row_df_value, list) else [row_df_value]
        if len(row_dfs) == 1 and row_dfs[0] is None:
            continue
        stats_row: List[Dict[str, Any]] = []
        for j, col_id in enumerate(ids):
            scores: List[float] = []
            for row_df in row_dfs:
                subset_df = _subset_encoder_df_for_target_classes(row_df, pair_by_col[col_id])
                if len(subset_df) == 0:
                    continue
                positive_mean = _extract_positive_mean_from_df(subset_df, positive_by_col[col_id])
                if metric == "positive_mean":
                    scores.append(positive_mean)
                    continue
                negative_mean = _extract_negative_mean_from_df(subset_df, negative_by_col[col_id])
                if metric == "negative_mean":
                    scores.append(negative_mean)
                else:
                    scores.append(positive_mean - negative_mean)
            if not scores:
                if allow_incomplete:
                    stats = {"aggregate": float("nan"), "n": 0}
                    stats_row.append(stats)
                    continue
                raise ValueError(
                    f"Cross-encoding found no rows for column pair {pair_by_col[col_id]!r} in row cache {row_id!r}. "
                    "This likely means the encoder cache was not created with include_other_classes=True."
                )
            stats = _aggregate_seed_scores(scores, seed_aggregate=seed_aggregate, dispersion=dispersion)
            matrix[i][j] = float(stats["aggregate"])
            available_mask[i][j] = True
            n_matrix[i][j] = int(stats["n"])
            stats_row.append(stats)
            all_n.append(int(stats["n"]))
        cell_stats.append(stats_row)
    payload: Dict[str, Any] = {
        "measure": f"cross_encoding_{metric}",
        "model_ids": ids,
        "matrix": matrix,
        "rows": ids,
        "columns": ids,
        "split": split,
        "max_size": max_size,
        "metric": metric,
        "positive_class_by_column": positive_by_col,
        "negative_class_by_column": negative_by_col,
        "available_mask": available_mask,
        "full_eval": include_other_classes,
        "run_evaluation": run_evaluation,
        "allow_incomplete": allow_incomplete,
        "seed_selection": seed_selection,
        "seed_aggregate": seed_aggregate,
        "dispersion": dispersion,
        "explicit_positive_class_by_trainer": explicit_positive_by_trainer,
        "n_matrix": n_matrix,
        "cell_stats": cell_stats,
        "multi_seed": seed_selection != "best",
    }
    if all_n:
        if min(all_n) == max(all_n):
            payload["global_n"] = int(all_n[0])
        else:
            payload["global_n_range"] = [int(min(all_n)), int(max(all_n))]
    return payload
