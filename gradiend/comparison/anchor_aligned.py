"""Anchor-aligned encoding matrices for symmetric pairwise trainer suites."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from gradiend.comparison.common import _safe_float


def _normalise_alignment(alignment: str) -> str:
    aliases = {
        "factual": "factual",
        "fact": "factual",
        "counterfactual": "counterfactual",
        "cf": "counterfactual",
        "alternative": "counterfactual",
        "alternatives": "counterfactual",
        "transition": "transition",
        "transitions": "transition",
    }
    key = str(alignment).strip().lower()
    if key not in aliases:
        raise ValueError("alignment must be one of: factual, counterfactual, transition")
    return aliases[key]


def _row_value(row: pd.Series, *names: str) -> Optional[str]:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            value = str(row[name])
            if value:
                return value
    return None


def _transition_id_from_row(row: pd.Series) -> Optional[str]:
    transition = _row_value(row, "transition_id", "feature_class_id")
    if transition:
        return transition
    factual = _row_value(row, "factual_id", "source_id")
    counterfactual = _row_value(row, "counterfactual_id", "target_id")
    if factual and counterfactual:
        return f"{factual}->{counterfactual}"
    return factual


def _aligned_column_from_row(row: pd.Series, alignment: str) -> Optional[str]:
    if alignment == "factual":
        return _row_value(row, "factual_id", "source_id")
    if alignment == "counterfactual":
        return _row_value(row, "counterfactual_id", "target_id")
    if alignment == "transition":
        return _transition_id_from_row(row)
    raise ValueError(f"Unsupported alignment {alignment!r}")


def build_anchor_aligned_encoding_rows(
    *,
    pair_by_id: Dict[str, Tuple[str, str]],
    encoder_summary: Dict[str, Any],
    feature_classes: Sequence[str],
    alignment: str = "factual",
    column_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build oriented rows used by anchor-aligned encoding matrices.

    Raw encoder rows are first averaged within each ``(trainer, transition,
    aligned_column)`` cell. Each binary GRADIEND then contributes once for its
    left class with sign ``+1`` and once for its right class with sign ``-1`` so
    pairwise models can be aligned into a shared feature-class coordinate frame.
    Neutral rows are skipped when a ``type`` column marks them as neutral.

    Args:
        pair_by_id: Mapping from trainer/model id to its ordered binary class
            pair ``(left_class, right_class)``.
        encoder_summary: Mapping from trainer/model id to an evaluation result
            dict containing an ``encoder_df`` DataFrame and optionally a
            ``correlation`` value.
        feature_classes: Feature classes that may appear as anchor rows.
        alignment: Column alignment mode. Supported aliases resolve to
            ``"factual"``, ``"counterfactual"``, or ``"transition"``.
            Factual columns use the factual/source class, counterfactual columns
            use the alternative/target class, and transition columns use
            directed ``source->target`` ids.
        column_ids: Optional whitelist/order for aligned columns. Rows outside
            this set are dropped.

    Returns:
        A DataFrame with columns including ``trainer_id``, ``pair_left``,
        ``pair_right``, ``anchor_class``, ``eval_class``, ``transition_id``,
        ``aligned_mean``, ``source_mean``, ``sign``, ``raw_n``,
        ``contribution_n``, ``alignment``, and ``correlation``.

    Raises:
        ValueError: If ``alignment`` is unsupported.
    """
    alignment = _normalise_alignment(alignment)
    rows: List[Dict[str, Any]] = []
    valid_classes = {str(value) for value in feature_classes}
    valid_columns = {str(value) for value in column_ids} if column_ids is not None else None
    for trainer_id, result in encoder_summary.items():
        if trainer_id not in pair_by_id:
            continue
        left, right = (str(pair_by_id[trainer_id][0]), str(pair_by_id[trainer_id][1]))
        encoder_df = result.get("encoder_df") if isinstance(result, dict) else None
        if not isinstance(encoder_df, pd.DataFrame) or encoder_df.empty:
            continue
        if "type" in encoder_df.columns:
            non_neutral_mask = ~encoder_df["type"].astype(str).str.contains("neutral", case=False, na=False)
            encoder_df = encoder_df[non_neutral_mask].copy()
        if encoder_df.empty or "encoded" not in encoder_df.columns:
            continue
        work = encoder_df.copy()
        work["_transition_id"] = work.apply(_transition_id_from_row, axis=1)
        work["_aligned_column"] = work.apply(lambda row: _aligned_column_from_row(row, alignment), axis=1)
        work = work[work["_transition_id"].notna() & work["_aligned_column"].notna()].copy()
        if valid_columns is not None:
            work = work[work["_aligned_column"].astype(str).isin(valid_columns)].copy()
        if work.empty:
            continue
        contributions = (
            work.groupby(["_transition_id", "_aligned_column"], dropna=False)["encoded"]
            .agg(["mean", "count"])
            .reset_index()
        )
        corr = _safe_float(result.get("correlation")) if isinstance(result, dict) else None
        for anchor, sign in ((left, 1.0), (right, -1.0)):
            if anchor not in valid_classes:
                continue
            for _, contribution in contributions.iterrows():
                aligned_column = str(contribution["_aligned_column"])
                value = _safe_float(contribution["mean"])
                if value is None:
                    continue
                rows.append(
                    {
                        "trainer_id": trainer_id,
                        "pair_left": left,
                        "pair_right": right,
                        "anchor_class": anchor,
                        "eval_class": aligned_column,
                        "aligned_column": aligned_column,
                        "transition_id": str(contribution["_transition_id"]),
                        "aligned_mean": sign * value,
                        "source_mean": value,
                        "sign": sign,
                        "raw_n": int(contribution["count"]),
                        "contribution_n": 1,
                        "alignment": alignment,
                        "correlation": corr,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_anchor_aligned_encoding_rows(
    aligned_df: pd.DataFrame,
    feature_classes: Sequence[str],
    *,
    column_ids: Optional[Sequence[str]] = None,
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Pivot anchor-aligned rows into a feature-class by column matrix.

    Args:
        aligned_df: DataFrame produced by
            :func:`build_anchor_aligned_encoding_rows`.
        feature_classes: Row order for the returned matrix.
        column_ids: Optional column order. Defaults to ``feature_classes``.
        aggregate: Aggregation applied to ``aligned_mean``. Supported values are
            ``"mean"``, ``"min"``, ``"max"``, and ``"std"``. ``"count"``
            sums ``contribution_n`` and ``"raw_count"`` sums ``raw_n``.

    Returns:
        A pandas DataFrame indexed by ``feature_classes`` and with columns
        ``column_ids``. Missing cells are left as ``NaN``.

    Raises:
        ValueError: If ``aggregate`` is unsupported.
    """
    classes = [str(value) for value in feature_classes]
    columns = [str(value) for value in (column_ids if column_ids is not None else feature_classes)]
    empty = pd.DataFrame(index=classes, columns=columns, dtype=float)
    if aligned_df.empty:
        return empty
    if aggregate not in {"mean", "min", "max", "std", "count", "raw_count"}:
        raise ValueError("aggregate must be one of 'mean', 'min', 'max', 'std', 'count', or 'raw_count'")
    if aggregate == "count":
        grouped = (
            aligned_df.groupby(["anchor_class", "eval_class"], dropna=False)["contribution_n"]
            .sum()
            .reset_index(name="value")
        )
    elif aggregate == "raw_count":
        grouped = (
            aligned_df.groupby(["anchor_class", "eval_class"], dropna=False)["raw_n"]
            .sum()
            .reset_index(name="value")
        )
    else:
        grouped = (
            aligned_df.groupby(["anchor_class", "eval_class"], dropna=False)["aligned_mean"]
            .agg(aggregate)
            .reset_index(name="value")
        )
    pivot = grouped.pivot(index="anchor_class", columns="eval_class", values="value")
    return pivot.reindex(index=classes, columns=columns)


def compute_anchor_aligned_encoding_matrix(
    *,
    pair_by_id: Dict[str, Tuple[str, str]],
    encoder_summary: Dict[str, Any],
    feature_classes: Sequence[str],
    aggregate: str = "mean",
    alignment: str = "factual",
    column_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute an oriented cross-encoding matrix for symmetric pairwise GRADIENDs.

    This helper is intended for symmetric suites where many binary GRADIENDs
    cover overlapping feature-class pairs. Rows index an anchor feature class,
    and signs are aligned so a class is comparable whether it appeared as the
    left or right class in an individual binary pair.

    Args:
        pair_by_id: Mapping from trainer/model id to its ordered binary class
            pair ``(left_class, right_class)``.
        encoder_summary: Mapping from trainer/model id to an evaluation result
            dict containing an ``encoder_df`` DataFrame and optionally a
            ``correlation`` value.
        feature_classes: Row order and default column order. Must contain at
            least two classes.
        aggregate: Aggregation applied to aligned contributions. Supported
            values are ``"mean"``, ``"min"``, ``"max"``, ``"std"``,
            ``"count"``, and ``"raw_count"``.
        alignment: Column alignment mode. Supported aliases resolve to
            ``"factual"``, ``"counterfactual"``, or ``"transition"``.
        column_ids: Optional explicit column order/filter. If omitted,
            factual/counterfactual alignment uses ``feature_classes`` and
            transition alignment uses observed transition ids.

    Returns:
        A payload with ``measure``, ``model_ids``, ``column_ids``, ``rows``,
        ``columns``, ``matrix``, ``aggregate``, ``alignment``, ``aligned_rows``,
        ``n_matrix``, ``raw_n_matrix``, and ``pair_by_trainer``.

    Raises:
        ValueError: If too few feature classes are passed, ``alignment`` is
            unsupported, or ``aggregate`` is unsupported.
    """
    alignment = _normalise_alignment(alignment)
    classes = [str(value) for value in feature_classes]
    if len(classes) < 2:
        raise ValueError("feature_classes must contain at least 2 classes")
    columns = [str(value) for value in column_ids] if column_ids is not None else None
    aligned_df = build_anchor_aligned_encoding_rows(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=classes,
        alignment=alignment,
        column_ids=columns,
    )
    if columns is None:
        if alignment == "transition" and not aligned_df.empty:
            columns = sorted(aligned_df["eval_class"].dropna().astype(str).unique().tolist())
        else:
            columns = list(classes)
    matrix_df = aggregate_anchor_aligned_encoding_rows(
        aligned_df,
        classes,
        column_ids=columns,
        aggregate=aggregate,
    )
    matrix = matrix_df.astype(float).values.tolist()
    counts_df = aggregate_anchor_aligned_encoding_rows(aligned_df, classes, column_ids=columns, aggregate="count")
    raw_counts_df = aggregate_anchor_aligned_encoding_rows(aligned_df, classes, column_ids=columns, aggregate="raw_count")
    return {
        "measure": f"anchor_aligned_encoding_{alignment}_{aggregate}",
        "model_ids": classes,
        "column_ids": columns,
        "matrix": matrix,
        "rows": classes,
        "columns": columns,
        "aggregate": aggregate,
        "alignment": alignment,
        "aligned_rows": aligned_df,
        "n_matrix": counts_df.fillna(0).astype(int).values.tolist(),
        "raw_n_matrix": raw_counts_df.fillna(0).astype(int).values.tolist(),
        "pair_by_trainer": {str(k): [str(v) for v in pair] for k, pair in pair_by_id.items()},
    }


def pair_by_id_from_trainers(trainers: Dict[str, object]) -> Dict[str, Tuple[str, str]]:
    """Extract binary target-class pairs from trainers.

    Args:
        trainers: Mapping from trainer id to trainer object. Each trainer must
            expose exactly two ``target_classes``.

    Returns:
        A mapping from string trainer id to ``(class_a, class_b)`` as strings.

    Raises:
        ValueError: If any trainer does not expose exactly two target classes.
    """
    pair_by_id: Dict[str, Tuple[str, str]] = {}
    for trainer_id, trainer in trainers.items():
        target_classes = getattr(trainer, "target_classes", None) or []
        if len(target_classes) != 2:
            raise ValueError(
                f"Trainer {trainer_id!r} must have exactly 2 target_classes for anchor-aligned encoding, "
                f"got {list(target_classes)!r}"
            )
        pair_by_id[str(trainer_id)] = (str(target_classes[0]), str(target_classes[1]))
    return pair_by_id
