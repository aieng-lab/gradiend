"""
Lightweight metric helpers used in core training/evaluation paths.

Why this exists:
- We only need a small subset of sklearn metrics (`accuracy_score`,
  weighted `precision/recall/f1`) in this project.
- Keeping these tiny implementations local avoids a heavy hard dependency.

If future needs grow beyond these basic metrics (multilabel, sparse targets,
advanced averaging/label handling), prefer replacing this module with
`sklearn.metrics` again.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union


ZeroDivisionType = Union[int, float, str]


def _to_list(values: Iterable) -> List:
    return list(values)


def _validate_inputs(y_true: Sequence, y_pred: Sequence) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )


def _resolve_zero_division(zero_division: ZeroDivisionType) -> float:
    # sklearn accepts 0, 1, or "warn". We map "warn" to 0.0 (no warning machinery here).
    if zero_division == "warn":
        return 0.0
    if zero_division in (0, 1):
        return float(zero_division)
    raise ValueError("zero_division must be one of {0, 1, 'warn'}")


def accuracy_score(y_true: Sequence, y_pred: Sequence) -> float:
    y_true_l = _to_list(y_true)
    y_pred_l = _to_list(y_pred)
    _validate_inputs(y_true_l, y_pred_l)
    n = len(y_true_l)
    if n == 0:
        return 0.0
    return float(sum(1 for t, p in zip(y_true_l, y_pred_l) if t == p) / n)


def _labels_in_order(y_true: Sequence, y_pred: Sequence) -> List:
    labels = []
    seen = set()
    for item in list(y_true) + list(y_pred):
        if item not in seen:
            seen.add(item)
            labels.append(item)
    return labels


def _per_label_stats(y_true: Sequence, y_pred: Sequence, label) -> tuple[int, int, int, int]:
    tp = fp = fn = support = 0
    for t, p in zip(y_true, y_pred):
        if t == label:
            support += 1
        if t == label and p == label:
            tp += 1
        elif t != label and p == label:
            fp += 1
        elif t == label and p != label:
            fn += 1
    return tp, fp, fn, support


def _average_metric(values: List[float], supports: List[int], average: str) -> float:
    if not values:
        return 0.0
    if average == "macro":
        return float(sum(values) / len(values))
    if average == "weighted":
        total_support = sum(supports)
        if total_support == 0:
            return 0.0
        return float(sum(v * s for v, s in zip(values, supports)) / total_support)
    raise ValueError("average must be one of {'weighted', 'macro', 'micro'}")


def precision_score(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    average: str = "binary",
    zero_division: ZeroDivisionType = "warn",
) -> float:
    y_true_l = _to_list(y_true)
    y_pred_l = _to_list(y_pred)
    _validate_inputs(y_true_l, y_pred_l)

    if len(y_true_l) == 0:
        return 0.0

    zd = _resolve_zero_division(zero_division)
    if average == "micro":
        tp_total = sum(1 for t, p in zip(y_true_l, y_pred_l) if t == p)
        fp_total = len(y_true_l) - tp_total
        denom = tp_total + fp_total
        return float(tp_total / denom) if denom else zd

    labels = _labels_in_order(y_true_l, y_pred_l)
    precisions: List[float] = []
    supports: List[int] = []
    for label in labels:
        tp, fp, _, support = _per_label_stats(y_true_l, y_pred_l, label)
        denom = tp + fp
        p = float(tp / denom) if denom else zd
        precisions.append(p)
        supports.append(support)
    return _average_metric(precisions, supports, average)


def recall_score(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    average: str = "binary",
    zero_division: ZeroDivisionType = "warn",
) -> float:
    y_true_l = _to_list(y_true)
    y_pred_l = _to_list(y_pred)
    _validate_inputs(y_true_l, y_pred_l)

    if len(y_true_l) == 0:
        return 0.0

    zd = _resolve_zero_division(zero_division)
    if average == "micro":
        tp_total = sum(1 for t, p in zip(y_true_l, y_pred_l) if t == p)
        fn_total = len(y_true_l) - tp_total
        denom = tp_total + fn_total
        return float(tp_total / denom) if denom else zd

    labels = _labels_in_order(y_true_l, y_pred_l)
    recalls: List[float] = []
    supports: List[int] = []
    for label in labels:
        tp, _, fn, support = _per_label_stats(y_true_l, y_pred_l, label)
        denom = tp + fn
        r = float(tp / denom) if denom else zd
        recalls.append(r)
        supports.append(support)
    return _average_metric(recalls, supports, average)


def f1_score(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    average: str = "binary",
    zero_division: ZeroDivisionType = "warn",
) -> float:
    y_true_l = _to_list(y_true)
    y_pred_l = _to_list(y_pred)
    _validate_inputs(y_true_l, y_pred_l)

    if len(y_true_l) == 0:
        return 0.0

    zd = _resolve_zero_division(zero_division)
    if average == "micro":
        # For multiclass single-label, micro precision == micro recall == accuracy.
        p = precision_score(y_true_l, y_pred_l, average="micro", zero_division=zero_division)
        r = recall_score(y_true_l, y_pred_l, average="micro", zero_division=zero_division)
        denom = p + r
        return float((2.0 * p * r) / denom) if denom else zd

    labels = _labels_in_order(y_true_l, y_pred_l)
    f1s: List[float] = []
    supports: List[int] = []
    for label in labels:
        tp, fp, fn, support = _per_label_stats(y_true_l, y_pred_l, label)
        p_denom = tp + fp
        r_denom = tp + fn
        p = float(tp / p_denom) if p_denom else zd
        r = float(tp / r_denom) if r_denom else zd
        denom = p + r
        f1 = float((2.0 * p * r) / denom) if denom else zd
        f1s.append(f1)
        supports.append(support)
    return _average_metric(f1s, supports, average)

