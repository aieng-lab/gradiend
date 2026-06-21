"""Ordering and grouping helpers for heatmap visualizations."""

from __future__ import annotations

from typing import Any, Dict, List, Union


def _validate_and_complete_pretty_groups(
    pretty_groups: Dict[str, List[str]], model_ids: List[str]
) -> None:
    covered = set()
    for group_name, ids in pretty_groups.items():
        for mid in ids:
            if mid not in model_ids:
                raise ValueError(
                    f"pretty_groups: id {mid!r} in group {group_name!r} is not in model_ids"
                )
            if mid in covered:
                raise ValueError(
                    f"pretty_groups: id {mid!r} appears in multiple groups (not disjoint)"
                )
            covered.add(mid)
    missing = set(model_ids) - covered
    if missing:
        pretty_groups["Other"] = sorted(missing)


def _reorder_comparison_data(
    comparison_data: Dict[str, Any],
    *,
    order: Union[str, List[str]],
    cluster: bool,
    pretty_groups: Optional[Dict[str, List[str]]],
) -> Dict[str, Any]:
    if "column_ids" in comparison_data:
        return _reorder_rectangular_comparison_data(
            comparison_data,
            order=order,
            cluster=cluster,
            pretty_groups=pretty_groups,
        )
    original_ids = list(comparison_data["model_ids"])
    original_matrix = comparison_data["matrix"]
    original_row_labels = comparison_data.get("row_labels")
    original_column_labels = comparison_data.get("column_labels")
    index_by_id = {mid: idx for idx, mid in enumerate(original_ids)}

    if pretty_groups is not None:
        pretty_groups = {k: list(v) for k, v in pretty_groups.items()}
        _validate_and_complete_pretty_groups(pretty_groups, original_ids)
        model_ids: List[str] = []
        for ids in pretty_groups.values():
            model_ids.extend(ids)
    elif isinstance(order, list):
        model_ids = list(order)
    elif order == "name":
        model_ids = sorted(original_ids)
    elif order == "input":
        model_ids = list(original_ids)
    else:
        raise ValueError("order must be 'input', 'name', or a list of model ids")

    if cluster:
        remaining = model_ids[:]
        path = [remaining.pop(0)]
        while remaining:
            last = path[-1]
            last_idx = index_by_id[last]
            best = max(
                remaining,
                key=lambda mid: float(original_matrix[last_idx][index_by_id[mid]]),
            )
            path.append(best)
            remaining.remove(best)
        model_ids = path

    new_matrix: List[List[float]] = []
    for mi in model_ids:
        row = []
        oi = index_by_id[mi]
        for mj in model_ids:
            oj = index_by_id[mj]
            row.append(float(original_matrix[oi][oj]))
        new_matrix.append(row)

    reordered = dict(comparison_data)
    reordered["model_ids"] = model_ids
    reordered["matrix"] = new_matrix
    if isinstance(original_row_labels, dict):
        reordered["row_labels"] = {mid: original_row_labels.get(mid, mid) for mid in model_ids}
    if isinstance(original_column_labels, dict):
        reordered["column_labels"] = {mid: original_column_labels.get(mid, mid) for mid in model_ids}
    if "resolved_topk" in reordered:
        reordered["resolved_topk"] = {
            mid: reordered["resolved_topk"][mid] for mid in model_ids
        }
    if "per_model" in reordered:
        reordered["per_model"] = {
            mid: reordered["per_model"][mid] for mid in model_ids
        }
    if "n_matrix" in reordered:
        reordered["n_matrix"] = [
            [reordered["n_matrix"][index_by_id[mi]][index_by_id[mj]] for mj in model_ids]
            for mi in model_ids
        ]
    if "cell_stats" in reordered:
        reordered["cell_stats"] = [
            [reordered["cell_stats"][index_by_id[mi]][index_by_id[mj]] for mj in model_ids]
            for mi in model_ids
        ]
    if pretty_groups is not None:
        reordered["pretty_groups"] = pretty_groups
    return reordered


def _reorder_rectangular_comparison_data(
    comparison_data: Dict[str, Any],
    *,
    order: Union[str, List[str]],
    cluster: bool,
    pretty_groups: Optional[Dict[str, List[str]]],
) -> Dict[str, Any]:
    row_ids = list(comparison_data["model_ids"])
    col_ids = list(comparison_data["column_ids"])
    matrix = comparison_data["matrix"]
    index_by_row = {mid: idx for idx, mid in enumerate(row_ids)}

    if pretty_groups is not None:
        pretty_groups = {k: list(v) for k, v in pretty_groups.items()}
        _validate_and_complete_pretty_groups(pretty_groups, row_ids)
        new_row_ids: List[str] = []
        for ids in pretty_groups.values():
            new_row_ids.extend(ids)
    elif isinstance(order, list):
        new_row_ids = list(order)
    elif order == "name":
        new_row_ids = sorted(row_ids)
    elif order == "input":
        new_row_ids = list(row_ids)
    else:
        raise ValueError("order must be 'input', 'name', or a list of model ids")

    if cluster:
        remaining = new_row_ids[:]
        path = [remaining.pop(0)]
        while remaining:
            last = path[-1]
            last_idx = index_by_row[last]
            best = max(
                remaining,
                key=lambda mid: float(
                    sum(matrix[last_idx][index_by_col] for index_by_col in range(len(col_ids)))
                    / max(len(col_ids), 1)
                ),
            )
            path.append(best)
            remaining.remove(best)
        new_row_ids = path

    new_matrix = [
        [float(matrix[index_by_row[ri]][cj]) for cj in range(len(col_ids))]
        for ri in new_row_ids
    ]
    reordered = dict(comparison_data)
    reordered["model_ids"] = new_row_ids
    reordered["column_ids"] = col_ids
    reordered["matrix"] = new_matrix
    row_labels = comparison_data.get("row_labels")
    if isinstance(row_labels, dict):
        reordered["row_labels"] = {mid: row_labels.get(mid, mid) for mid in new_row_ids}
    if "n_matrix" in reordered:
        n_matrix = comparison_data["n_matrix"]
        reordered["n_matrix"] = [
            [n_matrix[index_by_row[ri]][cj] for cj in range(len(col_ids))]
            for ri in new_row_ids
        ]
    if "raw_n_matrix" in reordered:
        raw_n_matrix = comparison_data["raw_n_matrix"]
        reordered["raw_n_matrix"] = [
            [raw_n_matrix[index_by_row[ri]][cj] for cj in range(len(col_ids))]
            for ri in new_row_ids
        ]
    if pretty_groups is not None:
        reordered["pretty_groups"] = pretty_groups
    return reordered
