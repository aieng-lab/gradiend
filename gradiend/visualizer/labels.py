"""
Shared plot label helpers (e.g. non-convergence markers).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from gradiend.trainer.core.stats import load_training_stats

# Latin cross — marks non-converged runs (distinct from ✗ which can look like a checkbox).
NON_CONVERGENCE_MARKER = "✝"
NON_CONVERGENCE_MARKER_TEX = r"\textdagger{}"

PLOTLY_LABEL_OVERRIDES = {
    "color": "Label",
    "data_split": "Split",
    "display_text": "Text",
    "encoded": "Encoded value",
    "factual": "Factual",
    "factual_token": "Factual token",
    "feature_class": "Feature class",
    "label": "Label",
    "masked": "Text",
    "plot_hue": "Split",
    "sentence": "Text",
    "source_id": "Source",
    "target": "Target",
    "target_id": "Target class",
    "target_token": "Target token",
    "text": "Text",
    "text_hover": "Text",
    "type": "Type",
}


def format_plotly_label(column: Any) -> str:
    """Return a user-facing label for Plotly axes, legends, and hover fields."""
    text = str(column)
    key = text.strip().casefold()
    if key in PLOTLY_LABEL_OVERRIDES:
        return PLOTLY_LABEL_OVERRIDES[key]
    for suffix in ("_:hover", "_hover", ":hover"):
        if key.endswith(suffix):
            base = key[: -len(suffix)].strip("_:")
            if base in PLOTLY_LABEL_OVERRIDES:
                return PLOTLY_LABEL_OVERRIDES[base]
            key = base
            break
    if key == "id":
        return "ID"
    return key.replace("_", " ").capitalize()


def plotly_labels_for(columns: Any) -> Dict[str, str]:
    """Build a Plotly labels mapping for the provided column names."""
    return {str(column): format_plotly_label(column) for column in columns if column is not None}


def converged_from_run_info(run_info: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Read convergence status from a training-stats payload.

    Args:
        run_info: Parsed ``training.json`` payload.
    """
    if not run_info:
        return None
    convergence_info = run_info.get("convergence_info")
    if isinstance(convergence_info, dict) and "converged" in convergence_info:
        return bool(convergence_info.get("converged"))
    return None


def converged_for_model_path(model_path: Optional[str]) -> Optional[bool]:
    """Read convergence status for a saved model path.

    Args:
        model_path: Directory containing ``training.json``.
    """
    if not model_path:
        return None
    try:
        return converged_from_run_info(load_training_stats(model_path))
    except Exception:
        return None


def converged_for_trainer(trainer: Any) -> Optional[bool]:
    """Resolve convergence status from a trainer or its saved model path.

    Args:
        trainer: Trainer-like object exposing training stats or model paths.
    """
    if trainer is None:
        return None
    get_stats = getattr(trainer, "get_training_stats", None)
    if get_stats is not None:
        try:
            run_info = get_stats()
            converged = converged_from_run_info(run_info)
            if converged is not None:
                return converged
        except Exception:
            pass
    get_model = getattr(trainer, "get_model", None)
    model_path = None
    if get_model is not None:
        try:
            model = get_model()
            model_path = getattr(model, "name_or_path", None)
        except Exception:
            pass
    if model_path is None:
        model_path = getattr(trainer, "model_path", None) or getattr(trainer, "experiment_dir", None)
    return converged_for_model_path(model_path)


def resolve_highlight_non_convergence(
    highlight_non_convergence: Optional[bool],
    *,
    trainer: Any = None,
    training_args: Any = None,
) -> bool:
    """Resolve highlight flag: explicit arg > trainer.training_args > default True.

    Args:
        highlight_non_convergence: Explicit override.
        trainer: Optional trainer whose training args provide the default.
        training_args: Optional training args object used before trainer lookup.
    """
    if highlight_non_convergence is not None:
        return bool(highlight_non_convergence)
    args = training_args
    if args is None and trainer is not None:
        args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
    if args is not None:
        return bool(getattr(args, "highlight_non_convergence", True))
    return True


def format_label_with_convergence(
    label: str,
    *,
    converged: Optional[bool] = None,
    highlight_non_convergence: bool = True,
) -> str:
    """Append the non-convergence marker when highlight is enabled and the run did not converge.

    Args:
        label: Base display label.
        converged: Whether the corresponding run converged.
        highlight_non_convergence: Whether to append the marker for non-converged runs.
    """
    text = str(label)
    if not highlight_non_convergence or converged is not False:
        return text
    marker = non_convergence_marker_for_matplotlib()
    if text.endswith(marker) or text.endswith(NON_CONVERGENCE_MARKER):
        return text
    return f"{text} {marker}"


def non_convergence_marker_for_matplotlib() -> str:
    """Return a non-convergence marker safe for the active Matplotlib text mode."""
    try:
        import matplotlib as mpl

        if bool(mpl.rcParams.get("text.usetex", False)):
            return NON_CONVERGENCE_MARKER_TEX
    except Exception:
        pass
    return NON_CONVERGENCE_MARKER


def resolve_plot_title_with_convergence(
    title: Union[str, bool, None],
    *,
    trainer: Any = None,
    run_info: Optional[Dict[str, Any]] = None,
    highlight_non_convergence: bool = True,
    default: str = "Training convergence",
) -> Union[str, bool]:
    """Resolve plot title and append non-convergence marker when applicable.

    Args:
        title: Explicit title, True for default, or None/False to disable.
        trainer: Optional trainer used for run id and convergence lookup.
        run_info: Optional parsed training stats.
        highlight_non_convergence: Whether to mark non-converged runs.
        default: Fallback title when no trainer run id is available.
    """
    if title is False or title is None:
        return False
    converged = None
    if run_info is not None:
        converged = converged_from_run_info(run_info)
    elif trainer is not None:
        converged = converged_for_trainer(trainer)
    if title is True:
        base = getattr(trainer, "run_id", None) if trainer is not None else None
        base = base or default
    else:
        base = str(title)
    if not highlight_non_convergence:
        return base
    return format_label_with_convergence(
        base,
        converged=converged,
        highlight_non_convergence=True,
    )


def _aggregate_contributor_convergence(values: Iterable[Optional[bool]]) -> Optional[bool]:
    """Return False if any contributor did not converge; True if all did."""
    observed = list(values)
    if not observed:
        return None
    if any(value is False for value in observed):
        return False
    if all(value is True for value in observed):
        return True
    return None


def converged_by_trainer_id(trainers: Optional[Dict[str, Any]]) -> Dict[str, Optional[bool]]:
    """Map trainer id to convergence status."""
    if not trainers:
        return {}
    return {str(trainer_id): converged_for_trainer(trainer) for trainer_id, trainer in trainers.items()}


def resolve_axis_convergence_for_comparison_heatmap(
    comparison_data: Dict[str, Any],
    *,
    models: Optional[Dict[str, Any]] = None,
    row_ids: Optional[Sequence[str]] = None,
    column_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Optional[bool]], Dict[str, Optional[bool]]]:
    """Resolve row/column convergence when axis ids are not trainer ids.

    For oriented cross-encoding matrices, a feature label is marked non-converged
    when any GRADIEND that contributed to that axis value did not converge.
    For GRADIEND × feature-class matrices, column labels use the same rule while
    row labels remain per-trainer.

    Args:
        comparison_data: Heatmap payload from comparison matrix helpers.
        models: Trainer mapping used for convergence lookup.
        row_ids: Final row axis ids after ordering.
        column_ids: Final column axis ids after ordering.
    """
    row_status: Dict[str, Optional[bool]] = {}
    col_status: Dict[str, Optional[bool]] = {}
    if not models:
        return row_status, col_status

    converged_map = converged_by_trainer_id(models)
    aligned_rows = comparison_data.get("aligned_rows")
    if aligned_rows is not None:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(aligned_rows, pd.DataFrame) and not aligned_rows.empty:
            row_contributors: Dict[str, list[Optional[bool]]] = defaultdict(list)
            col_contributors: Dict[str, list[Optional[bool]]] = defaultdict(list)
            for _, row in aligned_rows.iterrows():
                trainer_id = str(row.get("trainer_id", ""))
                converged = converged_map.get(trainer_id)
                anchor = row.get("anchor_class")
                if anchor is not None and str(anchor):
                    row_contributors[str(anchor)].append(converged)
                eval_class = row.get("eval_class")
                if eval_class is None:
                    eval_class = row.get("aligned_column")
                if eval_class is not None and str(eval_class):
                    col_contributors[str(eval_class)].append(converged)
            row_status = {
                axis_id: _aggregate_contributor_convergence(values)
                for axis_id, values in row_contributors.items()
            }
            col_status = {
                axis_id: _aggregate_contributor_convergence(values)
                for axis_id, values in col_contributors.items()
            }
            return row_status, col_status

    measure = str(comparison_data.get("measure", ""))
    n_matrix = comparison_data.get("n_matrix")
    trainer_ids = [str(value) for value in comparison_data.get("model_ids", [])]
    columns = [str(value) for value in (column_ids or comparison_data.get("column_ids") or [])]
    if (
        n_matrix
        and trainer_ids
        and columns
        and measure.startswith(("gradiend_feature_cross_encoding_", "gradiend_transition_cross_encoding_"))
    ):
        row_status = {trainer_id: converged_map.get(trainer_id) for trainer_id in trainer_ids}
        col_contributors: Dict[str, list[Optional[bool]]] = defaultdict(list)
        for row_index, trainer_id in enumerate(trainer_ids):
            converged = converged_map.get(trainer_id)
            row_counts = n_matrix[row_index] if row_index < len(n_matrix) else []
            for col_index, column_id in enumerate(columns):
                count = row_counts[col_index] if col_index < len(row_counts) else 0
                if count and int(count) > 0:
                    col_contributors[column_id].append(converged)
        col_status = {
            axis_id: _aggregate_contributor_convergence(values)
            for axis_id, values in col_contributors.items()
        }
        return row_status, col_status

    if row_ids is not None:
        for axis_id in row_ids:
            key = str(axis_id)
            if key in converged_map:
                row_status[key] = converged_map[key]
    if column_ids is not None:
        for axis_id in column_ids:
            key = str(axis_id)
            if key in converged_map:
                col_status[key] = converged_map[key]
    return row_status, col_status


def format_model_labels_with_convergence(
    model_ids: list,
    *,
    models: Optional[Dict[str, Any]] = None,
    converged_by_id: Optional[Dict[str, Optional[bool]]] = None,
    highlight_non_convergence: bool = True,
) -> Dict[str, str]:
    """Map model_id -> display label, optionally suffixing non-convergence marker.

    Args:
        model_ids: Model identifiers to format.
        models: Optional model mapping used for convergence lookup.
        converged_by_id: Optional explicit convergence status by model id.
        highlight_non_convergence: Whether to mark non-converged runs.
    """
    out: Dict[str, str] = {}
    for mid in model_ids:
        key = str(mid)
        converged = None
        if converged_by_id is not None:
            converged = converged_by_id.get(mid)
            if converged is None:
                converged = converged_by_id.get(key)
        elif models is not None and mid in models:
            model = models[mid]
            model_path = getattr(model, "name_or_path", None)
            converged = converged_for_model_path(model_path)
        out[key] = format_label_with_convergence(
            key,
            converged=converged,
            highlight_non_convergence=highlight_non_convergence,
        )
    return out
