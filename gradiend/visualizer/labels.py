"""
Shared plot label helpers (e.g. non-convergence markers).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Dagger — marks non-converged runs. Prefer U+2020 over the Latin cross (✝):
# the latter is missing from common Matplotlib fonts and can disappear in PDFs.
NON_CONVERGENCE_MARKER = "†"
NON_CONVERGENCE_MARKER_TEX = r"\textdagger{}"

_TRANSITION_DIRECTED_RE = re.compile(r"\s*(?:->|→)\s*")
_TRANSITION_BIDI_RE = re.compile(r"\s*(?:<->|↔)\s*")

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


def matplotlib_usetex_enabled() -> bool:
    """Whether matplotlib is currently rendering text through LaTeX."""
    try:
        import matplotlib as mpl

        return bool(mpl.rcParams.get("text.usetex", False))
    except Exception:
        return False


def label_contains_matplotlib_latex(text: Any) -> bool:
    """True when *text* includes an inline math segment for matplotlib usetex."""
    return "$" in str(text)


def transition_bidi_arrow(*, use_latex: Optional[bool] = None) -> str:
    """Bidirectional transition arrow for matplotlib labels (GRADIEND pair notation)."""
    from gradiend.visualizer.plot_style_config import resolve_transition_arrow_mode

    mode = resolve_transition_arrow_mode(use_latex=use_latex)
    if mode == "latex":
        return "$\\rightleftarrows$"
    if mode == "ascii":
        return " <-> "
    return " ↔ "


def transition_directed_arrow(*, use_latex: Optional[bool] = None) -> str:
    """Directed transition arrow for matplotlib labels (source to target)."""
    from gradiend.visualizer.plot_style_config import resolve_transition_arrow_mode

    mode = resolve_transition_arrow_mode(use_latex=use_latex)
    if mode == "latex":
        return "$\\rightarrow$"
    if mode == "ascii":
        return " -> "
    return " → "


def format_transition_label(label: Any, *, use_latex: Optional[bool] = None) -> str:
    """Replace transition delimiters with matplotlib-appropriate arrows.

    When ``use_latex`` is omitted, follows :func:`resolve_transition_arrow_mode`.
    Idempotent for labels that already contain ``$...$`` math segments.
    """
    from gradiend.visualizer.plot_style_config import resolve_transition_arrow_mode

    text = str(label)
    if label_contains_matplotlib_latex(text):
        return text
    mode = resolve_transition_arrow_mode(use_latex=use_latex)
    if mode == "latex":
        if _TRANSITION_BIDI_RE.search(text):
            return _TRANSITION_BIDI_RE.sub(
                lambda _match: transition_bidi_arrow(use_latex=True),
                text,
            )
        if _TRANSITION_DIRECTED_RE.search(text):
            return _TRANSITION_DIRECTED_RE.sub(
                lambda _match: transition_directed_arrow(use_latex=True),
                text,
            )
        return text
    if mode == "ascii":
        if _TRANSITION_BIDI_RE.search(text):
            return _TRANSITION_BIDI_RE.sub(" <-> ", text)
        return _TRANSITION_DIRECTED_RE.sub(" -> ", text)
    if _TRANSITION_BIDI_RE.search(text):
        return _TRANSITION_BIDI_RE.sub(" ↔ ", text)
    return _TRANSITION_DIRECTED_RE.sub(" → ", text)


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


def converged_from_seed_report(
    report: Optional[Dict[str, Any]],
    *,
    min_convergent_seeds: Optional[int] = None,
) -> Optional[bool]:
    """Resolve convergence from a multi-seed ``seed_report.json`` payload.

    The current requested seed requirement takes precedence over the cached
    report's requirement. That makes stale one-seed caches visibly invalid in a
    three-seed plotting run instead of silently inheriting ``min=1`` from disk.
    """
    if not isinstance(report, dict):
        return None
    required = min_convergent_seeds
    if required is None:
        cached_required = report.get("min_convergent_seeds")
        if isinstance(cached_required, int):
            required = cached_required
    if required is None:
        required = 1
    if required <= 0:
        return True
    count = report.get("convergent_count")
    if isinstance(count, int):
        return count >= required
    runs = report.get("runs")
    if isinstance(runs, list):
        observed = [bool(run.get("converged")) for run in runs if isinstance(run, dict)]
        if observed:
            return sum(observed) >= required
    return None


def converged_for_model_path(model_path: Optional[str]) -> Optional[bool]:
    """Read convergence status for a saved model path.

    Args:
        model_path: Directory containing ``training.json``.
    """
    if not model_path:
        return None
    try:
        from gradiend.trainer.core.stats import load_training_stats

        return converged_from_run_info(load_training_stats(model_path))
    except Exception:
        return None


def _min_convergent_seeds_for_trainer(trainer: Any) -> Optional[int]:
    args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
    if args is None:
        return None
    value = getattr(args, "min_convergent_seeds", None)
    return value if isinstance(value, int) else None


def converged_for_trainer(trainer: Any) -> Optional[bool]:
    """Resolve convergence status from a trainer or its saved model path.

    Args:
        trainer: Trainer-like object exposing training stats or model paths.
    """
    if trainer is None:
        return None
    required = _min_convergent_seeds_for_trainer(trainer)
    get_seed_report = getattr(trainer, "get_seed_report", None)
    if get_seed_report is not None:
        try:
            converged = converged_from_seed_report(
                get_seed_report(),
                min_convergent_seeds=required,
            )
            if converged is not None:
                return converged
        except Exception:
            pass
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
    marker: Optional[str] = None,
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
    marker = marker if marker is not None else non_convergence_marker_for_matplotlib()
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
        if base is None:  # no run id and no default: no title (not the text "None")
            return False
    else:
        base = str(title)
    if not highlight_non_convergence:
        return base
    return format_label_with_convergence(
        base,
        converged=converged,
        highlight_non_convergence=True,
    )


def converged_by_trainer_id(trainers: Optional[Dict[str, Any]]) -> Dict[str, Optional[bool]]:
    """Map trainer id to convergence status."""
    if not trainers:
        return {}
    return {str(trainer_id): converged_for_trainer(trainer) for trainer_id, trainer in trainers.items()}


def _transition_axis_ids_for_pair(left: str, right: str) -> set[str]:
    return {
        f"{left}_{right}",
        f"{left}->{right}",
        f"{right}->{left}",
        f"{left}<->{right}",
        f"{right}<->{left}",
        f"{left}→{right}",
        f"{right}→{left}",
        f"{left}↔{right}",
        f"{right}↔{left}",
    }


def _unique_pair_axis_convergence(
    comparison_data: Dict[str, Any],
    converged_map: Dict[str, Optional[bool]],
) -> Dict[str, Optional[bool]]:
    """Map axis ids to convergence only when they identify exactly one trainer."""
    pair_by_trainer = comparison_data.get("pair_by_trainer")
    if not isinstance(pair_by_trainer, dict):
        return {}
    candidates: Dict[str, list[Optional[bool]]] = {}
    for trainer_id, pair in pair_by_trainer.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        converged = converged_map.get(str(trainer_id))
        for axis_id in _transition_axis_ids_for_pair(left, right):
            candidates.setdefault(axis_id, []).append(converged)
    return {
        axis_id: statuses[0]
        for axis_id, statuses in candidates.items()
        if len(statuses) == 1
    }


def resolve_axis_convergence_for_comparison_heatmap(
    comparison_data: Dict[str, Any],
    *,
    models: Optional[Dict[str, Any]] = None,
    row_ids: Optional[Sequence[str]] = None,
    column_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Optional[bool]], Dict[str, Optional[bool]]]:
    """Resolve row/column convergence when axis ids are not trainer ids.

    Convergence is a property of trained GRADIEND runs, not of broad derived
    feature classes. Therefore axes are marked only when their ids directly name
    a trainer/model, or when a transition/pair axis id uniquely resolves to one
    trainer. Ambiguous shared feature axes intentionally remain unmarked.

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
        return row_status, col_status

    pair_axis_status = _unique_pair_axis_convergence(comparison_data, converged_map)
    if row_ids is not None:
        for axis_id in row_ids:
            key = str(axis_id)
            if key in converged_map:
                row_status[key] = converged_map[key]
            elif key in pair_axis_status:
                row_status[key] = pair_axis_status[key]
    if column_ids is not None:
        for axis_id in column_ids:
            key = str(axis_id)
            if key in converged_map:
                col_status[key] = converged_map[key]
            elif key in pair_axis_status:
                col_status[key] = pair_axis_status[key]
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
            format_transition_label(key),
            converged=converged,
            highlight_non_convergence=highlight_non_convergence,
        )
    return out
