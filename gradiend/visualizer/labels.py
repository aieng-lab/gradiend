"""
Shared plot label helpers (e.g. non-convergence markers).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from gradiend.trainer.core.stats import load_training_stats

# Latin cross — marks non-converged runs (distinct from ✗ which can look like a checkbox).
NON_CONVERGENCE_MARKER = "✝"


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
    marker = NON_CONVERGENCE_MARKER
    if text.endswith(marker):
        return text
    return f"{text} {marker}"


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
        title: Explicit title, None/True for default, or False to disable.
        trainer: Optional trainer used for run id and convergence lookup.
        run_info: Optional parsed training stats.
        highlight_non_convergence: Whether to mark non-converged runs.
        default: Fallback title when no trainer run id is available.
    """
    if title is False:
        return False
    converged = None
    if run_info is not None:
        converged = converged_from_run_info(run_info)
    elif trainer is not None:
        converged = converged_for_trainer(trainer)
    if title is True or title is None:
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
