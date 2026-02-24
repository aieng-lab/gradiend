"""
Training convergence plot: mean_by_class and mean_by_feature_class over steps,
with correlation in a separate subplot. Best checkpoint step is marked.

Requires matplotlib. If missing, raises ImportError with install instructions.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np

from gradiend.util.paths import resolve_output_path, ARTIFACT_CONVERGENCE_PLOT
from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.trainer.core.stats import load_training_stats
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _normalize_step_keys(d: Dict[str, Any]) -> Dict[int, Any]:
    """Convert string keys to int (JSON loads step keys as strings)."""
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _steps_and_values(
    training_stats: Dict[str, Any],
    mean_by_class: bool = True,
    mean_by_feature_class: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, List[Tuple[int, float]]], Dict[str, List[Tuple[int, float]]]]:
    """
    Build step array and series for mean_by_class and mean_by_feature_class.
    Returns (steps, series_by_class, series_by_feature_class).
    """
    steps = None
    series_by_class: Dict[str, List[Tuple[int, float]]] = {}
    series_by_fc: Dict[str, List[Tuple[int, float]]] = {}

    if mean_by_class:
        mbc = training_stats.get("mean_by_class") or {}
        mbc = _normalize_step_keys(mbc) if isinstance(mbc, dict) else {}
        if mbc:
            steps = np.array(sorted(mbc.keys()))
            for step, label_means in mbc.items():
                if not isinstance(label_means, dict):
                    continue
                for label, val in label_means.items():
                    if isinstance(val, (int, float)):
                        key = str(label)
                        if key not in series_by_class:
                            series_by_class[key] = []
                        series_by_class[key].append((step, float(val)))

    if mean_by_feature_class:
        mbfc = training_stats.get("mean_by_feature_class") or {}
        mbfc = _normalize_step_keys(mbfc) if isinstance(mbfc, dict) else {}
        if mbfc:
            if steps is None:
                steps = np.array(sorted(mbfc.keys()))
            for step, fc_means in mbfc.items():
                if not isinstance(fc_means, dict):
                    continue
                for fc, val in fc_means.items():
                    if isinstance(val, (int, float)):
                        if fc not in series_by_fc:
                            series_by_fc[fc] = []
                        series_by_fc[fc].append((step, float(val)))

    if steps is None and (series_by_class or series_by_fc):
        # Build steps from union of all series steps
        all_steps = set()
        for series in list(series_by_class.values()) + list(series_by_fc.values()):
            for s, _ in series:
                all_steps.add(s)
        steps = np.array(sorted(all_steps)) if all_steps else None
    return steps, series_by_class, series_by_fc


def _class_names_from_series(
    series_by_class: Dict[str, List],
    label_value_to_class_name: Optional[Dict[str, str]],
) -> Set[str]:
    """Map series_by_class keys (label values) to class names via label_value_to_class_name."""
    names: Set[str] = set()
    lv2name = label_value_to_class_name or {}
    for k in series_by_class:
        v = lv2name.get(k)
        if isinstance(v, str):
            names.add(v)
            continue
        try:
            k_float = float(k)
            v = lv2name.get(k_float)
            if isinstance(v, str):
                names.add(v)
                continue
        except (TypeError, ValueError):
            pass
        names.add(str(k))
    return names


def _is_mean_by_feature_class_redundant(
    training_stats: Dict[str, Any],
    series_by_class: Dict[str, List],
    series_by_fc: Dict[str, List],
) -> bool:
    """
    True if mean_by_feature_class adds no information beyond mean_by_class.

    This happens when all_classes == target_classes or no identity transitions are added:
    both plots would show the same curves (target pair only). In that case the default
    for plotting mean_by_feature_class should be False.
    """
    if not series_by_fc:
        return True
    if not series_by_class:
        return False  # feature_class has data, class doesn't -> show it
    lv2name = training_stats.get("label_value_to_class_name")
    class_names = _class_names_from_series(series_by_class, lv2name)
    fc_keys = set(series_by_fc.keys())
    return class_names == fc_keys


def _correlation_series(training_stats: Dict[str, Any]) -> List[Tuple[int, float]]:
    """(step, correlation) from training_stats['scores']."""
    scores = training_stats.get("scores") or {}
    scores = _normalize_step_keys(scores) if isinstance(scores, dict) else {}
    return [(s, float(v)) for s, v in sorted(scores.items()) if isinstance(v, (int, float))]


def draw_convergence_axes(
    training_stats: Dict[str, Any],
    axes: List[Any],
    best_step_val: Optional[int] = None,
    best_corr: Optional[float] = None,
    label_name_mapping: Optional[Dict[str, str]] = None,
    plot_mean_by_class: bool = True,
    plot_mean_by_feature_class: Optional[bool] = None,
    plot_correlation: bool = True,
) -> None:
    """
    Draw convergence curves into existing axes (live or static).
    axes[0] = mean_by_class (if plot_mean_by_class), axes[1] = mean_by_feature_class (if plot_mean_by_feature_class),
    axes[2] = correlation (if plot_correlation). Caller must pass enough axes for the enabled options.

    plot_mean_by_feature_class: None = auto (False when redundant with mean_by_class, True otherwise).
    """
    ts = training_stats
    steps, series_by_class, series_by_fc = _steps_and_values(
        ts, mean_by_class=plot_mean_by_class, mean_by_feature_class=plot_mean_by_feature_class is not False
    )
    if plot_mean_by_feature_class is None:
        plot_mean_by_feature_class = not _is_mean_by_feature_class_redundant(ts, series_by_class, series_by_fc)
    corr_series = _correlation_series(ts) if plot_correlation else []

    def _name(k: str) -> str:
        if label_name_mapping is not None and k in label_name_mapping:
            return label_name_mapping[k]
        lv2name = ts.get("label_value_to_class_name")
        if isinstance(lv2name, dict):
            if isinstance(lv2name.get(k), str):
                class_name = lv2name[k]
                if label_name_mapping is not None and class_name in label_name_mapping:
                    return label_name_mapping[class_name]
                return class_name
            try:
                k_float = float(k)
            except (TypeError, ValueError):
                k_float = None
            if k_float is not None and isinstance(lv2name.get(k_float), str):
                class_name = lv2name[k_float]
                if label_name_mapping is not None and class_name in label_name_mapping:
                    return label_name_mapping[class_name]
                return class_name
        return str(k)

    ax_idx = 0
    if plot_mean_by_class and series_by_class and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax.clear()
        for label_key, points in sorted(series_by_class.items(), key=lambda x: (x[0],)):
            pts = sorted(points)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=_name(label_key), marker=".", markersize=2)
        if best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
        ax.set_ylabel("encoded value")
        ax.set_xlabel("Step" if not (plot_mean_by_feature_class or plot_correlation) else "")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Mean by class")
        ax_idx += 1

    if plot_mean_by_feature_class and series_by_fc and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax.clear()
        for fc, points in sorted(series_by_fc.items(), key=lambda x: x[0]):
            pts = sorted(points)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=_name(fc), marker=".", markersize=2)
        if best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
        ax.set_ylabel("Mean encoded value")
        ax.set_xlabel("Step" if not plot_correlation else "")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Mean by feature class")
        ax_idx += 1

    if plot_correlation and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax.clear()
        if corr_series:
            xs, ys = zip(*corr_series)
            ax.plot(xs, ys, color="C0", marker=".", markersize=2, label="correlation")
        if best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
            if best_corr is not None:
                ax.scatter([best_step_val], [float(best_corr)], color="red", s=40, zorder=5, label="best")
        ax.set_ylabel("Correlation")
        ax.set_xlabel("Step")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Correlation")


def plot_training_convergence(
    trainer: Any = None,
    model_path: Optional[str] = None,
    training_stats: Optional[Dict[str, Any]] = None,
    *,
    plot_mean_by_class: bool = True,
    plot_mean_by_feature_class: Optional[bool] = None,
    plot_correlation: bool = True,
    best_step: bool = True,
    label_name_mapping: Optional[Dict[str, str]] = None,
    output: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show: bool = True,
    title: Union[str, bool] = True,
    figsize: Optional[Tuple[float, float]] = None,
    img_format: str = "pdf",
    dpi: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Plot training convergence: up to three subplots (mean_by_class, mean_by_feature_class, correlation).

    Data source: exactly one of trainer, model_path, or training_stats.
    - trainer: uses trainer.get_training_stats() (or in-memory stats if available).
    - model_path: uses load_training_stats(model_path).
    - training_stats: dict with keys training_stats, best_score_checkpoint (or raw training_stats dict).

    Three plot options, each in its own subplot when enabled:
    - plot_mean_by_class: mean encoded value per label over steps.
    - plot_mean_by_feature_class: mean encoded value per feature class over steps.
    - plot_correlation: correlation over steps. Best checkpoint step is marked in each subplot.

    Args:
        trainer: Trainer instance with get_training_stats(model_path) or similar.
        model_path: Path to saved model dir (training.json).
        training_stats: Pre-loaded run info or raw training_stats dict.
        plot_mean_by_class: Add a subplot for mean_by_class.
        plot_mean_by_feature_class: Add a subplot for mean_by_feature_class.
            None = auto (False when redundant with mean_by_class, True otherwise).
        plot_correlation: Add a subplot for correlation.
        best_step: Mark the best checkpoint step (vertical line + point on correlation).
        label_name_mapping: Optional display names for label values.
        output: Explicit output path for the plot file.
        experiment_dir: Used with resolve_output_path for default artifact path.
        show: Whether to call plt.show().
        title: True (default run_id), False, or custom string.
        figsize: (width, height) for figure.

    Returns:
        Path to saved plot file, or "" if nothing to plot or no path.
    """
    if training_stats is not None:
        if "training_stats" in training_stats and "best_score_checkpoint" in training_stats:
            run_info = training_stats
        else:
            run_info = {"training_stats": training_stats, "best_score_checkpoint": {}}
    elif model_path:
        run_info = load_training_stats(model_path)
        if run_info is None:
            logger.warning("No training.json at %s", model_path)
            return ""
    elif trainer is not None:
        get_stats = getattr(trainer, "get_training_stats", None)
        if get_stats is not None:
            run_info = get_stats()
        else:
            run_info = None
        if run_info is None:
            logger.warning("Could not get training stats from trainer")
            return ""
    else:
        raise ValueError("Provide one of trainer, model_path, or training_stats")

    plt = _require_matplotlib()

    ts = run_info.get("training_stats") or run_info
    if isinstance(ts, dict) and "training_stats" in ts:
        ts = ts["training_stats"]
    if not ts:
        logger.warning("No training_stats to plot")
        return ""

    bsc = run_info.get("best_score_checkpoint") or {}
    best_step_val = bsc.get("global_step")
    if best_step_val is not None:
        try:
            best_step_val = int(best_step_val)
        except (TypeError, ValueError):
            best_step_val = None

    steps, series_by_class, series_by_fc = _steps_and_values(
        ts, mean_by_class=plot_mean_by_class, mean_by_feature_class=plot_mean_by_feature_class is not False
    )
    if plot_mean_by_feature_class is None:
        plot_mean_by_feature_class = not _is_mean_by_feature_class_redundant(ts, series_by_class, series_by_fc)
    corr_series = _correlation_series(ts) if plot_correlation else []

    has_mbc = plot_mean_by_class and bool(series_by_class)
    has_mbfc = plot_mean_by_feature_class and bool(series_by_fc)
    has_corr = plot_correlation and bool(corr_series)
    if not (has_mbc or has_mbfc or has_corr):
        logger.warning("No plottable series in training_stats")
        return ""

    n_sub = (1 if has_mbc else 0) + (1 if has_mbfc else 0) + (1 if has_corr else 0)
    fig, axes = plt.subplots(n_sub, 1, sharex=True, figsize=figsize or (6, 1.5 * n_sub))
    if n_sub == 1:
        axes = [axes]
    ax_idx = 0

    def _name(k: str) -> str:
        if label_name_mapping is not None and k in label_name_mapping:
            return label_name_mapping[k]
        lv2name = ts.get("label_value_to_class_name")
        if isinstance(lv2name, dict):
            v = lv2name.get(k)
            if isinstance(v, str):
                if label_name_mapping is not None and v in label_name_mapping:
                    return label_name_mapping[v]
                return v
            try:
                k_float = float(k)
            except (TypeError, ValueError):
                k_float = None
            if k_float is not None:
                v = lv2name.get(k_float)
                if isinstance(v, str):
                    if label_name_mapping is not None and v in label_name_mapping:
                        return label_name_mapping[v]
                    return v
        return str(k)

    if has_mbc:
        ax = axes[ax_idx]
        ax_idx += 1
        for label_key, points in sorted(series_by_class.items(), key=lambda x: (x[0],)):
            pts = sorted(points)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=_name(label_key), marker=".", markersize=2)
        if best_step and best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
        ax.set_ylabel("Mean encoded value")
        ax.set_xlabel("Step" if not (has_mbfc or has_corr) else "")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Mean by class")

    if has_mbfc:
        ax = axes[ax_idx]
        ax_idx += 1
        for fc, points in sorted(series_by_fc.items(), key=lambda x: x[0]):
            pts = sorted(points)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=_name(fc), marker=".", markersize=2)
        if best_step and best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
        ax.set_ylabel("Mean encoded value")
        ax.set_xlabel("Step" if not has_corr else "")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Mean by feature class")

    if has_corr:
        ax = axes[ax_idx]
        xs, ys = zip(*corr_series)
        ax.plot(xs, ys, color="C0", marker=".", markersize=2, label="correlation")
        if best_step and best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
            best_corr_val = bsc.get("correlation")
            if best_corr_val is not None:
                ax.scatter([best_step_val], [float(best_corr_val)], color="red", s=40, zorder=5, label="best")
        ax.set_ylabel("Correlation")
        ax.set_xlabel("Step")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Correlation")

    if title is not False:
        if isinstance(title, str):
            fig.suptitle(title, fontsize=10)
        else:
            rid = getattr(trainer, "run_id", None) if trainer is not None else None
            fig.suptitle(rid or "Training convergence", fontsize=10)
    plt.tight_layout()

    out_path = None
    if output:
        out_path = output
    else:
        exp_dir = experiment_dir or (getattr(trainer, "experiment_dir", None) if trainer is not None else None)
        out_path = resolve_output_path(exp_dir, None, ARTIFACT_CONVERGENCE_PLOT)
    if out_path and img_format:
        ext = img_format if img_format.startswith(".") else f".{img_format}"
        out_path = os.path.splitext(out_path)[0] + ext

    if out_path is None and not show:
        raise ValueError(
            "output is required when experiment_dir is not set and not show=True."
        )

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        plt.savefig(out_path, **save_kwargs)
        logger.info("Saved convergence plot: %s", out_path)
    if show:
        plt.show()
    plt.close()
    return out_path or ""
