"""
Training convergence plot: mean_by_class and mean_by_feature_class over steps,
with correlation in a separate subplot. Best checkpoint step is marked.

Requires matplotlib. If missing, raises ImportError with install instructions.
"""

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Set

import numpy as np

from gradiend.util.paths import resolve_output_path, ARTIFACT_CONVERGENCE_PLOT
from gradiend.visualizer.labels import resolve_highlight_non_convergence, resolve_plot_title_with_convergence
from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.trainer.core.stats import load_training_stats
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

ClassSpreadMode = Optional[Literal["minmax", "iqr", "ci95"]]


def _validate_class_spread(class_spread: Any) -> ClassSpreadMode:
    if class_spread is None or class_spread in {"minmax", "iqr", "ci95"}:
        return class_spread
    raise ValueError("class_spread must be one of None, 'minmax', 'iqr', or 'ci95'")


def _class_spread_keys(mode: Literal["minmax", "iqr"]) -> Tuple[str, str]:
    if mode == "iqr":
        return "q1_by_class", "q3_by_class"
    return "min_by_class", "max_by_class"


def _feature_class_spread_keys(mode: Literal["minmax", "iqr"]) -> Tuple[str, str]:
    if mode == "iqr":
        return "q1_by_feature_class", "q3_by_feature_class"
    return "min_by_feature_class", "max_by_feature_class"


def _class_spread_title_suffix(mode: Literal["minmax", "iqr", "ci95"]) -> str:
    if mode == "iqr":
        return " (shaded: IQR)"
    if mode == "ci95":
        return " (shaded: 95% CI)"
    return " (shaded: min-max)"


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


def _range_series(
    training_stats: Dict[str, Any],
    min_key: str,
    max_key: str,
) -> Dict[str, List[Tuple[int, float, float]]]:
    """Build step -> (min, max) series keyed by label/feature class."""
    mins = training_stats.get(min_key) or {}
    maxs = training_stats.get(max_key) or {}
    mins = _normalize_step_keys(mins) if isinstance(mins, dict) else {}
    maxs = _normalize_step_keys(maxs) if isinstance(maxs, dict) else {}
    if not mins or not maxs:
        return {}

    series: Dict[str, List[Tuple[int, float, float]]] = {}
    for step, label_mins in mins.items():
        label_maxs = maxs.get(step)
        if not isinstance(label_mins, dict) or not isinstance(label_maxs, dict):
            continue
        for label, min_val in label_mins.items():
            max_val = label_maxs.get(label)
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                key = str(label)
                if key not in series:
                    series[key] = []
                series[key].append((step, float(min_val), float(max_val)))
    return series


def _confidence_interval_series(
    training_stats: Dict[str, Any],
    mean_key: str,
    std_key: str,
    n_key: str,
    *,
    z: float = 1.96,
) -> Dict[str, List[Tuple[int, float, float]]]:
    """Build step -> mean +/- z * standard error series keyed by label/feature class."""
    means = training_stats.get(mean_key) or {}
    stds = training_stats.get(std_key) or {}
    ns = training_stats.get(n_key) or {}
    means = _normalize_step_keys(means) if isinstance(means, dict) else {}
    stds = _normalize_step_keys(stds) if isinstance(stds, dict) else {}
    ns = _normalize_step_keys(ns) if isinstance(ns, dict) else {}
    if not means or not stds or not ns:
        return {}

    series: Dict[str, List[Tuple[int, float, float]]] = {}
    for step, label_means in means.items():
        label_stds = stds.get(step)
        label_ns = ns.get(step)
        if not isinstance(label_means, dict) or not isinstance(label_stds, dict) or not isinstance(label_ns, dict):
            continue
        for label, mean_val in label_means.items():
            std_val = label_stds.get(label)
            n_val = label_ns.get(label)
            if std_val is None and not isinstance(label, str):
                std_val = label_stds.get(str(label))
            if n_val is None and not isinstance(label, str):
                n_val = label_ns.get(str(label))
            if not isinstance(mean_val, (int, float)) or not isinstance(std_val, (int, float)):
                continue
            try:
                n = int(n_val)
            except (TypeError, ValueError):
                continue
            if n <= 0:
                continue
            half_width = 0.0 if n <= 1 else float(z) * float(std_val) / float(np.sqrt(n))
            key = str(label)
            if key not in series:
                series[key] = []
            mean = float(mean_val)
            series[key].append((step, mean - half_width, mean + half_width))
    return series


def _plot_mean_series_with_range(
    ax: Any,
    series: Dict[str, List[Tuple[int, float]]],
    range_series: Optional[Dict[str, List[Tuple[int, float, float]]]],
    *,
    show_range: bool,
    name_fn: Any,
) -> None:
    """Plot mean lines per key, optionally shading min-max range per step."""
    for label_key, points in sorted(series.items(), key=lambda x: (x[0],)):
        pts = sorted(points)
        if not pts:
            continue
        xs, ys = zip(*pts)
        (line,) = ax.plot(xs, ys, label=name_fn(label_key), marker=".", markersize=2)
        if show_range and range_series:
            range_pts = sorted(range_series.get(label_key, []))
            if range_pts:
                range_by_step = {step: (lo, hi) for step, lo, hi in range_pts}
                ymins = []
                ymaxs = []
                for step in xs:
                    bounds = range_by_step.get(step)
                    if bounds is None:
                        ymins.append(np.nan)
                        ymaxs.append(np.nan)
                    else:
                        ymins.append(bounds[0])
                        ymaxs.append(bounds[1])
                ax.fill_between(
                    xs,
                    ymins,
                    ymaxs,
                    color=line.get_color(),
                    alpha=0.2,
                    linewidth=0,
                )


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
    class_spread: ClassSpreadMode = None,
    legend_ncol: Optional[int] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Optional[str] = None,
) -> None:
    """
    Draw convergence curves into existing axes (live or static).

    Args:
        training_stats: Training statistics dictionary containing convergence series.
        axes: Matplotlib axes to draw into. ``axes[0]`` receives ``mean_by_class`` when
            enabled, the next axis receives ``mean_by_feature_class`` when enabled, and
            the next axis receives correlation when enabled.
        best_step_val: Optional checkpoint step to mark with a vertical line.
        best_corr: Optional best correlation value to mark on the correlation axis.
        label_name_mapping: Optional mapping from raw label ids/names to display names.
        plot_mean_by_class: Whether to draw the mean-by-class subplot.
        plot_mean_by_feature_class: Whether to draw the mean-by-feature-class subplot.
            ``None`` auto-disables it when it would duplicate ``mean_by_class``.
        plot_correlation: Whether to draw the correlation subplot.
        class_spread: Optional spread to shade behind each mean line.
            ``"minmax"`` shades the min-max encoded value range per class
            (and per feature class), ``"iqr"`` shades the interquartile range (Q1-Q3),
            and ``"ci95"`` shades mean +/- 1.96 standard errors.
            Requires the corresponding keys in training stats (recorded from new runs).
        legend_ncol: Number of columns for the external legend when many series are shown.
        legend_bbox_to_anchor: Matplotlib legend anchor used for the external legend.
        legend_loc: Matplotlib legend location used for legends.
    """
    ts = training_stats
    steps, series_by_class, series_by_fc = _steps_and_values(
        ts, mean_by_class=plot_mean_by_class, mean_by_feature_class=plot_mean_by_feature_class is not False
    )
    if plot_mean_by_feature_class is None:
        plot_mean_by_feature_class = not _is_mean_by_feature_class_redundant(ts, series_by_class, series_by_fc)
    corr_series = _correlation_series(ts) if plot_correlation else []
    spread_mode = _validate_class_spread(class_spread)
    if spread_mode == "ci95":
        range_by_class = _confidence_interval_series(ts, "mean_by_class", "std_by_class", "n_by_class")
        range_by_fc = _confidence_interval_series(
            ts,
            "mean_by_feature_class",
            "std_by_feature_class",
            "n_by_feature_class",
        )
    else:
        range_by_class = (
            _range_series(ts, *_class_spread_keys(spread_mode)) if spread_mode else {}
        )
        range_by_fc = (
            _range_series(ts, *_feature_class_spread_keys(spread_mode)) if spread_mode else {}
        )

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
    n_series = len(series_by_class) if series_by_class else 0
    n_fc = len(series_by_fc) if series_by_fc else 0
    use_external_legend = max(n_series, n_fc) >= 6
    legend_fontsize = 6 if use_external_legend else 8
    leg_loc = legend_loc if legend_loc is not None else ("center left" if use_external_legend else "best")
    leg_bbox = legend_bbox_to_anchor if legend_bbox_to_anchor is not None else ((1.02, 1.0) if use_external_legend else None)

    if plot_mean_by_class and series_by_class and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax.clear()
        _plot_mean_series_with_range(
            ax,
            series_by_class,
            range_by_class,
            show_range=spread_mode is not None and bool(range_by_class),
            name_fn=_name,
        )
        if best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
        ax.set_ylabel("encoded value")
        ax.set_xlabel("Step" if not (plot_mean_by_feature_class or plot_correlation) else "")
        if not use_external_legend:
            leg_kw = {"loc": leg_loc, "fontsize": legend_fontsize}
            if leg_bbox is not None:
                leg_kw["bbox_to_anchor"] = leg_bbox
            ax.legend(**leg_kw)
        ax.grid(True, alpha=0.3)
        title = "Mean by class"
        if spread_mode and range_by_class:
            title += _class_spread_title_suffix(spread_mode)
        ax.set_title(title)
        ax_idx += 1

    legend_fontsize_fc = 6 if use_external_legend else 8
    leg_loc_fc = legend_loc if legend_loc is not None else ("center left" if use_external_legend else "best")
    leg_bbox_fc = legend_bbox_to_anchor if legend_bbox_to_anchor is not None else ((1.02, 1.0) if use_external_legend else None)

    if plot_mean_by_feature_class and series_by_fc and ax_idx < len(axes):
        ax = axes[ax_idx]
        ax.clear()
        _plot_mean_series_with_range(
            ax,
            series_by_fc,
            range_by_fc,
            show_range=spread_mode is not None and bool(range_by_fc),
            name_fn=_name,
        )
        if best_step_val is not None:
            ax.axvline(x=best_step_val, color="gray", linestyle="--", alpha=0.8, label="best step")
        ax.set_ylabel("Mean encoded value")
        ax.set_xlabel("Step" if not plot_correlation else "")
        if not use_external_legend:
            leg_kw_fc = {"loc": leg_loc_fc, "fontsize": legend_fontsize_fc}
            if leg_bbox_fc is not None:
                leg_kw_fc["bbox_to_anchor"] = leg_bbox_fc
            ax.legend(**leg_kw_fc)
        ax.grid(True, alpha=0.3)
        title = "Mean by feature class"
        if spread_mode and range_by_fc:
            title += _class_spread_title_suffix(spread_mode)
        ax.set_title(title)
        ax_idx += 1

    if use_external_legend and axes:
        legend_axes = []
        i = 0
        if plot_mean_by_class and series_by_class:
            legend_axes.append(axes[i])
            i += 1
        if plot_mean_by_feature_class and series_by_fc:
            legend_axes.append(axes[i])
        all_handles, all_labels = [], []
        for ax in legend_axes:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li == "best step" and "best step" in all_labels:
                    continue
                all_handles.append(hi)
                all_labels.append(li)
        fig = axes[0].figure
        fig.legend(
            all_handles,
            all_labels,
            loc=legend_loc or "center left",
            bbox_to_anchor=legend_bbox_to_anchor or (1.02, 0.5),
            ncol=legend_ncol if legend_ncol is not None else 1,
            fontsize=6,
        )

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
    class_spread: ClassSpreadMode = None,
    best_step: bool = True,
    label_name_mapping: Optional[Dict[str, str]] = None,
    output: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show: bool = True,
    title: Union[str, bool] = True,
    figsize: Optional[Tuple[float, float]] = None,
    img_format: str = "png",
    dpi: Optional[int] = None,
    legend_ncol: Optional[int] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Optional[str] = None,
    highlight_non_convergence: Optional[bool] = None,
    return_fig_ax: bool = False,
    **kwargs: Any,
) -> Any:
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
    - class_spread: shade encoded value spread per class behind each mean line

      (``"minmax"`` = min-max, ``"iqr"`` = Q1-Q3, ``"ci95"`` = 95% confidence interval).

    Args:
        trainer: Trainer instance with get_training_stats(model_path) or similar.
        model_path: Path to saved model dir (training.json).
        training_stats: Pre-loaded run info or raw training_stats dict.
        plot_mean_by_class: Add a subplot for mean_by_class.
        plot_mean_by_feature_class: Add a subplot for mean_by_feature_class.
            None = auto (False when redundant with mean_by_class, True otherwise).
        plot_correlation: Add a subplot for correlation.
        class_spread: Shade spread per class (and feature class) behind mean lines.
            ``"minmax"``: min-max band. ``"iqr"``: interquartile range (Q1-Q3).
            ``"ci95"``: 95% confidence interval around the mean (mean +/- 1.96 SE).
            ``None`` disables spread shading.
            Requires matching keys in training stats (recorded from new runs).
        best_step: Mark the best checkpoint step (vertical line + point on correlation).
        label_name_mapping: Optional display names for label values.
        output: Explicit output path for the plot file.
        experiment_dir: Used with resolve_output_path for default artifact path.
        show: Whether to call plt.show().
        title: True (default run_id), False, or custom string.
        figsize: (width, height) for figure.
        img_format: File extension used when resolving the default output path.
        dpi: Optional Matplotlib savefig DPI.
        legend_ncol: Number of columns for the external legend when there are >= 6 series (default 1).
        legend_bbox_to_anchor: (x, y) for the external legend when >= 6 series (default (1.02, 0.5)).
        legend_loc: Matplotlib loc for the external legend when >= 6 series (default "center left").
        highlight_non_convergence: When True, append a non-convergence marker to the title for
            non-converged runs. ``None`` uses ``TrainingArguments.highlight_non_convergence``.
        return_fig_ax: If True, return ``(fig, axes)`` and leave the figure open so callers can
            customize it before showing, saving again, or closing it. Existing saving/display
            behavior still runs when ``output``/``experiment_dir`` or ``show`` are set.
        **kwargs: Reserved for compatibility with trainer visualizer wrappers.

    Returns:
        Path to saved plot file, or "" if nothing to plot or no path. If ``return_fig_ax=True``,
        returns ``(fig, axes)``.
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
    n_legend_entries = max(len(series_by_class) if series_by_class else 0, len(series_by_fc) if series_by_fc else 0)
    # When many legend entries (e.g. identity transitions), use larger height so subplots are not squashed by legend
    height_per_sub = 2.0 if n_legend_entries >= 6 else 1.5
    fig, axes = plt.subplots(n_sub, 1, sharex=True, figsize=figsize or (6, height_per_sub * n_sub))
    if n_sub == 1:
        axes = [axes]
    # Leave space for figure-level legend when >= 6 series
    if n_legend_entries >= 6:
        fig.subplots_adjust(right=0.72)

    best_corr_val = bsc.get("correlation")
    if best_corr_val is not None:
        try:
            best_corr_val = float(best_corr_val)
        except (TypeError, ValueError):
            best_corr_val = None

    draw_convergence_axes(
        ts,
        axes,
        best_step_val=best_step_val if best_step else None,
        best_corr=best_corr_val if best_step else None,
        label_name_mapping=label_name_mapping,
        plot_mean_by_class=has_mbc,
        plot_mean_by_feature_class=has_mbfc,
        plot_correlation=has_corr,
        class_spread=class_spread,
        legend_ncol=legend_ncol,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_loc=legend_loc,
    )

    highlight = resolve_highlight_non_convergence(highlight_non_convergence, trainer=trainer)
    resolved_title = resolve_plot_title_with_convergence(
        title,
        trainer=trainer,
        run_info=run_info,
        highlight_non_convergence=highlight,
    )
    if resolved_title is not False:
        fig.suptitle(str(resolved_title), fontsize=10)
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

    if out_path is None and not show and not return_fig_ax:
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
    if return_fig_ax:
        return fig, axes
    plt.close(fig)
    return out_path or ""
