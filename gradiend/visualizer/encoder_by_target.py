"""
Encoder plots with target tokens on the x-axis, grouped by feature class, hue = data split.
"""

from __future__ import annotations

import os
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gradiend.util.encoder_splits import order_split_names
from gradiend.util.logging import get_logger
from gradiend.util.paths import ARTIFACT_ENCODER_PLOT, resolve_output_path
from gradiend.visualizer.labels import (
    converged_for_trainer,
    format_label_with_convergence,
    resolve_highlight_non_convergence,
)
from gradiend.visualizer.plot_optional import _require_matplotlib, _require_seaborn

logger = get_logger(__name__)

PlotStyle = Literal["strip", "box", "violin"]
SUPPORTED_PLOT_STYLES = frozenset(("strip", "box", "violin"))
_TARGET_GROUP_PAD = 0.45


def _encoded_scalar(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "__len__") and len(value) > 0:
        return float(value[0])
    return None


def build_encoder_target_plot_frame(
    encoder_df: pd.DataFrame,
    *,
    target_col: str = "factual_token",
    class_col: str = "source_id",
    hue_col: str = "data_split",
    class_order: Optional[Sequence[str]] = None,
    id2label: Optional[dict] = None,
    types: Sequence[str] = ("training",),
) -> Tuple[pd.DataFrame, List[str], List[str], dict, dict]:
    """Prepare training rows for target-token x-axis plots.

    Args:
        encoder_df: Encoder analysis DataFrame with at least ``type`` and ``encoded`` columns.
        target_col: Column containing the masked target token shown on the x-axis.
        class_col: Column used to group target tokens by feature class.
        hue_col: Column used as plot hue, typically ``data_split``.
        class_order: Optional feature-class display order. Values are mapped through
            ``id2label`` when available.
        id2label: Optional mapping from numeric/source ids to display labels.
        types: Encoder row types to include, by default training rows only.

    Returns:
        plot_df, target_order, split_hue_order, target_to_class_display, target_to_split
    """
    plot_df = encoder_df[encoder_df["type"].isin(types)].copy()
    if plot_df.empty:
        return plot_df, [], [], {}, {}

    if hue_col not in plot_df.columns:
        plot_df[hue_col] = "test"
    plot_df[hue_col] = plot_df[hue_col].astype(str)

    def _display(val: Any) -> str:
        if id2label and val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                int_v = int(val)
                return str(id2label.get(int_v, id2label.get(str(int_v), id2label.get(val, val))))
            except (ValueError, TypeError):
                return str(id2label.get(val, id2label.get(str(val), val)))
        return str(val)

    plot_df["feature_class"] = plot_df[class_col].map(_display).astype(str)
    if target_col in plot_df.columns:
        plot_df["target_token"] = plot_df[target_col].astype(str).str.strip()
    else:
        plot_df["target_token"] = plot_df[class_col].map(_display).astype(str)

    plot_df = plot_df[plot_df["target_token"].astype(str).str.len() > 0].copy()
    plot_df["encoded"] = plot_df["encoded"].map(_encoded_scalar)
    plot_df = plot_df.dropna(subset=["encoded"])

    if class_order is None:
        class_order = sorted(plot_df["feature_class"].dropna().unique().tolist())
    else:
        class_order = [_display(c) for c in class_order]

    split_hue_order = order_split_names(plot_df[hue_col].dropna().unique().tolist())
    split_rank = {name: rank for rank, name in enumerate(split_hue_order)}
    plot_df["plot_hue"] = plot_df[hue_col]

    target_order: List[str] = []
    target_to_class: dict = {}
    target_to_split: dict = {}
    for cls in class_order:
        sub = plot_df[plot_df["feature_class"] == cls]
        token_splits = (
            sub.groupby("target_token", sort=False)[hue_col]
            .agg(
                lambda values: order_split_names(
                    values.dropna().astype(str).unique().tolist()
                )[0]
            )
        )
        tokens = sorted(
            sub["target_token"].dropna().unique().tolist(),
            key=lambda tok: (
                split_rank.get(str(token_splits.get(tok, "")), len(split_rank)),
                str(tok).casefold(),
            ),
        )
        for tok in tokens:
            if tok not in target_order:
                target_order.append(tok)
                target_to_class[tok] = cls
                target_to_split[tok] = str(token_splits.get(tok, ""))

    return plot_df, target_order, split_hue_order, target_to_class, target_to_split


def _add_feature_class_group_brackets(
    ax: Any,
    *,
    target_order: Sequence[str],
    target_to_class: dict,
    class_order: Sequence[str],
) -> None:
    if not target_order or not target_to_class:
        return
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_line = 1.02
    for cls in class_order:
        idxs = [i for i, tok in enumerate(target_order) if target_to_class.get(tok) == cls]
        if not idxs:
            continue
        x0, x1 = min(idxs) - _TARGET_GROUP_PAD, max(idxs) + _TARGET_GROUP_PAD
        ax.axvspan(x0, x1, color="0.99", zorder=0)
        mid = 0.5 * (x0 + x1)
        ax.plot(
            [x0, x0, x1, x1],
            [y_line, y_line + 0.035, y_line + 0.035, y_line],
            transform=trans,
            color="0.35",
            clip_on=False,
        )
        ax.text(
            mid,
            y_line + 0.045,
            str(cls),
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.25",
        )


def _split_color_map(split_hue_order: Sequence[str]) -> dict:
    sns = _require_seaborn()
    palette = sns.color_palette(n_colors=max(len(split_hue_order), 1))
    return {split: palette[idx % len(palette)] for idx, split in enumerate(split_hue_order)}


def _color_from_legend_handle(handle: Any) -> Tuple[float, float, float]:
    import matplotlib.colors as mcolors

    if hasattr(handle, "get_facecolor"):
        fc = np.asarray(handle.get_facecolor())
        rgb = fc[0][:3] if fc.ndim > 1 else fc[:3]
        return mcolors.to_rgb(rgb)
    return (0.5, 0.5, 0.5)


def _legend_split_colors(handles: Sequence[Any], labels: Sequence[str]) -> dict:
    return {str(label): _color_from_legend_handle(handle) for handle, label in zip(handles, labels)}


def _draw_split_spine_marks(
    ax: Any,
    *,
    target_order: Sequence[str],
    target_to_split: dict,
    target_to_class: dict,
    class_order: Sequence[str],
    split_color_map: dict,
    linewidth: float = 1.5,
) -> None:
    """Draw one colored segment on the x-axis spine per split run within each feature class."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    default_color = (0.35, 0.35, 0.35)
    pad = _TARGET_GROUP_PAD
    for cls in class_order:
        cls_idxs = [i for i, tok in enumerate(target_order) if target_to_class.get(tok) == cls]
        if not cls_idxs:
            continue
        runs: List[Tuple[str, List[int]]] = []
        for idx in cls_idxs:
            tok = target_order[idx]
            split = str(target_to_split.get(tok, ""))
            if not runs or runs[-1][0] != split:
                runs.append((split, [idx]))
            else:
                runs[-1][1].append(idx)
        for split, idxs in runs:
            color = split_color_map.get(split, default_color)
            x0 = min(idxs) - pad
            x1 = max(idxs) + pad
            ax.plot(
                [x0, x1],
                [0.0, 0.0],
                transform=trans,
                color=color,
                linewidth=linewidth,
                solid_capstyle="butt",
                clip_on=False,
                zorder=10,
            )


def _tighten_target_axis_xlim(ax: Any, n_targets: int, *, pad: float = _TARGET_GROUP_PAD) -> None:
    """Remove default categorical margins so the plot area hugs the y-axes."""
    if n_targets < 1:
        return
    ax.margins(x=0)
    ax.set_xlim(-pad, (n_targets - 1) + pad)


def _wrap_hover_text(text: Any, line_chars: int = 90) -> str:
    if text is None:
        return ""
    words = str(text).split()
    if not words:
        return str(text)
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > line_chars:
            lines.append(current)
            current = word
        else:
            current = word if not current else f"{current} {word}"
    if current:
        lines.append(current)
    return "<br>".join(lines)


def _text_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("masked", "text", "sentence"):
        if col in df.columns:
            return col
    return None


def _plot_encoder_by_target_interactive(
    plot_df: pd.DataFrame,
    *,
    target_order: Sequence[str],
    split_hue_order: Sequence[str],
    title: Optional[str],
    output: Optional[str],
    show: bool,
    height: int,
) -> Any:
    try:
        import plotly.express as px
    except ImportError:
        logger.warning("plotly not installed; install with pip install plotly")
        return None

    df = plot_df.copy()
    text_col = _text_column(df)
    if text_col is not None:
        df["text_hover"] = df[text_col].map(_wrap_hover_text)
    hover_data = [
        col for col in [
            "text_hover",
            "target_token",
            "feature_class",
            "plot_hue",
            "encoded",
            "data_split",
            "type",
        ]
        if col in df.columns
    ]
    fig = px.strip(
        df,
        x="target_token",
        y="encoded",
        color="plot_hue",
        category_orders={
            "target_token": list(target_order),
            "plot_hue": list(split_hue_order),
        },
        hover_data=hover_data or None,
        title=title,
        height=height,
    )
    fig.update_traces(marker={"opacity": 0.78})
    fig.update_layout(
        xaxis_title="Target",
        yaxis_title="Encoded value",
        legend_title_text="Split",
        hovermode="closest",
    )
    if output:
        html_output = output
        base, ext = os.path.splitext(html_output)
        if ext.lower() not in {".html", ".htm"}:
            html_output = f"{base}.html"
        os.makedirs(os.path.dirname(html_output) or ".", exist_ok=True)
        fig.write_html(html_output)
        logger.info("Saved interactive encoder by-target plot: %s", html_output)
    if show:
        fig.show()
    return fig


def plot_encoder_by_target(
    trainer: Any = None,
    encoder_df: Optional[pd.DataFrame] = None,
    *,
    plot_style: PlotStyle = "strip",
    target_col: str = "factual_token",
    class_col: str = "source_id",
    hue_col: str = "data_split",
    class_order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    output: Optional[str] = None,
    output_dir: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    jitter: float = 0.25,
    dodge: bool = True,
    point_size: float = 1.5,
    interactive: bool = False,
    height: int = 520,
    legend_loc: str = "upper right",
    highlight_non_convergence: Optional[bool] = None,
    **kwargs: Any,
) -> Optional[str]:
    """
    Encoder distribution by masked target token (x), grouped by feature class, colored by split.

    Args:
        trainer: Optional trainer used to compute ``encoder_df`` and resolve labels/convergence.
        encoder_df: Optional precomputed encoder analysis DataFrame.
        plot_style: ``strip`` (default), ``box``, or ``violin``.
        target_col: Column containing the masked target token shown on the x-axis.
        class_col: Column used to group target tokens by feature class.
        hue_col: Column used as plot hue, typically ``data_split``.
        class_order: Optional feature-class display order.
        interactive: If True, create a Plotly strip plot with hover instead of a static Matplotlib plot.
        legend_loc: Matplotlib legend location for static plots.
        point_size: Marker size for strip plot points (seaborn ``size``).
        title: Optional plot title. None (default) draws no title.
        output: Explicit output path. Interactive plots are written as HTML.
        output_dir: Directory used when resolving the default output filename.
        experiment_dir: Experiment directory used when resolving the default output filename.
        show: Whether to display the plot.
        figsize: Static figure size in inches.
        jitter: Jitter width for static strip plots.
        dodge: Whether to dodge static strip points by hue.
        height: Plotly figure height for interactive plots.
        highlight_non_convergence: When True, append a non-convergence marker to the title
            for non-converged runs. ``None`` uses trainer settings when available.
        **kwargs: Forwarded to ``trainer.analyze_encoder`` when ``encoder_df`` is not supplied.
    """
    removed_label_kwargs = {
        "label_points",
        "label_indices",
        "label_col",
        "label_max_chars",
        "label_formatter",
        "label_sample_per_group",
        "adjust_labels",
        "label_fontsize",
        "outlier_method",
        "outlier_k",
    }
    passed_removed = sorted(removed_label_kwargs & set(kwargs))
    if passed_removed:
        raise TypeError(
            "plot_encoder_by_target does not accept point-label arguments: "
            + ", ".join(passed_removed)
        )
    if encoder_df is None:
        if trainer is None:
            raise ValueError("Provide encoder_df or trainer")
        encoder_df = trainer.analyze_encoder(getattr(trainer, "get_model", lambda: None)(), **kwargs)
    if encoder_df is None or encoder_df.empty:
        logger.warning("No encoder data for by-target plot")
        return None

    id2label = {}
    if trainer is not None:
        id2label = dict(getattr(trainer, "_id2label", None) or {})
        config_obj = getattr(trainer, "config", None)
        config_map = getattr(config_obj, "id2label", None) if config_obj is not None else None
        if isinstance(config_map, dict):
            id2label.update(config_map)
    if class_order is None and trainer is not None:
        pair = getattr(trainer, "pair", None)
        if pair:
            class_order = list(pair)

    plot_df, target_order, split_hue_order, target_to_class, target_to_split = (
        build_encoder_target_plot_frame(
        encoder_df,
        target_col=target_col,
        class_col=class_col,
        hue_col=hue_col,
        class_order=class_order,
        id2label=id2label or None,
    )
    )
    if plot_df.empty or not target_order:
        logger.warning("No plottable target tokens for by-target encoder plot")
        return None

    highlight = resolve_highlight_non_convergence(highlight_non_convergence, trainer=trainer)
    plot_title: Optional[str] = None
    if title is not None:
        plot_title = format_label_with_convergence(
            title,
            converged=converged_for_trainer(trainer),
            highlight_non_convergence=highlight,
        )

    class_order_resolved = class_order or sorted({target_to_class[t] for t in target_order})
    plot_df["x_group"] = plot_df["target_token"].astype(str)

    style = str(plot_style).lower()
    if style not in SUPPORTED_PLOT_STYLES:
        raise ValueError(
            f"plot_style must be one of {sorted(SUPPORTED_PLOT_STYLES)}; got {plot_style!r}"
        )

    if interactive:
        return _plot_encoder_by_target_interactive(
            plot_df,
            target_order=target_order,
            split_hue_order=split_hue_order,
            title=plot_title,
            output=output,
            show=show,
            height=height,
        )

    plt = _require_matplotlib()
    sns = _require_seaborn()
    width = max(4.0, 0.14 * len(target_order))
    _figsize = figsize if figsize is not None else (width, 2.7)
    fig, ax = plt.subplots(figsize=_figsize)
    split_colors = _split_color_map(split_hue_order)
    palette = [split_colors[split] for split in split_hue_order]

    common = dict(
        data=plot_df,
        x="target_token",
        y="encoded",
        hue="plot_hue",
        hue_order=split_hue_order,
        order=target_order,
        palette=palette,
        ax=ax,
    )
    splits_per_target = plot_df.groupby("target_token")["plot_hue"].nunique(dropna=True)
    center_single_split_targets = bool(not splits_per_target.empty and splits_per_target.max() <= 1)
    effective_dodge = bool(dodge and not center_single_split_targets)
    if style == "box":
        sns.boxplot(**common, dodge=effective_dodge, fliersize=2)
    elif style == "violin":
        sns.violinplot(**common, dodge=effective_dodge, inner="box", cut=0)
    elif style == "strip":
        sns.stripplot(**common, dodge=effective_dodge, jitter=jitter, size=point_size, alpha=0.85)

    ax.set_ylabel("Encoded value")
    ax.set_xlabel("Target")
    if plot_title:
        ax.set_title(plot_title, pad=30)
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    handles, labels = ax.get_legend_handles_labels()
    mark_colors = (
        _legend_split_colors(handles, labels)
        if handles and labels
        else split_colors
    )
    _draw_split_spine_marks(
        ax,
        target_order=target_order,
        target_to_split=target_to_split,
        target_to_class=target_to_class,
        class_order=class_order_resolved,
        split_color_map=mark_colors,
    )
    if handles and labels:
        for handle in handles:
            if hasattr(handle, "set_edgecolor"):
                handle.set_edgecolor("none")
        ax.legend(handles, labels, title="Split", loc=legend_loc, fontsize=8)
    _add_feature_class_group_brackets(
        ax,
        target_order=target_order,
        target_to_class=target_to_class,
        class_order=class_order_resolved,
    )
    _tighten_target_axis_xlim(ax, len(target_order))

    fig.tight_layout(rect=(0, 0, 1, 0.88 if plot_title else 0.92))

    out = output
    if out is None:
        run_id = getattr(trainer, "run_id", None) if trainer is not None else None
        out = resolve_output_path(
            experiment_dir or (getattr(trainer, "experiment_dir", None) if trainer is not None else None),
            output_dir,
            ARTIFACT_ENCODER_PLOT,
            run_id=run_id,
        )
        if out is not None:
            base, _ = os.path.splitext(out)
            out = f"{base}_by_target_{style}.png"
        elif output_dir:
            out = os.path.join(output_dir, f"encoder_by_target_{style}.png")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        logger.info("Saved encoder by-target plot: %s", out)
    if show:
        plt.show()
    plt.close(fig)
    return out
