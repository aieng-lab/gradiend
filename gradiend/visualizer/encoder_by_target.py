"""
Encoder plots with target tokens on the x-axis, grouped by feature class, hue = data split.
"""

from __future__ import annotations

import os
import hashlib
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gradiend.util.encoder_splits import order_split_names
from gradiend.util.logging import get_logger
from gradiend.util.paths import ARTIFACT_ENCODER_PLOT, resolve_output_path
from gradiend.visualizer.labels import (
    converged_for_trainer,
    format_label_with_convergence,
    format_plotly_label,
    plotly_labels_for,
    resolve_highlight_non_convergence,
)
from gradiend.visualizer.plot_optional import _require_matplotlib, _require_seaborn

logger = get_logger(__name__)

PlotStyle = Literal["strip", "box", "violin"]
SUPPORTED_PLOT_STYLES = frozenset(("strip", "box", "violin"))
_TARGET_GROUP_PAD = 0.45
_TARGET_GROUP_INNER_PAD = 0.2
TEXT_HOVER_COLUMNS = ("display_text", "text_:hover", "text_hover", "template", "masked", "input_text", "text", "sentence")


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
    target_col: str = "source_token",
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

    def _has_value(row: pd.Series, col: str) -> bool:
        return col in row.index and pd.notna(row[col]) and str(row[col]).strip() != ""

    def _display_class_value(row: pd.Series) -> Any:
        if (
            class_col == "source_id"
            and str(row.get("input_type", "")).lower() == "alternative"
        ):
            for candidate in ("target_id", "counterfactual_id", "alternative_id"):
                if _has_value(row, candidate):
                    return row[candidate]
        return row[class_col] if class_col in row.index else row.get("source_id")

    def _target_token_value(row: pd.Series) -> Any:
        if target_col != "source_token":
            return row[target_col] if target_col in row.index else None
        if _has_value(row, "source_token"):
            return row["source_token"]
        if str(row.get("input_type", "")).lower() == "alternative":
            for candidate in ("alternative_token", "counterfactual_token"):
                if _has_value(row, candidate):
                    return row[candidate]
        return row["factual_token"] if "factual_token" in row.index else None

    plot_df["feature_class"] = plot_df.apply(_display_class_value, axis=1).map(_display).astype(str)
    plot_df["target_token"] = plot_df.apply(_target_token_value, axis=1).astype(str).str.strip()
    if plot_df["target_token"].isna().all():
        plot_df["target_token"] = plot_df[class_col].map(_display).astype(str)
    else:
        missing_token = plot_df["target_token"].isin(["", "None", "nan"])
        if missing_token.any():
            plot_df.loc[missing_token, "target_token"] = plot_df.loc[missing_token, class_col].map(_display).astype(str)

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


def _feature_class_group_xlim(
    idxs: Sequence[int],
    cls: str,
    classes_present: Sequence[str],
    *,
    outer_pad: float = _TARGET_GROUP_PAD,
    inner_pad: float = _TARGET_GROUP_INNER_PAD,
) -> Tuple[float, float]:
    """Span bounds for a feature-class group with tighter padding between adjacent groups."""
    if not idxs:
        return 0.0, 0.0
    pos = classes_present.index(cls)
    left_pad = outer_pad if pos == 0 else inner_pad
    right_pad = outer_pad if pos == len(classes_present) - 1 else inner_pad
    return min(idxs) - left_pad, max(idxs) + right_pad


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

    classes_present = [
        cls
        for cls in class_order
        if any(target_to_class.get(tok) == cls for tok in target_order)
    ]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_line = 1.02
    for cls in class_order:
        idxs = [i for i, tok in enumerate(target_order) if target_to_class.get(tok) == cls]
        if not idxs:
            continue
        x0, x1 = _feature_class_group_xlim(idxs, cls, classes_present)
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


def _shade_feature_class_groups(
    ax: Any,
    *,
    target_order: Sequence[str],
    target_to_class: dict,
    class_order: Sequence[str],
) -> None:
    if not target_order or not target_to_class:
        return
    classes_present = [
        cls
        for cls in class_order
        if any(target_to_class.get(tok) == cls for tok in target_order)
    ]
    for cls in class_order:
        idxs = [i for i, tok in enumerate(target_order) if target_to_class.get(tok) == cls]
        if not idxs:
            continue
        x0, x1 = _feature_class_group_xlim(idxs, cls, classes_present)
        ax.axvspan(x0, x1, color="0.99", zorder=0)


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


def _add_example_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_col = _text_column(out)
    id_cols = [
        col for col in [
            text_col,
            "target_token",
            "feature_class",
            "factual_token",
            "alternative_token",
            "source_token",
        ]
        if col and col in out.columns
    ]
    if not id_cols:
        out["example_id"] = [f"row-{idx}" for idx in out.index]
        return out

    def _id(row: pd.Series) -> str:
        raw = "||".join(str(row.get(col, "")) for col in id_cols)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]

    out["example_id"] = out.apply(_id, axis=1)
    return out


def _plot_encoder_by_target_seed_grid_interactive(
    plot_df: pd.DataFrame,
    *,
    seed_col: str,
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

    df = _add_example_ids(plot_df)
    text_col = _text_column(df)
    if text_col is not None:
        df["text_hover"] = df[text_col].map(_wrap_hover_text)
    hover_data = [
        col for col in [
            "example_id",
            "text_hover",
            "target_token",
            "feature_class",
            seed_col,
            "plot_hue",
            "encoded",
        ]
        if col in df.columns
    ]
    fig = px.strip(
        df,
        x="target_token",
        y="encoded",
        color="plot_hue",
        facet_row=seed_col,
        category_orders={
            "target_token": list(target_order),
            "plot_hue": list(split_hue_order),
        },
        hover_data=hover_data or None,
        custom_data=["example_id"],
        labels=plotly_labels_for(["target_token", "encoded", "plot_hue", seed_col, *hover_data]),
        title=title,
        height=height,
    )
    fig.update_traces(marker={"opacity": 0.75})
    fig.update_layout(
        xaxis_title=format_plotly_label("target"),
        yaxis_title=format_plotly_label("encoded"),
        legend_title_text=format_plotly_label("plot_hue"),
        hovermode="closest",
    )
    if output:
        html_output = output
        base, ext = os.path.splitext(html_output)
        if ext.lower() not in {".html", ".htm"}:
            html_output = f"{base}.html"
        os.makedirs(os.path.dirname(html_output) or ".", exist_ok=True)
        highlight_script = """
        const gd = document.getElementById('{plot_id}');
        function clearSelection() {
          const updates = [];
          const traceIds = [];
          for (let i = 0; i < gd.data.length; i++) {
            updates.push(null);
            traceIds.push(i);
          }
          Plotly.restyle(gd, {'selectedpoints': updates}, traceIds);
        }
        gd.on('plotly_click', function(eventData) {
          if (!eventData.points || !eventData.points.length) return;
          const clicked = eventData.points[0].customdata && eventData.points[0].customdata[0];
          if (!clicked) return;
          const selected = [];
          const traceIds = [];
          for (let i = 0; i < gd.data.length; i++) {
            const trace = gd.data[i];
            const points = [];
            const custom = trace.customdata || [];
            for (let j = 0; j < custom.length; j++) {
              if (custom[j] && custom[j][0] === clicked) points.push(j);
            }
            selected.push(points);
            traceIds.push(i);
          }
          Plotly.restyle(gd, {
            'selectedpoints': selected,
            'selected.marker.opacity': 1.0,
            'selected.marker.size': 9,
            'unselected.marker.opacity': 0.12
          }, traceIds);
        });
        gd.on('plotly_doubleclick', function() {
          clearSelection();
        });
        """
        fig.write_html(html_output, post_script=highlight_script)
        logger.info("Saved interactive multi-seed encoder by-target plot: %s", html_output)
        if show:
            fig.show()
        return html_output
    if show:
        fig.show()
    return fig


def _plot_encoder_by_target_seed_errorbar(
    plot_df: pd.DataFrame,
    *,
    seed_col: str,
    target_order: Sequence[str],
    split_hue_order: Sequence[str],
    target_to_class: dict,
    class_order: Sequence[str],
    output: Optional[str],
    output_dir: Optional[str],
    experiment_dir: Optional[str],
    show: bool,
    figsize: Optional[Tuple[float, float]],
    title: Optional[str],
    error_stat: str,
    show_seed_points: bool,
    error_group_by_split: bool,
) -> Optional[str]:
    plt = _require_matplotlib()
    sns = _require_seaborn()
    error_stat = str(error_stat).strip().lower()
    if error_stat not in {"std", "sem"}:
        raise ValueError("error_stat must be 'std' or 'sem'")

    group_cols = [seed_col, "target_token", "feature_class"]
    if error_group_by_split:
        group_cols.append("plot_hue")
    seed_target = (
        plot_df.groupby(group_cols, sort=False)["encoded"]
        .mean()
        .reset_index(name="seed_mean")
    )
    summary_group_cols = ["target_token", "feature_class"]
    if error_group_by_split:
        summary_group_cols.append("plot_hue")
    summary = (
        seed_target.groupby(summary_group_cols, sort=False)["seed_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["err"] = summary["std"].fillna(0.0)
    if error_stat == "sem":
        summary["err"] = summary["err"] / np.sqrt(summary["count"].clip(lower=1))

    class_order_resolved = list(class_order)
    color_order = list(split_hue_order) if error_group_by_split else class_order_resolved
    palette = sns.color_palette(n_colors=max(len(color_order), 1))
    colors = {str(value): palette[idx % len(palette)] for idx, value in enumerate(color_order)}
    split_offsets = (
        np.linspace(-0.24, 0.24, len(split_hue_order))
        if error_group_by_split and len(split_hue_order) > 1
        else np.array([0.0])
    )
    split_to_offset = {str(split): float(split_offsets[idx]) for idx, split in enumerate(split_hue_order)}
    x_by_target = {tok: idx for idx, tok in enumerate(target_order)}
    width = max(5.0, 0.16 * len(target_order))
    _figsize = figsize if figsize is not None else (width, 2.9)
    fig, ax = plt.subplots(figsize=_figsize)

    if show_seed_points:
        for _, row in seed_target.iterrows():
            tok = row["target_token"]
            if tok not in x_by_target:
                continue
            split = str(row.get("plot_hue", ""))
            x = x_by_target[tok] + (split_to_offset.get(split, 0.0) if error_group_by_split else 0.0)
            ax.scatter(
                x,
                row["seed_mean"],
                color="0.35",
                alpha=0.28,
                s=9,
                linewidths=0,
                zorder=2,
            )

    for _, row in summary.iterrows():
        tok = row["target_token"]
        if tok not in x_by_target:
            continue
        color_key = str(row.get("plot_hue", row["feature_class"])) if error_group_by_split else str(row["feature_class"])
        x = x_by_target[tok] + (split_to_offset.get(str(row.get("plot_hue", "")), 0.0) if error_group_by_split else 0.0)
        ax.errorbar(
            x,
            row["mean"],
            yerr=row["err"],
            fmt="o",
            color=colors.get(color_key, "0.2"),
            ecolor=colors.get(color_key, "0.2"),
            capsize=2.5,
            elinewidth=1.0,
            markersize=4.0,
            zorder=3,
        )

    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=1)
    ax.set_ylabel("Mean encoded value")
    ax.set_xlabel("Target")
    if title:
        ax.set_title(title, pad=30)
    ax.set_xticks(list(range(len(target_order))))
    ax.set_xticklabels(list(target_order), rotation=90, ha="center", fontsize=8)
    _add_feature_class_group_brackets(
        ax,
        target_order=target_order,
        target_to_class=target_to_class,
        class_order=class_order_resolved,
    )
    _tighten_target_axis_xlim(ax, len(target_order))
    if error_group_by_split:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors.get(str(split), "0.2"), markersize=5)
            for split in split_hue_order
        ]
        ax.legend(handles, [str(split) for split in split_hue_order], title="Split", loc="upper right")
    fig.tight_layout()

    out = output or resolve_output_path(
        output_dir=output_dir,
        experiment_dir=experiment_dir,
        default_name="encoder_by_target_multi_seed_errorbar.pdf",
        artifact_subdir=ARTIFACT_ENCODER_PLOT,
    )
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def _plot_encoder_by_target_seed_strip_combined(
    plot_df: pd.DataFrame,
    *,
    seed_col: str,
    target_order: Sequence[str],
    split_hue_order: Sequence[str],
    target_to_class: dict,
    class_order: Sequence[str],
    output: Optional[str],
    output_dir: Optional[str],
    experiment_dir: Optional[str],
    show: bool,
    figsize: Optional[Tuple[float, float]],
    point_size: float,
    title: Optional[str],
) -> Optional[str]:
    plt = _require_matplotlib()
    seed_order = sorted(
        plot_df[seed_col].dropna().astype(str).unique().tolist(),
        key=lambda value: int(value) if str(value).lstrip("-").isdigit() else str(value),
    )
    if not seed_order:
        logger.warning("No seed values for combined multi-seed by-target plot")
        return None
    split_colors = _split_color_map(split_hue_order)
    seed_offsets = np.linspace(-0.32, 0.32, len(seed_order)) if len(seed_order) > 1 else np.array([0.0])
    seed_to_offset = {seed: float(seed_offsets[idx]) for idx, seed in enumerate(seed_order)}
    x_by_target = {tok: idx for idx, tok in enumerate(target_order)}

    df = plot_df.copy()
    df["_target_x"] = df["target_token"].map(x_by_target).astype(float)
    df["_seed_label"] = df[seed_col].astype(str)
    df["_x"] = df["_target_x"] + df["_seed_label"].map(seed_to_offset).astype(float)

    width = max(5.0, 0.18 * len(target_order))
    _figsize = figsize if figsize is not None else (width, 3.1)
    fig, ax = plt.subplots(figsize=_figsize)

    for split in split_hue_order:
        sub = df[df["plot_hue"] == split]
        if sub.empty:
            continue
        ax.scatter(
            sub["_x"],
            sub["encoded"],
            color=split_colors.get(split, "0.35"),
            s=max(1.0, point_size * 4.0),
            alpha=0.72,
            linewidths=0,
            label=str(split),
            zorder=3,
        )

    for seed in seed_order:
        offset = seed_to_offset[seed]
        ax.scatter([], [], color="0.35", s=max(1.0, point_size * 4.0), label=f"seed {seed}", alpha=0.35)
        for idx in range(len(target_order)):
            ax.plot(
                [idx + offset, idx + offset],
                [-1.04, -1.0],
                color="0.45",
                linewidth=0.45,
                clip_on=False,
                zorder=2,
            )

    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=1)
    ax.set_ylabel("Encoded value")
    ax.set_xlabel("Target")
    if title:
        ax.set_title(title, pad=30)
    ax.set_xticks(list(range(len(target_order))))
    ax.set_xticklabels(list(target_order), rotation=90, ha="center", fontsize=8)
    _add_feature_class_group_brackets(
        ax,
        target_order=target_order,
        target_to_class=target_to_class,
        class_order=class_order,
    )
    _tighten_target_axis_xlim(ax, len(target_order))
    handles, labels = ax.get_legend_handles_labels()
    split_handles = [(h, l) for h, l in zip(handles, labels) if not str(l).startswith("seed ")]
    if split_handles:
        ax.legend(
            [h for h, _ in split_handles],
            [l for _, l in split_handles],
            title="Split",
            loc="upper right",
        )
    fig.tight_layout()

    out = output or resolve_output_path(
        output_dir=output_dir,
        experiment_dir=experiment_dir,
        default_name="encoder_by_target_multi_seed_combined_strip.pdf",
        artifact_subdir=ARTIFACT_ENCODER_PLOT,
    )
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_encoder_by_target_seed_grid(
    encoder_df: pd.DataFrame,
    *,
    seed_col: str = "seed",
    target_col: str = "source_token",
    class_col: str = "source_id",
    hue_col: str = "data_split",
    class_order: Optional[Sequence[str]] = None,
    id2label: Optional[dict] = None,
    output: Optional[str] = None,
    output_dir: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    jitter: float = 0.25,
    point_size: float = 1.5,
    title: Optional[str] = None,
    plot_style: str = "strip",
    interactive: bool = False,
    height: int = 900,
    error_stat: str = "std",
    show_seed_points: bool = True,
    combine_seed_rows: bool = False,
    error_group_by_split: bool = True,
) -> Optional[str]:
    """Plot held-out target encodings as one shared-x row per seed."""
    if encoder_df is None or encoder_df.empty:
        logger.warning("No encoder data for multi-seed by-target plot")
        return None
    if seed_col not in encoder_df.columns:
        raise ValueError(f"seed column {seed_col!r} not in encoder DataFrame")

    plot_df, target_order, split_hue_order, target_to_class, target_to_split = (
        build_encoder_target_plot_frame(
            encoder_df,
            target_col=target_col,
            class_col=class_col,
            hue_col=hue_col,
            class_order=class_order,
            id2label=id2label,
        )
    )
    if plot_df.empty or not target_order:
        logger.warning("No plottable target tokens for multi-seed by-target plot")
        return None

    plot_df[seed_col] = encoder_df.loc[plot_df.index, seed_col].astype(str)
    seed_order = sorted(plot_df[seed_col].dropna().unique().tolist(), key=lambda value: int(value) if str(value).lstrip("-").isdigit() else str(value))
    class_order_resolved = class_order or sorted({target_to_class[t] for t in target_order})

    style = str(plot_style).strip().lower()
    if style in {"error", "errorbar", "summary"}:
        return _plot_encoder_by_target_seed_errorbar(
            plot_df,
            seed_col=seed_col,
            target_order=target_order,
            split_hue_order=split_hue_order,
            target_to_class=target_to_class,
            class_order=class_order_resolved,
            output=output,
            output_dir=output_dir,
            experiment_dir=experiment_dir,
            show=show,
            figsize=figsize,
            title=title,
            error_stat=error_stat,
            show_seed_points=show_seed_points,
            error_group_by_split=error_group_by_split,
        )
    if style != "strip":
        raise ValueError("multi-seed by-target plot_style must be 'strip' or 'errorbar'")

    if combine_seed_rows:
        return _plot_encoder_by_target_seed_strip_combined(
            plot_df,
            seed_col=seed_col,
            target_order=target_order,
            split_hue_order=split_hue_order,
            target_to_class=target_to_class,
            class_order=class_order_resolved,
            output=output,
            output_dir=output_dir,
            experiment_dir=experiment_dir,
            show=show,
            figsize=figsize,
            point_size=point_size,
            title=title,
        )

    if interactive:
        return _plot_encoder_by_target_seed_grid_interactive(
            plot_df,
            seed_col=seed_col,
            target_order=target_order,
            split_hue_order=split_hue_order,
            title=title,
            output=output,
            show=show,
            height=height,
        )

    plt = _require_matplotlib()
    sns = _require_seaborn()
    width = max(5.0, 0.15 * len(target_order))
    height = max(1.15 * len(seed_order) + 0.8, 2.2)
    _figsize = figsize if figsize is not None else (width, height)
    fig, axes = plt.subplots(
        len(seed_order),
        1,
        figsize=_figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    split_colors = _split_color_map(split_hue_order)
    palette = [split_colors[split] for split in split_hue_order]

    for row_idx, seed in enumerate(seed_order):
        ax = axes[row_idx][0]
        sub = plot_df[plot_df[seed_col] == seed]
        sns.stripplot(
            data=sub,
            x="target_token",
            y="encoded",
            hue="plot_hue",
            hue_order=split_hue_order,
            order=target_order,
            palette=palette,
            ax=ax,
            dodge=False,
            jitter=jitter,
            size=point_size,
            alpha=0.85,
        )
        ax.set_ylabel(f"seed {seed}")
        ax.set_xlabel("")
        if row_idx < len(seed_order) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Target")
            ax.tick_params(axis="x", labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        _shade_feature_class_groups(
            ax,
            target_order=target_order,
            target_to_class=target_to_class,
            class_order=class_order_resolved,
        )
        _tighten_target_axis_xlim(ax, len(target_order))
        if row_idx > 0:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    first_ax = axes[0][0]
    handles, labels = first_ax.get_legend_handles_labels()
    if handles and labels:
        first_ax.legend(handles, labels, title="Split", loc="upper right")
    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985 if title else 1))
    fig.subplots_adjust(hspace=0.04)

    out = output or resolve_output_path(
        output_dir=output_dir,
        experiment_dir=experiment_dir,
        default_name="encoder_by_target_multi_seed.pdf",
        artifact_subdir=ARTIFACT_ENCODER_PLOT,
    )
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


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
    fallback: Optional[str] = None
    for col in TEXT_HOVER_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        non_empty = values[values.ne("")]
        if non_empty.empty:
            continue
        if non_empty.str.contains("[MASK]", regex=False).any():
            return col
        if fallback is None:
            fallback = col
    return fallback


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
        ]
        if col in df.columns
    ]
    plotly_labels = plotly_labels_for(["target_token", "encoded", "plot_hue", *hover_data])
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
        labels=plotly_labels,
        title=title,
        height=height,
    )
    fig.update_traces(marker={"opacity": 0.78})
    fig.update_layout(
        xaxis_title=format_plotly_label("target"),
        yaxis_title=format_plotly_label("encoded"),
        legend_title_text=format_plotly_label("plot_hue"),
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
    target_col: str = "source_token",
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
