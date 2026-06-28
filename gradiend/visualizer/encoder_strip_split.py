"""
Matplotlib strip-style encoder scatter grouped by feature class with optional point labels.

Used for split-colored encoder stability plots (train / validation / test dodge).
Supports efficient outlier-only annotation with optional adjustText de-overlap.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from gradiend.util.encoder_splits import order_split_names
from gradiend.util.logging import get_logger
from gradiend.util.paths import ARTIFACT_ENCODER_PLOT, resolve_output_path
from gradiend.visualizer.labels import (
    resolve_highlight_non_convergence,
    resolve_plot_title_with_convergence,
)
from gradiend.visualizer.encoder_neutral import (
    NEUTRAL_ENCODER_TYPES,
    NEUTRAL_TYPE_VIOLIN_LABELS,
    encoder_plot_xlabel,
)
from gradiend.visualizer.encoder_scatter import _truncate_text_around_mask
from gradiend.visualizer.plot_optional import _require_matplotlib, _require_seaborn

logger = get_logger(__name__)

DEFAULT_PLOT_TYPES = ("training",)
NEUTRAL_PLOT_TYPES = NEUTRAL_ENCODER_TYPES
_DEFAULT_STRIP_BY_SPLIT_HEIGHT = 4.5
LabelMode = Union[bool, Literal["outliers", "outliers+sample", "sample"], str]


def _encoded_scalar(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "__len__") and len(value) > 0:
        return float(value[0])
    return None


def select_encoded_outlier_indices(
    df: pd.DataFrame,
    *,
    y_col: str = "encoded",
    group_col: str = "x_group",
    method: Literal["iqr", "zscore"] = "iqr",
    k: float = 1.5,
) -> List[Any]:
    """Row indices of encoded-value outliers within each ``group_col`` stratum.

    Args:
        df: Plot DataFrame containing encoded values.
        y_col: Numeric encoded-value column.
        group_col: Column defining independent outlier strata.
        method: Outlier rule, either interquartile range (``iqr``) or standard score
            (``zscore``).
        k: Rule multiplier. For IQR this is the whisker multiplier; for z-score this is
            the absolute z-score threshold.
    """
    if df.empty or y_col not in df.columns:
        return []
    indices: List[Any] = []
    for _, grp in df.groupby(group_col, sort=False):
        y = grp[y_col].map(_encoded_scalar)
        valid = y.notna()
        if not valid.any():
            continue
        yv = y[valid].astype(float)
        if method == "iqr":
            q1, q3 = yv.quantile(0.25), yv.quantile(0.75)
            iqr = float(q3 - q1)
            if iqr <= 0:
                continue
            mask = (yv < q1 - k * iqr) | (yv > q3 + k * iqr)
        else:
            std = float(yv.std())
            if std <= 0:
                continue
            mask = (yv - float(yv.mean())).abs() > k * std
        indices.extend(yv.index[mask].tolist())
    return indices


def select_trend_sample_indices(
    df: pd.DataFrame,
    *,
    y_col: str = "encoded",
    group_cols: Sequence[str] = ("x_group",),
    n_per_group: int = 2,
) -> List[Any]:
    """Evenly spaced encoded samples within each group stratum.

    Args:
        df: Plot DataFrame containing encoded values.
        y_col: Numeric encoded-value column.
        group_cols: Columns defining independent sampling strata.
        n_per_group: Maximum number of sampled row indices per stratum.
    """
    if n_per_group <= 0 or df.empty:
        return []
    indices: List[Any] = []
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        cols = ["x_group"]
    for _, grp in df.groupby(cols, sort=False):
        y = grp[y_col].map(_encoded_scalar)
        valid = y.notna()
        if not valid.any():
            continue
        ordered = y[valid].sort_values()
        if len(ordered) <= n_per_group:
            indices.extend(ordered.index.tolist())
        else:
            pos = np.linspace(0, len(ordered) - 1, n_per_group, dtype=int)
            indices.extend(ordered.iloc[pos].index.tolist())
    return list(dict.fromkeys(indices))


def _resolve_label_text(
    row: pd.Series,
    *,
    label_col: str,
    label_max_chars: int,
    label_formatter: Optional[Callable[[pd.Series], str]],
) -> str:
    if label_formatter is not None:
        return str(label_formatter(row))
    for col in (label_col, "factual_token", "text", "masked", "source_id"):
        if col not in row.index:
            continue
        val = row[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        text = str(val).strip()
        if not text:
            continue
        if col == "text":
            return _truncate_text_around_mask(text, label_max_chars)
        if len(text) > label_max_chars:
            return text[: max(0, label_max_chars - 3)] + "..."
        return text
    return ""


def _label_indices_from_mode(
    plot_df: pd.DataFrame,
    *,
    label_points: LabelMode,
    label_indices: Optional[Sequence[Any]],
    outlier_method: Literal["iqr", "zscore"],
    outlier_k: float,
    label_sample_per_group: int = 2,
    sample_group_cols: Sequence[str] = ("x_group", "data_split"),
) -> List[Any]:
    indices: List[Any] = []
    if label_indices:
        index_set = set(plot_df.index)
        indices.extend(i for i in label_indices if i in index_set)
    if label_points is False or label_points is None:
        return list(dict.fromkeys(indices))

    want_outliers = label_points is True or label_points in ("outliers", "outliers+sample")
    want_sample = label_points is True or label_points in ("sample", "outliers+sample")

    if want_outliers:
        indices.extend(
            select_encoded_outlier_indices(
                plot_df,
                method=outlier_method,
                k=outlier_k,
            )
        )
    if want_sample:
        indices.extend(
            select_trend_sample_indices(
                plot_df,
                group_cols=sample_group_cols,
                n_per_group=label_sample_per_group,
            )
        )

    if not want_outliers and not want_sample:
        if isinstance(label_points, str):
            if label_points == "outliers":
                indices.extend(
                    select_encoded_outlier_indices(
                        plot_df,
                        method=outlier_method,
                        k=outlier_k,
                    )
                )
            elif label_points == "sample":
                indices.extend(
                    select_trend_sample_indices(
                        plot_df,
                        group_cols=sample_group_cols,
                        n_per_group=label_sample_per_group,
                    )
                )
            elif label_points == "outliers+sample":
                pass  # handled above via want_* flags when passed as literal - unreachable
            elif label_points in plot_df.columns:
                col = plot_df[label_points]
                indices.extend(plot_df.index[col.notna()].tolist())
            else:
                raise ValueError(
                    f"label_points={label_points!r} is not a valid mode or column. "
                    "Use True, 'outliers', 'sample', 'outliers+sample', or a column name."
                )
        else:
            raise TypeError(
                f"label_points must be bool, 'outliers', 'sample', 'outliers+sample', or str; "
                f"got {type(label_points).__name__}"
            )
    return list(dict.fromkeys(indices))


def _apply_point_labels(
    ax: Any,
    xs: Sequence[float],
    ys: Sequence[float],
    texts: Sequence[str],
    *,
    adjust_labels: bool,
    fontsize: float,
) -> None:
    if not texts:
        return
    text_artists = [
        ax.text(float(x), float(y), str(t), fontsize=fontsize, ha="center", va="bottom")
        for x, y, t in zip(xs, ys, texts)
        if t
    ]
    if not text_artists:
        return
    if adjust_labels:
        try:
            from adjustText import adjust_text

            adjust_text(
                text_artists,
                ax=ax,
                expand=(1.05, 1.2),
                arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.6},
            )
            return
        except ImportError:
            logger.info(
                "adjustText not installed; using simple label offsets. "
                "Install adjustText for better label de-overlap: pip install gradiend[plot]"
            )
    y0, y1 = ax.get_ylim()
    y_span = max(y1 - y0, 1e-9)
    step = 0.035 * y_span
    for i, artist in enumerate(text_artists):
        x, y = artist.get_position()
        artist.set_position((x, y + (i % 4) * step))


def _prepare_strip_plot_frame(
    encoder_df: pd.DataFrame,
    *,
    types: Optional[Sequence[str]] = None,
    x_group_col: str = "source_id",
    neutral_type_labels: Optional[dict] = None,
    hue_col: str = "data_split",
    default_hue: str = "test",
) -> pd.DataFrame:
    allowed = set(types or DEFAULT_PLOT_TYPES)
    plot_df = encoder_df[encoder_df["type"].isin(allowed)].copy()
    if plot_df.empty:
        return plot_df
    if hue_col not in plot_df.columns:
        plot_df[hue_col] = default_hue
    plot_df["x_group"] = plot_df[x_group_col].astype(str)
    for neutral_type, label in (neutral_type_labels or NEUTRAL_TYPE_VIOLIN_LABELS).items():
        plot_df.loc[plot_df["type"] == neutral_type, "x_group"] = label
    plot_df["encoded"] = plot_df["encoded"].map(_encoded_scalar)
    return plot_df.dropna(subset=["encoded"])


def aggregate_strip_targets_to_mean(
    plot_df: pd.DataFrame,
    *,
    target_col: str = "factual_token",
    hue_col: str = "data_split",
) -> pd.DataFrame:
    """Collapse repeated target rows to mean encoded value per target/group/split.

    Args:
        plot_df: Prepared strip-plot DataFrame.
        target_col: Target-token column used for aggregation.
        hue_col: Hue/split column preserved during aggregation.
    """
    if plot_df.empty or target_col not in plot_df.columns:
        return plot_df
    group_cols = [c for c in ("x_group", hue_col, target_col, "type") if c in plot_df.columns]
    if target_col not in group_cols or "x_group" not in group_cols:
        return plot_df
    value_cols = {target_col, "x_group", hue_col, "type", "encoded"}
    first_cols = [c for c in plot_df.columns if c not in value_cols]
    agg = {"encoded": "mean"}
    agg.update({c: "first" for c in first_cols})
    out = plot_df.groupby(group_cols, sort=False, dropna=False).agg(agg).reset_index()
    return out[plot_df.columns.intersection(out.columns).tolist()]


def _default_strip_by_split_figsize(n_groups: int) -> Tuple[float, float]:
    """Width scales with x groups; keep class-level overviews compact."""
    width = max(6.0, 1.8 * max(n_groups, 1))
    return (width, _DEFAULT_STRIP_BY_SPLIT_HEIGHT)


def plot_encoder_strip_by_split(
    trainer: Any = None,
    encoder_df: Optional[pd.DataFrame] = None,
    *,
    types: Optional[Sequence[str]] = None,
    include_neutral: bool = False,
    x_group_col: str = "source_id",
    neutral_type_labels: Optional[dict] = None,
    hue_col: str = "data_split",
    default_hue: str = "test",
    title: Union[str, bool, None] = True,
    output: Optional[str] = None,
    output_dir: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    jitter: float = 0.08,
    dodge: bool = True,
    point_size: float = 5.0,
    seed: int = 42,
    aggregate_targets: bool = True,
    aggregate_target_col: str = "factual_token",
    label_points: LabelMode = False,
    label_indices: Optional[Sequence[Any]] = None,
    label_col: str = "factual_token",
    label_max_chars: int = 40,
    label_formatter: Optional[Callable[[pd.Series], str]] = None,
    label_sample_per_group: int = 2,
    adjust_labels: bool = True,
    label_fontsize: float = 7.0,
    outlier_method: Literal["iqr", "zscore"] = "iqr",
    outlier_k: float = 1.5,
    highlight_non_convergence: Optional[bool] = None,
    **kwargs: Any,
) -> Optional[str]:
    """
    Strip-style encoder scatter: x = feature group, y = encoded, hue = data split.

    Args:
        trainer: Optional trainer used to compute ``encoder_df`` and resolve convergence.
        encoder_df: Encoder analysis DataFrame (or obtained via trainer + kwargs).
        types: Encoder row types to plot. ``None`` uses training rows and, when requested,
            neutral row types.
        x_group_col: Column mapped to x-axis groups for non-neutral rows.
        neutral_type_labels: Display labels for neutral row types.
        hue_col: Column used as hue, typically ``data_split``.
        default_hue: Hue value inserted when ``hue_col`` is missing.
        title: Plot title. ``True`` (default) uses "Encoded values by split";
            ``None``/``False`` disables it.
        output: Explicit output path.
        output_dir: Directory used when resolving the default output filename.
        experiment_dir: Experiment directory used when resolving the default output filename.
        show: Whether to display the plot.
        jitter: Horizontal jitter width.
        dodge: Whether to dodge points by hue.
        point_size: Marker size.
        seed: Random seed for deterministic jitter.
        aggregate_target_col: Target-token column used when ``aggregate_targets`` is true.
        label_points: ``False`` (default) for no labels; ``True`` or ``'outliers+sample'`` for
            outliers plus evenly spaced points per group; ``'outliers'`` / ``'sample'`` for one
            mode only; or a column name to label all non-null rows.
        label_sample_per_group: Trend sample count per (x_group, split) when sampling is enabled.
        include_neutral: When False (default), only training rows are plotted. Set True to
            add neutral_dataset / neutral_training_masked groups (x-axis label becomes Target).
        label_indices: Explicit row indices to annotate (unioned with *label_points*).
        adjust_labels: When True, use adjustText if installed to reduce label overlap.
        label_col: Preferred column for label text (falls back to text / factual_token).
        aggregate_targets: When True (default), repeated target tokens are shown once per
            feature group and split using their mean encoded value.
        figsize: Figure size in inches. When None (default), width scales with the number of
            x groups (minimum 6 inches).
        label_max_chars: Maximum text length for generated point labels.
        label_formatter: Optional callable that formats label text from each row.
        label_fontsize: Font size for point labels.
        outlier_method: Outlier rule for outlier labels.
        outlier_k: Outlier-rule multiplier.
        highlight_non_convergence: When True, append a non-convergence marker to the title
            for non-converged runs. ``None`` uses trainer settings when available.
        **kwargs: Forwarded to ``trainer.analyze_encoder`` when ``encoder_df`` is not supplied.
    """
    _ = kwargs
    if encoder_df is None:
        if trainer is None:
            raise ValueError("Provide encoder_df or trainer")
        encoder_df = trainer.analyze_encoder(getattr(trainer, "get_model", lambda: None)(), **kwargs)
    if encoder_df is None or encoder_df.empty:
        logger.warning("No encoder data for strip-by-split plot")
        return None

    plot_types = types
    if plot_types is None:
        plot_types = DEFAULT_PLOT_TYPES + (NEUTRAL_PLOT_TYPES if include_neutral else ())
    includes_neutral = include_neutral or any(t in NEUTRAL_PLOT_TYPES for t in plot_types)

    plot_df = _prepare_strip_plot_frame(
        encoder_df,
        types=plot_types,
        x_group_col=x_group_col,
        neutral_type_labels=neutral_type_labels,
        hue_col=hue_col,
        default_hue=default_hue,
    )
    if plot_df.empty:
        logger.warning("No plottable encoder rows for strip-by-split plot")
        return None
    if aggregate_targets:
        plot_df = aggregate_strip_targets_to_mean(
            plot_df,
            target_col=aggregate_target_col,
            hue_col=hue_col,
        )

    highlight = resolve_highlight_non_convergence(highlight_non_convergence, trainer=trainer)
    default_title = "Encoded values by split"
    plot_title = resolve_plot_title_with_convergence(
        title,
        trainer=trainer,
        highlight_non_convergence=highlight,
        default=default_title,
    )

    annotate_indices = _label_indices_from_mode(
        plot_df,
        label_points=label_points,
        label_indices=label_indices,
        outlier_method=outlier_method,
        outlier_k=outlier_k,
        label_sample_per_group=label_sample_per_group,
        sample_group_cols=(hue_col, "x_group"),
    )

    plt = _require_matplotlib()
    sns = _require_seaborn()
    rng = np.random.default_rng(seed)

    group_order = list(dict.fromkeys(plot_df["x_group"].astype(str).tolist()))
    hue_values = plot_df[hue_col].dropna().astype(str).tolist()
    hue_order = order_split_names(set(hue_values)) if hue_values else ["test"]

    n_groups = max(len(group_order), 1)
    plot_figsize = figsize if figsize is not None else _default_strip_by_split_figsize(n_groups)

    fig, ax = plt.subplots(figsize=plot_figsize)
    n_hue = max(len(hue_order), 1)
    dodge_width = 0.8 / n_hue if dodge else 0.0
    effective_jitter = float(jitter)
    if dodge and dodge_width > 0:
        effective_jitter = min(effective_jitter, dodge_width * 0.35)
    palette = sns.color_palette(n_colors=n_hue)

    xs_all: List[float] = []
    ys_all: List[float] = []
    label_xs: List[float] = []
    label_ys: List[float] = []
    label_texts: List[str] = []

    for hi, hue_val in enumerate(hue_order):
        hue_offset = (hi - (n_hue - 1) / 2.0) * dodge_width if dodge else 0.0
        sub_hue = plot_df[plot_df[hue_col].astype(str) == hue_val]
        xs: List[float] = []
        ys: List[float] = []
        for gi, group in enumerate(group_order):
            sub = sub_hue[sub_hue["x_group"].astype(str) == group]
            for idx, row in sub.iterrows():
                x = float(gi) + hue_offset + float(rng.uniform(-effective_jitter, effective_jitter))
                y = float(row["encoded"])
                xs.append(x)
                ys.append(y)
                xs_all.append(x)
                ys_all.append(y)
                if idx in annotate_indices:
                    label_xs.append(x)
                    label_ys.append(y)
                    label_texts.append(
                        _resolve_label_text(
                            row,
                            label_col=label_col,
                            label_max_chars=label_max_chars,
                            label_formatter=label_formatter,
                        )
                    )
        if xs:
            ax.scatter(
                xs,
                ys,
                s=point_size**2,
                color=palette[hi % len(palette)],
                alpha=0.85,
                label=str(hue_val),
                edgecolors="none",
            )

    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order)
    ax.set_ylabel("Encoded value")
    ax.set_xlabel(encoder_plot_xlabel(includes_neutral_groups=includes_neutral))
    if plot_title:
        ax.set_title(str(plot_title))
    if n_hue > 1:
        ax.legend(title="Split", loc="best", fontsize=8)

    _apply_point_labels(
        ax,
        label_xs,
        label_ys,
        label_texts,
        adjust_labels=adjust_labels,
        fontsize=label_fontsize,
    )

    fig.tight_layout()

    out = output
    if out is None:
        run_id = getattr(trainer, "run_id", None) if trainer is not None else None
        out = resolve_output_path(
            experiment_dir or (getattr(trainer, "experiment_dir", None) if trainer is not None else None),
            None,
            ARTIFACT_ENCODER_PLOT,
            run_id=run_id,
        )
        if out is not None:
            base, _ = os.path.splitext(out)
            out = f"{base}_strip_by_split.png"
        elif output_dir:
            out = os.path.join(output_dir, "encoder_scatter_by_split.png")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        logger.info("Saved encoder strip-by-split plot: %s", out)

    if show:
        plt.show()
    plt.close(fig)
    return out
