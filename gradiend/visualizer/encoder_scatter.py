"""
Interactive 1D encoder scatter: categorical x-axis, encoded value on y-axis, colored by label, with hover.
For outlier analysis in Jupyter. Uses Plotly for interaction.
Colors match encoder distribution violins (tab20) when cmap is used.
Colormap lookup requires matplotlib; if missing, raises ImportError with install instructions.
"""

import os
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from gradiend.util.paths import resolve_output_path, ARTIFACT_ENCODER_PLOT
from gradiend.visualizer.labels import (
    format_plotly_label,
    plotly_labels_for,
    resolve_highlight_non_convergence,
    resolve_plot_title_with_convergence,
)
from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

# Default cmap for scatter to match encoder_distributions violins
DEFAULT_SCATTER_CMAP = "tab20"
TEXT_HOVER_COLUMNS = ("display_text", "text_:hover", "text_hover", "template", "masked", "input_text", "text", "sentence")


def _palette_for_categories(categories: List[str], cmap: str = DEFAULT_SCATTER_CMAP) -> dict:
    """Map category -> hex color using the same cmap as encoder violins (tab20)."""
    plt = _require_matplotlib()
    cmap_obj = plt.get_cmap(cmap)
    n_cmap = getattr(cmap_obj, "N", 20) or 20
    n_cat = max(1, len(categories))
    colors = [
        cmap_obj((i % n_cmap) / n_cmap) if n_cmap > 0 else cmap_obj(i / max(1, n_cat - 1))
        for i in range(n_cat)
    ]
    hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in colors]
    return dict(zip(categories, hex_colors))


def _encoded_1d(encoded_cell: Any) -> Optional[float]:
    """Extract single float from 1D encoded value (list, array, or scalar)."""
    if encoded_cell is None:
        return None
    if isinstance(encoded_cell, (int, float)):
        return float(encoded_cell)
    if hasattr(encoded_cell, "__len__") and len(encoded_cell) > 0:
        return float(encoded_cell[0])
    return None


def _wrap_hover_text(text: str, line_chars: int = 90) -> str:
    if line_chars <= 0:
        return text
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


def _truncate_text_around_mask(text: str, max_chars: int = 180, *, line_chars: int = 90) -> str:
    """Truncate hover text without hiding or chopping through the [MASK] context."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return _wrap_hover_text(text, line_chars=line_chars)

    def _trim_to_word_start(value: str, start: int, protected_end: int) -> int:
        if start <= 0:
            return 0
        while start < protected_end and not value[start].isspace() and not value[start - 1].isspace():
            start += 1
        return min(start, protected_end)

    def _trim_to_word_end(value: str, end: int, protected_start: int) -> int:
        if end >= len(value):
            return len(value)
        while end > protected_start and not value[end - 1].isspace() and not value[end].isspace():
            end -= 1
        return max(end, protected_start)

    def _truncate_prefix(value: str, limit: int) -> str:
        limit = max(1, limit)
        if len(value) <= limit:
            return value
        end = _trim_to_word_end(value, limit, 1)
        return value[:end].rstrip() if end > 1 else value[:limit].rstrip()

    mask = "[MASK]"
    idx = text.find(mask)
    if idx < 0:
        return _wrap_hover_text(_truncate_prefix(text, max_chars - 3) + "...", line_chars=line_chars)

    mask_end = idx + len(mask)
    context_budget = max(0, max_chars - len(mask) - 6)
    left_available = idx
    right_available = len(text) - mask_end
    left_budget = min(left_available, context_budget // 2)
    right_budget = min(right_available, context_budget - left_budget)
    left_budget = min(left_available, context_budget - right_budget)

    start = _trim_to_word_start(text, max(0, idx - left_budget), idx)
    end = _trim_to_word_end(text, min(len(text), mask_end + right_budget), mask_end)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    snippet = text[start:end].strip()
    return _wrap_hover_text(prefix + snippet + suffix, line_chars=line_chars)


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _first_masked_or_non_empty_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    fallback: Optional[str] = None
    for col in candidates:
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


def _is_text_hover_column(column: str) -> bool:
    return str(column).strip().casefold() in {c.casefold() for c in TEXT_HOVER_COLUMNS}


def _ordered_existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for column in columns:
        if column in df.columns and column not in out:
            out.append(column)
    return out


def _natural_category_order(values: Sequence[Any]) -> List[str]:
    def key(value: Any) -> tuple:
        text = str(value)
        try:
            return (0, float(text))
        except ValueError:
            pass
        lowered = text.casefold()
        if lowered in {"negative", "-1", "-1.0"}:
            return (1, 0, lowered)
        if lowered in {"neutral", "0", "0.0"}:
            return (1, 1, lowered)
        if lowered in {"positive", "1", "1.0"}:
            return (1, 2, lowered)
        return (2, lowered)

    return sorted({str(v) for v in values}, key=key)


def _stratified_subsample(
    df: pd.DataFrame,
    max_points: int,
    stratify_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Subsample up to max_points, stratified by stratify_col (same proportion per stratum)."""
    if stratify_col not in df.columns or len(df) <= max_points:
        return df
    groups = df.groupby(stratify_col, group_keys=False)
    n_per = max(1, max_points // len(groups))
    sampled = groups.apply(lambda g: g.sample(n=min(len(g), n_per), random_state=rng) if len(g) > 0 else g)
    if isinstance(sampled.index, pd.MultiIndex):
        sampled = sampled.droplevel(0)
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=rng)
    return sampled


def plot_encoder_scatter(
    trainer: Any = None,
    encoder_df: Optional[pd.DataFrame] = None,
    *,
    color_by: str = "label",
    hover_cols: Optional[List[str]] = None,
    x_col: Optional[str] = None,
    label_name_mapping: Optional[dict] = None,
    max_points: Optional[int] = None,
    stratify_by: Optional[str] = None,
    cmap: str = DEFAULT_SCATTER_CMAP,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show: bool = True,
    title: Union[str, bool, None] = True,
    height: int = 500,
    split: str = "test",
    hover_text_max_chars: int = 180,
    hover_text_line_chars: int = 90,
    highlight_non_convergence: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """
    Interactive 1D encoder scatter: x = target/category, y = encoded value, color by label, with hover.

    For 1D encoding only. Use in Jupyter for outlier inspection (hover shows point data).
    Colors use the same cmap as encoder distribution violins (default tab20).

    Data source: encoder_df, or trainer (calls trainer.analyze_encoder with split and **kwargs).

    Args:
        trainer: Trainer with analyze_encoder(); used when encoder_df is None.
        encoder_df: Pre-computed DataFrame with 'encoded' and 'label' (and optional 'text', etc.).
        color_by: Column name for point color (default 'label').
        hover_cols: Columns to show on hover. Default includes display text, target, feature class, label, split, and type.
        x_col: Column for the categorical x-axis. Default prefers target/factual token.
        label_name_mapping: Optional mapping from raw color labels to display names.
        max_points: If set, show at most this many points; subsampling is stratified (see stratify_by).
        stratify_by: Column for stratified subsampling when max_points is set. Default: 'feature_class' if present, else color_by.
        cmap: Matplotlib colormap name for point colors (default 'tab20', same as encoder violins).
        output_path: Path to save HTML (Plotly).
        output_dir: Directory for HTML when output_path and experiment_dir are not set.
        experiment_dir: For resolve_output_path.
        show: Whether to display the figure (e.g. in Jupyter).
        title: Plot title. ``True`` (default) uses a default title; ``None``/``False`` disables it.
        height: Figure height in pixels.
        split: Dataset split when encoder_df is None (passed to analyze_encoder).
        hover_text_max_chars: Max characters for 'text' in hover; truncated around first [MASK] with '...'. Default 50.
        hover_text_line_chars: Approximate line length for hover-text wrapping.
        highlight_non_convergence: When True, append a non-convergence marker to the title for
            non-converged runs. ``None`` uses ``TrainingArguments.highlight_non_convergence``.
        **kwargs: Passed to trainer.analyze_encoder when encoder_df is None.

    Returns:
        Plotly Figure (or path string if not show).
    """
    if encoder_df is None:
        if trainer is None:
            raise ValueError("Provide encoder_df or trainer")
        model = getattr(trainer, "get_model", lambda: None)()
        encoder_df = trainer.analyze_encoder(model, split=split, **kwargs)
    if encoder_df is None or encoder_df.empty:
        logger.warning("No encoder data to plot")
        return None

    if "encoded" not in encoder_df.columns:
        logger.warning("encoder_df has no 'encoded' column")
        return None

    rng = np.random.default_rng(42)
    if max_points is not None and max_points > 0 and len(encoder_df) > max_points:
        stratify_col = stratify_by
        if stratify_col is None:
            if "feature_class" in encoder_df.columns:
                stratify_col = "feature_class"
            elif "feature_class_id" in encoder_df.columns:
                stratify_col = "feature_class_id"
            else:
                stratify_col = color_by if color_by in encoder_df.columns else "label"
        encoder_df = _stratified_subsample(encoder_df, max_points, stratify_col, rng)
        if encoder_df.empty:
            logger.warning("Stratified subsample produced no rows")
            return None

    y_vals = encoder_df["encoded"].map(_encoded_1d)
    if y_vals.isna().all():
        logger.warning("No numeric encoded values (1D) in encoder_df")
        return None
    y = y_vals.tolist()
    n = len(y)

    color_col = color_by if color_by in encoder_df.columns else "label"
    if color_col not in encoder_df.columns:
        color_col = None
    color_vals = encoder_df[color_col].astype(str).tolist() if color_col else None

    resolved_x_col = x_col or _first_existing_column(
        encoder_df,
        ["target_token", "factual_token", "factual", "feature_class", "source_id", "type"],
    )
    if resolved_x_col is None:
        x_vals = ["all"] * n
    else:
        x_vals = encoder_df[resolved_x_col].fillna("").astype(str).tolist()
        x_vals = [value if value else "neutral" for value in x_vals]

    text_col = _first_masked_or_non_empty_column(encoder_df, TEXT_HOVER_COLUMNS)
    default_hover = [
        "target_token",
        "factual_token",
        "factual",
        "feature_class",
        "label",
        "data_split",
        "source_id",
        "target_id",
    ]
    if hover_cols is None:
        hover_cols = _ordered_existing_columns(encoder_df, default_hover)
    else:
        hover_cols = [c for c in hover_cols if c in encoder_df.columns and c != "type"]

    try:
        import plotly.express as px
    except ImportError:
        logger.warning("plotly not installed; install with pip install plotly")
        return None

    df_plot = pd.DataFrame({"target": x_vals, "encoded": y})
    if text_col is not None:
        df_plot["text"] = [
            _truncate_text_around_mask(v, hover_text_max_chars, line_chars=hover_text_line_chars)
            for v in encoder_df[text_col].tolist()
        ]
    for c in hover_cols:
        if c != "encoded" and not _is_text_hover_column(c):
            vals = encoder_df[c].tolist()
            df_plot[c] = vals
    hover_cols_resolved = []
    if "text" in df_plot.columns:
        hover_cols_resolved.append("text")
    hover_cols_resolved.extend([c for c in hover_cols if c in df_plot.columns and c != "text"])
    hover_data = hover_cols_resolved or None
    plotly_labels = plotly_labels_for(["target", "encoded", "color", *(hover_cols_resolved or [])])

    highlight = resolve_highlight_non_convergence(highlight_non_convergence, trainer=trainer)
    default_title = "Encoded values (1D) by label" if color_col else "Encoded values (1D)"
    plot_title = resolve_plot_title_with_convergence(
        title,
        trainer=trainer,
        highlight_non_convergence=highlight,
        default=default_title,
    )
    plotly_title = str(plot_title) if plot_title else None

    if color_col:
        df_plot["color"] = [
            str((label_name_mapping or {}).get(value, value))
            for value in color_vals
        ]
        unique_cats = _natural_category_order(df_plot["color"].unique().tolist())
        color_map = _palette_for_categories(unique_cats, cmap=cmap)
        fig = px.scatter(
            df_plot,
            x="target",
            y="encoded",
            color="color",
            title=plotly_title,
            height=height,
            hover_data=hover_data,
            labels=plotly_labels,
            category_orders={"color": unique_cats, "target": _natural_category_order(df_plot["target"].tolist())},
            color_discrete_map=color_map if color_map else None,
        )
        fig.update_traces(marker={"opacity": 0.78})
        fig.update_layout(
            xaxis_title=format_plotly_label("target"),
            yaxis_title=format_plotly_label("encoded"),
            legend_title_text=format_plotly_label(color_col),
        )
    else:
        fig = px.scatter(
            df_plot,
            x="target",
            y="encoded",
            title=plotly_title,
            height=height,
            hover_data=hover_data,
            labels=plotly_labels,
            category_orders={"target": _natural_category_order(df_plot["target"].tolist())},
        )
        fig.update_traces(marker={"opacity": 0.78})
        fig.update_layout(
            xaxis_title=format_plotly_label("target"),
            yaxis_title=format_plotly_label("encoded"),
        )

    if output_path:
        out = output_path
    else:
        run_id = getattr(trainer, "run_id", None) if trainer is not None else None
        out = resolve_output_path(
            experiment_dir or (getattr(trainer, "experiment_dir", None) if trainer is not None else None),
            None,
            ARTIFACT_ENCODER_PLOT,
            run_id=run_id,
        )
        if out is not None:
            base, _ = os.path.splitext(out)
            out = base + "_scatter.html"
        elif output_dir:
            out = os.path.join(output_dir, "encoder_scatter.html")
        else:
            out = None
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.write_html(out)
        logger.info("Saved encoder scatter: %s", out)

    if show:
        fig.show()
    return fig
