"""
Interactive 1D encoder scatter: jitter on x-axis, encoded value on y-axis, colored by label, with hover.
For outlier analysis in Jupyter. Uses Plotly for interaction.
Colors match encoder distribution violins (tab20) when cmap is used.
Colormap lookup requires matplotlib; if missing, raises ImportError with install instructions.
"""

import os
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from gradiend.util.paths import resolve_output_path, ARTIFACT_ENCODER_PLOT
from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

# Default cmap for scatter to match encoder_distributions violins
DEFAULT_SCATTER_CMAP = "tab20"


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


def _truncate_text_around_mask(text: str, max_chars: int = 50) -> str:
    """Truncate text to at most max_chars, centered on first [MASK] if present."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    if len(text) <= max_chars:
        return text
    mask = "[MASK]"
    idx = text.find(mask)
    content_max = max(10, max_chars - 6)  # reserve for "..." and "..."
    if idx < 0:
        return "..." + text[-content_max:]
    half = (content_max - len(mask)) // 2
    start = max(0, idx - half)
    end = min(len(text), idx + len(mask) + (content_max - half - len(mask)))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


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
    jitter_scale: float = 0.15,
    max_points: Optional[int] = None,
    stratify_by: Optional[str] = None,
    cmap: str = DEFAULT_SCATTER_CMAP,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
    height: int = 500,
    split: str = "test",
    hover_text_max_chars: int = 50,
    **kwargs: Any,
) -> Any:
    """
    Interactive 1D encoder scatter: x = jitter, y = encoded value, color by label, with hover.

    For 1D encoding only. Use in Jupyter for outlier inspection (hover shows point data).
    Colors use the same cmap as encoder distribution violins (default tab20).

    Data source: encoder_df, or trainer (calls trainer.analyze_encoder with split and **kwargs).

    Args:
        trainer: Trainer with analyze_encoder(); used when encoder_df is None.
        encoder_df: Pre-computed DataFrame with 'encoded' and 'label' (and optional 'text', etc.).
        color_by: Column name for point color (default 'label').
        hover_cols: Columns to show on hover. Default: ['text', 'label', 'encoded', 'source_id', 'target_id', 'type'] (existing cols only).
        jitter_scale: Scale of random jitter on x (default 0.15).
        max_points: If set, show at most this many points; subsampling is stratified (see stratify_by).
        stratify_by: Column for stratified subsampling when max_points is set. Default: 'feature_class' if present, else color_by.
        cmap: Matplotlib colormap name for point colors (default 'tab20', same as encoder violins).
        output_path: Path to save HTML (Plotly).
        output_dir: Directory for HTML when output_path and experiment_dir are not set.
        experiment_dir: For resolve_output_path.
        show: Whether to display the figure (e.g. in Jupyter).
        title: Plot title.
        height: Figure height in pixels.
        split: Dataset split when encoder_df is None (passed to analyze_encoder).
        hover_text_max_chars: Max characters for 'text' in hover; truncated around first [MASK] with '...'. Default 50.
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
    x_jitter = rng.uniform(-jitter_scale, jitter_scale, size=n)

    color_col = color_by if color_by in encoder_df.columns else "label"
    if color_col not in encoder_df.columns:
        color_col = None
    color_vals = encoder_df[color_col].astype(str).tolist() if color_col else None

    default_hover = ["text", "label", "encoded", "source_id", "target_id", "type"]
    if hover_cols is None:
        hover_cols = [c for c in default_hover if c in encoder_df.columns]
    hover_cols = [c for c in hover_cols if c in encoder_df.columns]

    try:
        import plotly.express as px
    except ImportError:
        logger.warning("plotly not installed; install with pip install plotly")
        return None

    df_plot = pd.DataFrame({"x_jitter": x_jitter, "encoded": y})
    for c in hover_cols:
        if c != "encoded":
            vals = encoder_df[c].tolist()
            if c == "text" and hover_text_max_chars > 0:
                vals = [_truncate_text_around_mask(v, hover_text_max_chars) for v in vals]
            df_plot[c] = vals
    hover_data = [c for c in hover_cols if c in df_plot.columns] or None

    if color_col:
        df_plot["color"] = color_vals
        unique_cats = sorted(df_plot["color"].unique(), key=str)
        color_map = _palette_for_categories(unique_cats, cmap=cmap)
        fig = px.scatter(
            df_plot,
            x="x_jitter",
            y="encoded",
            color="color",
            title=title or "Encoded values (1D) by label",
            height=height,
            hover_data=hover_data,
            color_discrete_map=color_map if color_map else None,
        )
        fig.update_layout(xaxis_title="jitter", yaxis_title="encoded value")
    else:
        fig = px.scatter(
            df_plot,
            x="x_jitter",
            y="encoded",
            title=title or "Encoded values (1D)",
            height=height,
            hover_data=hover_data,
        )
        fig.update_layout(xaxis_title="jitter", yaxis_title="encoded value")

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