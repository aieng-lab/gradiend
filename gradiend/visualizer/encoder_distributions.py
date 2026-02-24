"""
Encoder distribution plots: grouped split violins from encoder analysis DataFrame.

Accepts encoder_df directly (self-managed data) or obtains it via trainer.analyze_encoder(...).
Requires matplotlib and seaborn. If missing, raises ImportError with install instructions.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from gradiend.util.paths import resolve_output_path, ARTIFACT_ENCODER_PLOT
from gradiend.visualizer.plot_optional import _require_matplotlib, _require_seaborn
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def plot_encoder_distributions(
    trainer: Any,
    encoder_df: pd.DataFrame = None,
    output: Optional[str] = None,
    output_dir: Optional[str] = None,
    show: bool = True,
    title: Union[str, bool] = True,
    violin_order: Optional[List[str]] = None,
    paired_legend_labels: Optional[List[str]] = None,
    legend_name_mapping: Optional[Dict[str, str]] = None,
    legend_group_mapping: Optional[Dict[str, List[str]]] = None,
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    axis_label_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
    colors: Optional[Dict[str, str]] = None,
    legend_loc: str = "best",
    legend_ncol: int = 2,
    cmap: str = "tab20",
    img_format: str = "pdf",
    dpi: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> str:
    """
    Plot encoder distributions as grouped split violins.

    Requires matplotlib and seaborn. If missing, raises ImportError with install instructions.

    Args:
        trainer: Trainer with id, pair, get_model(), and analyze_encoder()
                 when encoder_df is None.
        encoder_df: Pre-computed encoder analysis DataFrame (columns: encoded, label, source_id,
                    target_id, type, ...). If provided, no call to trainer.analyze_encoder.
        output: Explicit path for saved PDF (overrides experiment_dir / output_dir).
        output_dir: Directory for saved PDF when output and experiment_dir are not set.
        show: If True, call plt.show() to display the plot.
        title: True (default run_id), False, or custom string for the plot title.
        violin_order: Optional ordering of violin groups on the x-axis.
        paired_legend_labels: Optional explicit ordering of raw legend labels to plot. If set,
                    the labels are paired consecutively into split violins: (0,1) -> violin 1,
                    (2,3) -> violin 2, etc. This bypasses the default inference of split halves
                    from source/target ids.
        legend_name_mapping: Optional dict mapping raw legend labels to display names.
        legend_group_mapping: Optional dict mapping a new legend label to a list of existing
                    legend labels (e.g. {"Gender swap": ["masc_nom -> fem_nom",
                    "fem_nom -> masc_nom"]}). Grouped transitions are downsampled
                    to the minimum count within the group so groups are balanced.
        title_fontsize: Font size for the title.
        label_fontsize: Font size for axis tick labels.
        axis_label_fontsize: Font size for axis labels.
        legend_fontsize: Font size for legend text.
        colors: Optional dict mapping legend labels to hex colors.
        legend_loc: Matplotlib legend location (default "best").
        legend_ncol: Number of columns for the legend (default 2).
        cmap: Matplotlib colormap name for palette (default "tab20").
        figsize: Figure size (width, height) in inches. If None, uses (max(6, 1.5 * n_groups), 3).

    Returns:
        Path to saved plot PDF, or "" if nothing to plot or plot was shown only (no save path).
        Raises ValueError only when show=False and no output path can be determined.
    """
    plt = _require_matplotlib()
    sns = _require_seaborn()


    run_id = getattr(trainer, "run_id", None)

    if encoder_df is None:
        raise ValueError(
            "encoder_df is required. Call analyze_encoder(...) first and pass the returned DataFrame."
        )
    df_all = encoder_df
    model_with_gradiend = getattr(trainer, "get_model", lambda: None)()

    if df_all is None or df_all.empty:
        raise ValueError("Encoder plot has no data (encoder_df or analyze_encoder returned empty).")

    source_type = "factual"
    if model_with_gradiend is not None and hasattr(model_with_gradiend, "gradiend"):
        g = getattr(model_with_gradiend.gradiend, "kwargs", None) or {}
        training_config = g.get("training", {}).get("training_args", {})
        source_type = training_config.get("source", "factual")

    training_pair = getattr(trainer, "pair", None)
    training_transitions = set()
    if training_pair and len(training_pair) >= 2:
        training_transitions.add(f"{training_pair[0]} -> {training_pair[1]}")
        training_transitions.add(f"{training_pair[1]} -> {training_pair[0]}")

    plot_rows = []
    df_training = df_all[df_all["type"] == "training"].copy()
    if not df_training.empty:
        df_training["violin_group"] = df_training["source_id"].astype(str)
        df_training["hue_label"] = df_training["target_id"].astype(str)
        targets_per_group = (
            df_training.groupby("violin_group")["hue_label"]
            .apply(lambda s: sorted(s.dropna().unique().tolist()))
            .to_dict()
        )

        def _assign_side(row):
            tgts = targets_per_group.get(row["violin_group"], [])
            if len(tgts) == 2:
                return "Left" if row["hue_label"] == tgts[0] else "Right"
            if len(tgts) == 1:
                return "Left" if row["hue_label"] == tgts[0] else None
            return None

        df_training["side"] = df_training.apply(_assign_side, axis=1)
        if source_type == "alternative":
            df_training["split_label"] = df_training.apply(
                lambda r: f"{r['source_id']} -> {r['target_id']}", axis=1
            )
            df_training["is_training_transition"] = df_training["split_label"].isin(training_transitions)
        else:
            df_training["is_training_transition"] = df_training["source_id"].isin(training_pair or [])
        plot_rows.append(df_training)

    df_neutral_training_masked = df_all[df_all["type"] == "neutral_training_masked"].copy()
    if not df_neutral_training_masked.empty:
        df_neutral_training_masked["violin_group"] = "Neutral"
        df_neutral_training_masked["hue_label"] = "Training masked"
        df_neutral_training_masked["side"] = "Left"
        plot_rows.append(df_neutral_training_masked)
    df_neutral_dataset = df_all[df_all["type"] == "neutral_dataset"].copy()
    if not df_neutral_dataset.empty:
        df_neutral_dataset["violin_group"] = "Neutral"
        df_neutral_dataset["hue_label"] = "Neutral dataset"
        # When "Neutral: Training masked" is missing, put Neutral dataset on the left so the right half stays empty.
        has_training_masked = not df_neutral_training_masked.empty
        df_neutral_dataset["side"] = "Right" if has_training_masked else "Left"
        plot_rows.append(df_neutral_dataset)

    if not plot_rows:
        raise ValueError("Encoder plot has no plottable rows (no training or neutral data).")

    def _legend_label(g: str, label: str) -> str:
        # Pass-through when already a full legend label (e.g. from paired mode or "X -> Y").
        if isinstance(label, str) and (" -> " in label or label.startswith("Neutral:")):
            return label
        if legend_group_mapping and isinstance(label, str) and label in legend_group_mapping:
            return label
        if g == "Neutral":
            return f"Neutral: {label}"
        return f"{g} -> {label}"

    df_plot = pd.concat(plot_rows, ignore_index=True)
    df_plot["legend_label"] = df_plot.apply(
        lambda r: _legend_label(r["violin_group"], r["hue_label"]), axis=1
    )
    if legend_group_mapping:
        flat_labels = [lbl for labels in legend_group_mapping.values() for lbl in labels]
        present_labels = set(df_plot["legend_label"].dropna().unique().tolist())
        missing_labels = [lbl for lbl in flat_labels if lbl not in present_labels]
        if missing_labels:
            raise ValueError(
                "legend_group_mapping labels not present in data: %s"
                % sorted(set(missing_labels))
            )

        indices_to_keep: set = set()
        for new_label, legend_labels in legend_group_mapping.items():
            subset = df_plot[df_plot["legend_label"].isin(legend_labels)]
            if subset.empty:
                raise ValueError(
                    f"Legend group '{new_label}' has no data (none of its labels are present in the plot)."
                )
            by_label = subset.groupby("legend_label", group_keys=False)
            min_count = by_label.size().min()
            if min_count == 0:
                continue
            sampled = by_label.sample(n=min_count, random_state=42)
            indices_to_keep.update(sampled.index.tolist())
        ungrouped = df_plot[~df_plot["legend_label"].isin(flat_labels)]
        indices_to_keep.update(ungrouped.index.tolist())
        df_plot = df_plot.loc[list(indices_to_keep)].copy()

        label_to_group: Dict[str, str] = {}
        for new_label, legend_labels in legend_group_mapping.items():
            for lbl in legend_labels:
                label_to_group[lbl] = new_label
        df_plot["legend_label"] = df_plot["legend_label"].map(
            lambda v: label_to_group.get(v, v)
        )

    used_paired_mode = True
    if paired_legend_labels:
        order = list(paired_legend_labels)
    else:
        order = df_plot["legend_label"].dropna().unique().tolist()

    if not order:
        raise ValueError("Encoder plot has no data to plot (no legend labels).")

    df_plot = df_plot[df_plot["legend_label"].isin(order)].copy()
    label_to_pair: Dict[str, Tuple[str, str]] = {}
    for i, raw_label in enumerate(order):
        pair_id = str(i // 2)
        side = "Left" if (i % 2 == 0) else "Right"
        label_to_pair[raw_label] = (pair_id, side)

    df_plot["violin_group"] = df_plot["legend_label"].map(lambda s: label_to_pair[str(s)][0])
    df_plot["side"] = df_plot["legend_label"].map(lambda s: label_to_pair[str(s)][1])
    df_plot["hue_label"] = df_plot["legend_label"]

    if used_paired_mode:
        default_group_order = sorted(df_plot["violin_group"].unique().tolist(), key=lambda s: int(str(s)))
    else:
        training_groups = sorted([g for g in df_plot["violin_group"].unique().tolist() if g != "Neutral"])
        has_neutral = "Neutral" in df_plot["violin_group"].values
        default_group_order = training_groups + (["Neutral"] if has_neutral else [])
    group_side_to_label = (
        df_plot.drop_duplicates(["violin_group", "side"])[["violin_group", "side", "hue_label"]]
        .set_index(["violin_group", "side"])["hue_label"]
        .to_dict()
    )
    # Only show legend entries that have data; do not add placeholder entries for missing halves.
    default_half_pairs = []
    for g in default_group_order:
        if (g, "Left") in group_side_to_label:
            default_half_pairs.append((g, "Left"))
        if (g, "Right") in group_side_to_label:
            default_half_pairs.append((g, "Right"))

    if (not used_paired_mode) and violin_order:
        name_to_half = {
            _legend_label(g, group_side_to_label[(g, side)]): (g, side)
            for (g, side) in default_half_pairs
        }
        half_pairs = []
        group_order = []
        for name in violin_order:
            if name in name_to_half:
                g, side = name_to_half[name]
                half_pairs.append((g, side))
                if g not in group_order:
                    group_order.append(g)
        if not half_pairs:
            half_pairs = default_half_pairs
            group_order = default_group_order
    else:
        half_pairs = default_half_pairs
        group_order = default_group_order

    group_to_x_id = {g: i for i, g in enumerate(group_order)}
    df_plot["x_id"] = df_plot["violin_group"].map(group_to_x_id)
    df_plot = df_plot[df_plot["x_id"].notna()].copy()
    df_plot["x_id"] = df_plot["x_id"].astype(int)
    x_id_order = list(range(len(group_order)))
    df_plot["x_cat"] = pd.Categorical(df_plot["x_id"], categories=x_id_order, ordered=True)

    _figsize = figsize if figsize is not None else (max(6, 1.5 * len(group_order)), 3)
    plt.figure(figsize=_figsize)
    ax = sns.violinplot(
        data=df_plot,
        x="x_cat",
        y="encoded",
        hue="side",
        hue_order=["Left", "Right"],
        split=True,
        inner="quartile",
        density_norm="width",
        linewidth=0.7,
        cut=0,
        zorder=5,
    )
    _require_matplotlib()  # Ensure matplotlib is available
    from matplotlib.collections import PolyCollection

    if ax.legend_ is not None:
        ax.legend_.remove()

    legend_label_to_group: Dict[str, str] = {}
    if legend_group_mapping:
        for new_label, labels in legend_group_mapping.items():
            for lbl in labels:
                legend_label_to_group[lbl] = new_label

    def _legend_group_for_half(g: str, side: str) -> str:
        label = group_side_to_label.get((g, side), "")
        raw = _legend_label(g, label)
        return legend_label_to_group.get(raw, raw)

    display_to_idx: Dict[str, int] = {}
    display_to_raw: Dict[str, str] = {}
    unique_displays: List[str] = []
    half_to_display_idx: Dict[Tuple[str, str], int] = {}

    def _display_label(raw: str) -> str:
        mapped = (legend_name_mapping or {}).get(raw)
        if mapped is not None:
            return mapped
        return raw

    for (g, side) in half_pairs:
        legend_group = _legend_group_for_half(g, side)
        display = _display_label(legend_group)
        if display not in display_to_idx:
            display_to_idx[display] = len(unique_displays)
            unique_displays.append(display)
            display_to_raw[display] = legend_group
        half_to_display_idx[(g, side)] = display_to_idx[display]

    n_legend_entries = len(unique_displays)
    try:
        cmap_obj = plt.get_cmap(cmap)
        n_cmap = getattr(cmap_obj, "N", None)
        if n_cmap is not None and n_legend_entries <= n_cmap:
            palette = [cmap_obj(i / n_cmap) for i in range(max(1, n_legend_entries))]
        else:
            palette = [
                cmap_obj(i / max(1, n_legend_entries - 1)) if n_legend_entries > 1 else cmap_obj(0.5)
                for i in range(max(1, n_legend_entries))
            ]
    except (ValueError, TypeError):
        palette = sns.color_palette(cmap, n_colors=max(1, n_legend_entries))

    train_pair_half_set = set()
    if used_paired_mode:
        for (g, side) in half_pairs:
            label = group_side_to_label.get((g, side))
            if label and label in training_transitions:
                train_pair_half_set.add((g, side))
    elif not df_training.empty and "is_training_transition" in df_training.columns:
        for (g, side) in half_pairs:
            if g == "Neutral":
                continue
            label = group_side_to_label.get((g, side))
            if label is None:
                continue
            sub = df_training[(df_training["violin_group"] == g) & (df_training["hue_label"] == label)]
            if len(sub) and sub["is_training_transition"].any():
                train_pair_half_set.add((g, side))

    x_centers = {i: i for i in range(len(group_order))}
    for coll in ax.collections:
        if not isinstance(coll, PolyCollection):
            continue
        paths = coll.get_paths() or []
        if not paths:
            continue
        xs = []
        for p in paths:
            verts = getattr(p, "vertices", None)
            if verts is None or len(verts) == 0:
                continue
            xs.append(sum(v[0] for v in verts) / len(verts))
        if not xs:
            continue
        x_mean = sum(xs) / len(xs)
        x_id = int(round(x_mean))
        if x_id not in x_centers:
            continue
        g = group_order[x_id]
        side = "Left" if x_mean < x_centers[x_id] else "Right"
        if (g, side) not in group_side_to_label:
            continue
        disp_idx = half_to_display_idx.get((g, side), 0)
        face = palette[disp_idx % len(palette)]
        coll.set_facecolor(face)
        coll.set_edgecolor("black")
        coll.set_alpha(1.0)
        lw = 2.5 if (g, side) in train_pair_half_set else 0.7
        coll.set_linewidth(lw)

    if title is False:
        pass
    elif isinstance(title, str):
        plt.title(title, fontsize=title_fontsize)
    elif run_id:
        plt.title(str(run_id), fontsize=title_fontsize)
    ax.set_xticklabels([])
    plt.xlabel("", fontsize=axis_label_fontsize)
    plt.ylabel("Encoded value", fontsize=axis_label_fontsize)
    if label_fontsize is not None:
        ax.tick_params(labelsize=label_fontsize)

    # Import matplotlib.patches locally where it's used
    _require_matplotlib()  # Ensure matplotlib is available before importing patches
    import matplotlib.patches as mpatches
    
    legend_handles = []
    legend_labels = []
    bold_legend_idxs = set()
    for (g, side) in half_pairs:
        if (g, side) in train_pair_half_set:
            bold_legend_idxs.add(half_to_display_idx[(g, side)])
    for i, display in enumerate(unique_displays):
        raw_label = display_to_raw.get(display, display)
        color = (colors or {}).get(display) or (colors or {}).get(raw_label) or palette[i % len(palette)]
        legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="black", label=display))
        legend_labels.append(display)
    legend = ax.legend(legend_handles, legend_labels, loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize)
    for i in bold_legend_idxs:
        try:
            legend.get_texts()[i].set_fontweight("bold")
        except Exception:
            pass

    experiment_dir = getattr(trainer, "experiment_dir", None)
    if callable(experiment_dir):
        experiment_dir = experiment_dir()
    resolved = resolve_output_path(
        experiment_dir, output, ARTIFACT_ENCODER_PLOT,
        run_id=run_id,
    )
    out_path = resolved or (
        os.path.join(output_dir, f"{run_id or 'run'}_encoder_distributions.pdf")
        if output_dir else None
    )
    if out_path and img_format:
        ext = img_format if img_format.startswith(".") else f".{img_format}"
        out_path = os.path.splitext(out_path)[0] + ext
    plt.tight_layout()
    plt.grid(axis="y", alpha=0.3, zorder=0)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        plt.savefig(out_path, **save_kwargs)
        logger.info("Saved encoder distribution plot: %s", out_path)
    elif not show:
        plt.close()
        raise ValueError(
            "No output path or experiment_dir set and show=False. "
            "Set experiment_dir on TrainingArguments, or pass output= / output_dir= to save, or show=True to display only."
        )
    if show:
        plt.show()
    plt.close()
    return out_path or ""
