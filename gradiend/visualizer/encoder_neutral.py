"""Neutral encoder row metadata and multi-split plot conventions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from gradiend.util.encoder_splits import order_split_names

NEUTRAL_ENCODER_TYPES: Tuple[str, ...] = ("neutral_training_masked", "neutral_dataset")

DEFAULT_NEUTRAL_DATA_SPLIT = "test"

NEUTRAL_TYPE_VIOLIN_LABELS: Dict[str, str] = {
    "neutral_training_masked": "Neutral (training masked)",
    "neutral_dataset": "Neutral (dataset)",
}

NEUTRAL_TYPE_HUE_SUFFIXES: Dict[str, str] = {
    "neutral_training_masked": "training masked",
    "neutral_dataset": "dataset",
}


def neutral_encoder_row_metadata(
    neutral_type: str,
    *,
    data_split: str = DEFAULT_NEUTRAL_DATA_SPLIT,
) -> Dict[str, str]:
    """Metadata fields to attach when encoding neutral encoder rows.

    Args:
        neutral_type: Neutral encoder row type.
        data_split: Split label assigned to neutral rows.
    """
    if neutral_type not in NEUTRAL_TYPE_VIOLIN_LABELS:
        raise ValueError(
            f"Unknown neutral encoder type {neutral_type!r}. "
            f"Expected one of: {list(NEUTRAL_TYPE_VIOLIN_LABELS)}"
        )
    return {
        "data_split": data_split,
        "neutral_variant": neutral_type,
    }


def neutral_hue_label(neutral_type: str, *, data_split: str = DEFAULT_NEUTRAL_DATA_SPLIT) -> str:
    """Legend label for a neutral variant in dodge/overlay multi-split plots.

    Args:
        neutral_type: Neutral encoder row type.
        data_split: Split label assigned to neutral rows.
    """
    suffix = NEUTRAL_TYPE_HUE_SUFFIXES[neutral_type]
    return f"{data_split} — {suffix}"


def append_multi_split_neutral_frames(
    df_all: pd.DataFrame,
    *,
    neutral_data_split: str = DEFAULT_NEUTRAL_DATA_SPLIT,
) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
    """Build plot frames for each present neutral encoder type.

    Args:
        df_all: Encoder analysis DataFrame containing training and neutral rows.
        neutral_data_split: Split label assigned to neutral rows.

    Returns:
        frames: DataFrames with violin_group, data_split, plot_hue, neutral_variant.
        violin_groups: x-axis group labels in canonical order.
        neutral_hue_labels: dodge/overlay hue labels for neutral rows only.
    """
    frames: List[pd.DataFrame] = []
    violin_groups: List[str] = []
    neutral_hue_labels: List[str] = []

    for neutral_type in NEUTRAL_ENCODER_TYPES:
        sub = df_all[df_all["type"] == neutral_type].copy()
        if sub.empty:
            continue
        violin_label = NEUTRAL_TYPE_VIOLIN_LABELS[neutral_type]
        hue_label = neutral_hue_label(neutral_type, data_split=neutral_data_split)
        sub["violin_group"] = violin_label
        sub["data_split"] = neutral_data_split
        sub["neutral_variant"] = neutral_type
        sub["plot_hue"] = hue_label
        frames.append(sub)
        violin_groups.append(violin_label)
        neutral_hue_labels.append(hue_label)

    return frames, violin_groups, neutral_hue_labels


def build_multi_split_encoder_plot_frame(
    df_train: pd.DataFrame,
    df_all: pd.DataFrame,
    *,
    neutral_data_split: str = DEFAULT_NEUTRAL_DATA_SPLIT,
    include_neutral: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], bool]:
    """Combine training + optional neutral rows for multi-split encoder violins.

    Training rows use ``plot_hue == data_split``. Each neutral encoder type keeps a
    distinct x-axis violin group while sharing the test (or configured) split bucket.

    Args:
        df_train: Training-row subset of the encoder analysis DataFrame.
        df_all: Full encoder analysis DataFrame, including possible neutral rows.
        neutral_data_split: Split label assigned to neutral rows.
        include_neutral: Whether neutral rows are included in the returned frame.

    Returns:
        df_plot, group_order, facet_split_order, dodge_hue_order, includes_neutral_groups
    """
    if df_train.empty:
        raise ValueError("Encoder plot has no training rows for data_split comparison.")

    df_train = df_train.copy()
    df_train["data_split"] = df_train["data_split"].astype(str)
    df_train["plot_hue"] = df_train["data_split"]

    training_group_order = sorted(df_train["violin_group"].dropna().unique().tolist())
    facet_split_order = order_split_names(df_train["data_split"].dropna().astype(str).unique().tolist())

    if include_neutral:
        neutral_frames, neutral_groups, neutral_hue_labels = append_multi_split_neutral_frames(
            df_all,
            neutral_data_split=neutral_data_split,
        )
    else:
        neutral_frames, neutral_groups, neutral_hue_labels = [], [], []

    plot_frames = [df_train, *neutral_frames]
    df_plot = pd.concat(plot_frames, ignore_index=True)

    group_order = training_group_order + neutral_groups
    dodge_hue_order = facet_split_order + neutral_hue_labels
    includes_neutral_groups = bool(neutral_groups)
    return df_plot, group_order, facet_split_order, dodge_hue_order, includes_neutral_groups


def encoder_plot_xlabel(*, includes_neutral_groups: bool) -> str:
    """X-axis label for multi-split encoder violins / strip plots.

    Args:
        includes_neutral_groups: Whether neutral groups are present on the x-axis.
    """
    return "Target" if includes_neutral_groups else "Feature class"
