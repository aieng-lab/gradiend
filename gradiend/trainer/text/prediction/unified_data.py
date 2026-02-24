"""
Unified data contract for text prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal

from gradiend.util.logging import get_logger
from gradiend.util import normalize_split_name

from gradiend.trainer.text.prediction.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
    UNIFIED_TRANSITION,
    transition_id,
)

logger = get_logger(__name__)

# Optional column for tracking raw class id after merge (for analysis)
RAW_LABEL_CLASS = "raw_label_class"
RAW_ALTERNATIVE_CLASS = "raw_alternative_class"


def apply_factual_casing(factual: str, alternative: str) -> str:
    """Apply the factual token's casing pattern to the alternative token.

    Counterfactual target tokens are compared case-insensitively; the returned
    string uses the factual label's casing: lowercase, title (first upper),
    or all caps. Otherwise the alternative is returned as-is.

    Args:
        factual: The factual token (e.g. "he", "He", "HE").
        alternative: The counterfactual token in any casing.

    Returns:
        alternative with casing applied to match factual's pattern.
    """
    if not factual or not alternative:
        return alternative
    f = factual
    a = alternative
    if f.islower():
        return a.lower()
    if f.isupper():
        return a.upper()
    if len(f) >= 1 and f[0].isupper() and (len(f) == 1 or f[1:].islower()):
        return a.title() if a else a
    return a


def _load_dataframe_from_path(path: Union[str, Path]) -> pd.DataFrame:
    """Load a DataFrame from a local CSV or Parquet file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Data path is not a file: {path}")
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file extension for data: {suf}. Use .csv or .parquet.")


def merge_per_class_dfs(
    class_dfs: Dict[str, pd.DataFrame],
    merge_map: Dict[str, List[str]],
    target_classes: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Merge per-class DataFrames by class_merge_map. Returns {group_id: df}.

    Base classes are grouped into merged keys. Unmapped classes (e.g. 3PL) are
    kept as-is when in target_classes. Each row gets raw_class_id for analysis.
    Adds a column named by the merged key (or keeps existing for unmapped) with
    the factual token so per_class_dict_to_unified can find it.
    """
    base_to_merged: Dict[str, str] = {}
    for merged, bases in merge_map.items():
        for b in bases:
            if b in base_to_merged:
                raise ValueError(
                    f"class_merge_map: base class {b!r} appears in multiple merged classes."
                )
            base_to_merged[b] = merged

    if target_classes and len(target_classes) == 2:
        keys_to_include = set(target_classes)
    else:
        keys_to_include = set(merge_map.keys()) | {
            c for c in class_dfs.keys() if c not in base_to_merged
        }

    result: Dict[str, pd.DataFrame] = {}
    for key in keys_to_include:
        if key in merge_map:
            dfs = []
            for base in merge_map[key]:
                if base not in class_dfs:
                    continue
                df = class_dfs[base].copy()
                df["raw_class_id"] = base
                # Add column named by merged key with factual token (base column has it)
                if base in df.columns:
                    df[key] = df[base]
                elif "label" in df.columns:
                    df[key] = df["label"]
                else:
                    df[key] = df["token"] if "token" in df.columns else df.iloc[:, 2]
                dfs.append(df)
            if dfs:
                result[key] = pd.concat(dfs, ignore_index=True)
        else:
            if key in class_dfs:
                df = class_dfs[key].copy()
                df["raw_class_id"] = key
                result[key] = df
    return result


def apply_class_merge_to_merged_df(
    df: pd.DataFrame,
    merge_map: Dict[str, List[str]],
    label_class_col: str = "label_class",
    target_class_col: str = "alternative_class",
    target_classes: Optional[List[str]] = None,
    keep_raw: bool = True,
    transition_groups: Optional[List[List[str]]] = None,
) -> pd.DataFrame:
    """Map label_class and target_class to merged group IDs; optionally limit base transitions.

    - Optionally filter base-class transitions using transition_groups (clusters of raw
      class ids). Only rows where BOTH raw classes lie in the same cluster (and differ)
      are kept (e.g. [["1SG","1PL"], ["3SG","3PL"]] keeps 1SG↔1PL and 3SG↔3PL).
    - Then map to merged ids via merge_map and drop same-group transitions.

    Use before merged_to_unified when data has base class IDs and class_merge_map
    is set.
    """
    base_to_merged: Dict[str, str] = {}
    for merged, bases in merge_map.items():
        for b in bases:
            if b in base_to_merged:
                raise ValueError(
                    f"class_merge_map: base class {b!r} appears in multiple merged classes."
                )
            base_to_merged[b] = merged

    out = df.copy()
    # Preserve raw base classes for analysis / transition grouping
    if keep_raw:
        out[RAW_LABEL_CLASS] = out[label_class_col].astype(str)
        out[RAW_ALTERNATIVE_CLASS] = out[target_class_col].astype(str)

    # Optional: restrict to transitions within specified base-class clusters
    if transition_groups:
        clusters = [set(g) for g in transition_groups if g]

        def _allowed(row: pd.Series) -> bool:
            src = str(row[label_class_col])
            tgt = str(row[target_class_col])
            if src == tgt:
                return False
            for g in clusters:
                if src in g and tgt in g:
                    return True
            return False

        out = out[out.apply(_allowed, axis=1)].copy()

    # Map base classes to merged ids
    out[label_class_col] = out[label_class_col].astype(str).map(
        lambda x: base_to_merged.get(x, x)
    )
    out[target_class_col] = out[target_class_col].astype(str).map(
        lambda x: base_to_merged.get(x, x)
    )
    # Drop same-group transitions after merge (e.g. singular->singular)
    out = out[out[label_class_col] != out[target_class_col]].copy()
    if target_classes and len(target_classes) == 2:
        pair_set = set(target_classes)
        out = out[
            out[label_class_col].isin(pair_set) & out[target_class_col].isin(pair_set)
        ].copy()
    return out


def resolve_dataframe(
    value: Optional[Union[pd.DataFrame, str, Path]],
    split: str = "train",
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Resolve input to a pandas DataFrame.

    Accepts a DataFrame, a local file path (str or Path to .csv/.parquet),
    a HuggingFace dataset ID (str), or None. For str, treats as local path
    if the path exists as a file, otherwise as HF dataset ID.

    Args:
        value: A pandas DataFrame, path (str or Path), HuggingFace dataset ID, or None.
        split: Dataset split to load when value is a HuggingFace ID.
            Defaults to "train".
        max_rows: Maximum number of rows to keep when loading from HuggingFace.
            If None, returns all rows.

    Returns:
        The DataFrame, or None if value was None.
    """
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, Path):
        return _load_dataframe_from_path(value)
    if isinstance(value, str):
        p = Path(value)
        if p.is_file():
            return _load_dataframe_from_path(p)
        from datasets import load_dataset
        df = load_dataset(value, split=split, trust_remote_code=True).to_pandas()
        if max_rows is not None:
            df = df.head(max_rows)
        return df
    raise TypeError(f"Expected DataFrame, path (str/Path), str (HF id), or None; got {type(value)}")


def all_subsets_to_mlm_df(
    class_dfs: Dict[str, pd.DataFrame],
    split: str = "train",
    masked_col: str = "masked",
    split_col: str = "split",
    use_class_names_as_columns: bool = True,
) -> pd.DataFrame:
    """Convert per-class DataFrames into a single MLM training DataFrame.

    Extracts masked sentences and their factual labels from each class's
    DataFrame for the given split, producing a two-column DataFrame
    (masked, label) suitable for masked language model training.

    Args:
        class_dfs: Mapping of class name to DataFrame. Each DataFrame must
            have a masked column and a factual/label column (per class).
        split: Split name to filter rows by (e.g., "train", "validation").
        masked_col: Column name for masked sentences.
        split_col: Column name indicating dataset split.
        use_class_names_as_columns: If True, prefer class names as factual
            column names when present.

    Returns:
        DataFrame with columns "masked" and "label".

    Raises:
        ValueError: If a class DataFrame lacks the masked column, lacks a
            factual column, or if no rows are produced for the split.
    """
    sn = normalize_split_name(split)
    rows = []

    def col_for_class(c: str, df: pd.DataFrame) -> Optional[str]:
        if use_class_names_as_columns and c in df.columns:
            return c
        if c in df.columns:
            return c
        return None

    for class_name, df in class_dfs.items():
        df = df.copy()
        if split_col not in df.columns:
            df[split_col] = "train"
        if masked_col not in df.columns:
            raise ValueError(f"Per-class data for '{class_name}' must have column '{masked_col}'")
        factual_col = col_for_class(class_name, df)
        if factual_col is None:
            for c in ["label", "token", class_name]:
                if c in df.columns:
                    factual_col = c
                    break
            if factual_col is None:
                raise ValueError(f"Per-class data for '{class_name}' must have a factual column. Columns: {list(df.columns)}")
        df[split_col] = df[split_col].astype(str).str.lower().apply(normalize_split_name)
        split_mask = df[split_col] == sn
        sub = df.loc[split_mask, [masked_col, factual_col]].dropna(subset=[factual_col])
        for _, row in sub.iterrows():
            rows.append({"masked": str(row[masked_col]), "label": str(row[factual_col])})

    if not rows:
        raise ValueError(f"all_subsets_to_mlm_df produced no rows for split '{split}'.")
    out = pd.DataFrame(rows)
    logger.info("MLM training data: %s rows from %s classes", len(out), len(class_dfs))
    return out


# Merged output column names (match TextPredictionConfig)
MERGED_LABEL_CLASS = "label_class"
MERGED_LABEL = "label"
MERGED_ALTERNATIVE_CLASS = "alternative_class"
MERGED_ALTERNATIVE = "alternative"


def _counterfactual_distribution(series: pd.Series):
    """Build case-insensitive token distribution for weighted sampling.

    Returns (unique_lower, weights, lower_to_canonical). Tokens are considered
    equal when compared case-insensitively; canonical form is the first
    occurrence per lower.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return [], np.array([]), {}
    lower = s.str.lower()
    counts = lower.value_counts()
    unique_lower = counts.index.tolist()
    weights = counts.values.astype(np.float64)
    weights = weights / weights.sum()
    lower_to_canonical = {}
    for _idx, token in s.items():
        lo = str(token).lower()
        if lo not in lower_to_canonical:
            lower_to_canonical[lo] = str(token)
    return unique_lower, weights, lower_to_canonical


def per_class_dict_to_unified(
    class_dfs: Dict[str, pd.DataFrame],
    classes: Union[List[str], Literal["all"]] = "all",
    masked_col: str = "masked",
    split_col: str = "split",
    use_class_names_as_columns: bool = True,
    pair: Optional[Tuple[str, str]] = None,
    include_identity_rows: bool = False,
    max_counterfactuals_per_sentence: int = 1,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Convert per-class DataFrames into unified schema (unified columns directly).

    When a DataFrame has standard "source" and "target" columns (pre-paired
    factual/alternative data), emits one unified row per row using source_id/
    target_id for classes when present; alternative casing follows factual.
    Otherwise uses same-row columns when the df has a column for another class,
    or pair + other_df to derive the alternative token. When deriving from
    other_df, counterfactual tokens are treated case-insensitively and sampled
    by weighted distribution; up to max_counterfactuals_per_sentence unique
    counterfactuals per base sentence are emitted; returned alternative string
    has factual casing applied.

    Args:
        class_dfs: Mapping of class name to DataFrame. Each must have
            masked column and factual/label column.
        classes: Class names to include, or "all" for all keys in class_dfs.
        masked_col: Column name for masked sentences.
        split_col: Column name for dataset split.
        use_class_names_as_columns: Prefer class names as factual columns.
            Fallback: label, token, source, target (source/target are standard).
        pair: Optional (class_a, class_b) for deriving alternative from other_df
            when the current df has no column for target_class.
        include_identity_rows: If True, add identity rows (factual_class =
            alternative_class, same token).
        max_counterfactuals_per_sentence: When deriving alternative from
            other_df, max number of unique (case-insensitive) counterfactual
            tokens per base sentence; must be unique (default 1).
        random_state: Seed for reproducible weighted sampling of counterfactuals.

    Returns:
        DataFrame with unified columns: UNIFIED_MASKED, UNIFIED_SPLIT,
        UNIFIED_FACTUAL_CLASS, UNIFIED_ALTERNATIVE_CLASS, UNIFIED_FACTUAL,
        UNIFIED_ALTERNATIVE, UNIFIED_TRANSITION.
    """

    def col_for_class(c: str, df: pd.DataFrame) -> Optional[str]:
        if use_class_names_as_columns and c in df.columns:
            return c
        if c in df.columns:
            return c
        return None

    rng = np.random.default_rng(random_state)
    rows = []
    classes_list = list(class_dfs.keys()) if classes == "all" else list(classes)
    for source_class, df in class_dfs.items():
        if source_class not in classes_list:
            continue
        df = df.copy()
        if split_col not in df.columns:
            df[split_col] = "train"
        if masked_col not in df.columns:
            raise ValueError(f"Per-class data for '{source_class}' must have column '{masked_col}'")
        factual_col = col_for_class(source_class, df)
        if factual_col is None:
            for c in ["label", "token", "source", "target"]:
                if c in df.columns:
                    factual_col = c
                    break
            if factual_col is None:
                raise ValueError(f"Per-class data for '{source_class}' must have a factual column. Columns: {list(df.columns)}")

        # Standard source/target: pre-paired data, one unified row per row; apply factual casing to target
        if "source" in df.columns and "target" in df.columns:
            for _, row in df.iterrows():
                src_val = row.get("source")
                tgt_val = row.get("target")
                if pd.isna(src_val) or pd.isna(tgt_val):
                    continue
                spl = row.get(split_col, "train")
                src_cls = str(row["source_id"]) if "source_id" in df.columns and pd.notna(row.get("source_id")) else source_class
                tgt_cls = str(row["target_id"]) if "target_id" in df.columns and pd.notna(row.get("target_id")) else (source_class.split("_to_")[-1] if "_to_" in source_class else source_class)
                rows.append({
                    UNIFIED_MASKED: row[masked_col],
                    UNIFIED_SPLIT: normalize_split_name(str(spl)),
                    UNIFIED_FACTUAL_CLASS: src_cls,
                    UNIFIED_ALTERNATIVE_CLASS: tgt_cls,
                    UNIFIED_FACTUAL: str(src_val),
                    UNIFIED_ALTERNATIVE: apply_factual_casing(str(src_val), str(tgt_val)),
                    UNIFIED_TRANSITION: transition_id(src_cls, tgt_cls),
                })
            continue

        other_classes = [c for c in classes_list if c != source_class]
        for target_class in other_classes:
            target_col = col_for_class(target_class, df)
            if target_col is not None:
                for _, row in df.iterrows():
                    factual_val = row[factual_col]
                    target_val = row[target_col]
                    if pd.isna(target_val):
                        continue
                    spl = row.get(split_col, "train")
                    factual_str = str(factual_val)
                    rows.append({
                        UNIFIED_MASKED: row[masked_col],
                        UNIFIED_SPLIT: normalize_split_name(str(spl)),
                        UNIFIED_FACTUAL_CLASS: source_class,
                        UNIFIED_ALTERNATIVE_CLASS: target_class,
                        UNIFIED_FACTUAL: factual_str,
                        UNIFIED_ALTERNATIVE: apply_factual_casing(factual_str, str(target_val)),
                        UNIFIED_TRANSITION: transition_id(source_class, target_class),
                    })
            else:
                # Current df has no column for target_class; get alternative from other class df
                other_df = class_dfs.get(target_class)
                if other_df is None:
                    continue
                if pair is not None and (source_class, target_class) != pair and (target_class, source_class) != pair:
                    continue
                other_factual_col = col_for_class(target_class, other_df) or (target_class if target_class in other_df.columns else "label")
                if other_factual_col not in other_df.columns:
                    continue
                unique_lower, weights, lower_to_canonical = _counterfactual_distribution(other_df[other_factual_col])
                if not unique_lower:
                    continue
                k = min(max_counterfactuals_per_sentence, len(unique_lower))
                for _, row in df.iterrows():
                    factual_str = str(row[factual_col])
                    spl = row.get(split_col, "train")
                    if k <= 0:
                        continue
                    try:
                        chosen_lower = rng.choice(unique_lower, size=k, replace=False, p=weights)
                    except Exception:
                        chosen_lower = rng.choice(unique_lower, size=k, replace=True, p=weights)
                    seen: set = set()
                    for low in chosen_lower:
                        if low in seen:
                            continue
                        seen.add(low)
                        canonical = lower_to_canonical.get(low, low)
                        alt_str = apply_factual_casing(factual_str, canonical)
                        rows.append({
                            UNIFIED_MASKED: row[masked_col],
                            UNIFIED_SPLIT: normalize_split_name(str(spl)),
                            UNIFIED_FACTUAL_CLASS: source_class,
                            UNIFIED_ALTERNATIVE_CLASS: target_class,
                            UNIFIED_FACTUAL: factual_str,
                            UNIFIED_ALTERNATIVE: alt_str,
                            UNIFIED_TRANSITION: transition_id(source_class, target_class),
                        })

        if include_identity_rows:
            for _, row in df.iterrows():
                factual_val = row[factual_col]
                if pd.isna(factual_val):
                    continue
                spl = row.get(split_col, "train")
                factual_str = str(factual_val)
                rows.append({
                    UNIFIED_MASKED: row[masked_col],
                    UNIFIED_SPLIT: normalize_split_name(str(spl)),
                    UNIFIED_FACTUAL_CLASS: source_class,
                    UNIFIED_ALTERNATIVE_CLASS: source_class,
                    UNIFIED_FACTUAL: factual_str,
                    UNIFIED_ALTERNATIVE: factual_str,
                    UNIFIED_TRANSITION: transition_id(source_class, source_class),
                })

    if not rows:
        raise ValueError("per_class_dict_to_unified produced no rows.")
    out = pd.DataFrame(rows)
    logger.debug("Unified data: %s rows from %s classes", len(out), len(class_dfs))
    return out


def merged_to_unified(
    df: pd.DataFrame,
    masked_col: str = "masked",
    split_col: str = "split",
    label_class_col: str = "label_class",
    label_col: str = "label",
    target_col: Optional[str] = None,
    target_class_col: Optional[str] = None,
    pair: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Convert a merged DataFrame into the unified schema.

    Handles two modes: (1) explicit target columns (target_col, target_class_col)
    provide factual and alternative tokens per row; (2) when those are absent,
    pair (class_a, class_b) is used to derive alternative tokens from the other
    class in the pair.

    Args:
        df: Merged DataFrame with masked sentences and label information.
        masked_col: Column name for masked sentences.
        split_col: Column name for dataset split.
        label_class_col: Column with the factual (source) class per row.
        label_col: Column with the factual token per row.
        target_col: Column with alternative token (used with target_class_col).
        target_class_col: Column with alternative class (used with target_col).
        pair: (class_a, class_b) for single-token mode when target columns
            are not provided. Alternative token is taken from the other class.

    Returns:
        DataFrame with unified columns: masked, split, factual_class,
        alternative_class, factual, alternative, transition.

    Raises:
        ValueError: If required columns are missing, or if pair is required
            but not provided or missing tokens for one class.
    """
    df = df.copy()
    if split_col not in df.columns:
        df[split_col] = "train"
    required = {masked_col, split_col, label_class_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Merged data missing columns: {missing}")

    has_explicit_target = target_col and target_class_col and target_col in df.columns and target_class_col in df.columns

    if has_explicit_target:
        rows = []
        for _, row in df.iterrows():
            src = row[label_class_col]
            tgt = row[target_class_col]
            if pd.isna(tgt) or pd.isna(row[target_col]):
                continue
            factual_str = str(row[label_col])
            alternative_str = str(row[target_col])
            rows.append({
                UNIFIED_MASKED: row[masked_col],
                UNIFIED_SPLIT: normalize_split_name(str(row[split_col])),
                UNIFIED_FACTUAL_CLASS: str(src),
                UNIFIED_ALTERNATIVE_CLASS: str(tgt),
                UNIFIED_FACTUAL: factual_str,
                UNIFIED_ALTERNATIVE: apply_factual_casing(factual_str, alternative_str),
                UNIFIED_TRANSITION: transition_id(str(src), str(tgt)),
            })
        return pd.DataFrame(rows)

    if pair is None:
        raise ValueError("Merged data without target_col/target_class_col requires pair.")
    c1, c2 = pair
    by_class = {}
    for cls in [c1, c2]:
        sub = df[df[label_class_col].astype(str) == str(cls)]
        if len(sub) > 0:
            tokens = sub[label_col].dropna().unique().tolist()
            by_class[cls] = str(tokens[0]) if tokens else None
    if by_class.get(c1) is None or by_class.get(c2) is None:
        raise ValueError(f"Merged single-token mode: missing token for pair {pair}.")
    rows = []
    for _, row in df.iterrows():
        src = str(row[label_class_col])
        if src not in pair:
            continue
        tgt = c2 if src == c1 else c1
        factual_str = str(row[label_col])
        alternative_raw = by_class[tgt]
        rows.append({
            UNIFIED_MASKED: row[masked_col],
            UNIFIED_SPLIT: normalize_split_name(str(row[split_col])),
            UNIFIED_FACTUAL_CLASS: src,
            UNIFIED_ALTERNATIVE_CLASS: tgt,
            UNIFIED_FACTUAL: factual_str,
            UNIFIED_ALTERNATIVE: apply_factual_casing(factual_str, alternative_raw),
            UNIFIED_TRANSITION: transition_id(src, tgt),
        })
    return pd.DataFrame(rows)


def load_hf_per_class(
    dataset_name: str,
    classes: Union[List[str], Literal["all"]] = "all",
    splits: Optional[List[str]] = None,
    masked_col: str = "masked",
    split_col: str = "split",
) -> Dict[str, pd.DataFrame]:
    """Load a HuggingFace dataset with per-class (per-config) subsets.

    Each config/subset of the dataset is loaded as a separate DataFrame.
    If classes is "all", uses get_dataset_config_names to discover all
    configs. Optionally filters to specific splits.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "organization/dataset").
        classes: Config/subset names to load, or "all" for all configs.
        splits: Optional list of split names to include. If None, includes
            all splits.
        masked_col: Expected column name for masked sentences; validated
            per subset.
        split_col: Column to populate with split name when missing.

    Returns:
        Mapping of class/config name to DataFrame. Each DataFrame includes
        all requested splits concatenated, with split_col populated.

    Raises:
        ImportError: If the datasets library is not installed.
        ValueError: If a subset has no configs, missing masked_col, or
            no splits match the requested list.
    """
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError as e:
        raise ImportError("HuggingFace datasets required. pip install datasets") from e

    if classes == "all":
        classes = get_dataset_config_names(dataset_name)
        if not classes:
            raise ValueError(f"Dataset {dataset_name} has no configs/subsets.")
        logger.info(f"Loading all {len(classes)} subsets from {dataset_name}: {classes}")

    normalized_splits = [normalize_split_name(s) for s in splits] if splits else None
    result: Dict[str, pd.DataFrame] = {}

    for class_name in classes:
        try:
            ds = load_dataset(dataset_name, class_name, trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Could not load subset '{class_name}' from {dataset_name}: {e}") from e

        dfs = []
        if hasattr(ds, "items"):
            for split_name, split_ds in ds.items():
                sn = normalize_split_name(split_name)
                if normalized_splits is not None and sn not in normalized_splits:
                    continue
                df = split_ds.to_pandas()
                if split_col not in df.columns:
                    df[split_col] = sn
                dfs.append(df)
        else:
            df = ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(ds)
            if split_col not in df.columns:
                df[split_col] = "train"
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No splits loaded for subset '{class_name}' (requested: {splits})")
        out = pd.concat(dfs, ignore_index=True)
        if masked_col not in out.columns:
            raise ValueError(f"Subset '{class_name}' missing column '{masked_col}'.")
        result[class_name] = out
        logger.debug("Loaded subset '%s': %s rows", class_name, len(out))
    return result
