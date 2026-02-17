"""
Base data loader: resolve from HF, DataFrame, CSV, or list of strings.

Modality-agnostic; used by both text prediction and (future) classification.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def resolve_base_data(
    source: Union[str, pd.DataFrame, List[str]],
    text_column: str = "text",
    max_size: Optional[int] = None,
    split: str = "train",
    seed: int = 42,
    hf_config: Optional[str] = None,
    trust_remote_code: bool = False,
) -> List[str]:
    """Resolve input to a list of text strings.

    Supports HuggingFace dataset ID, pandas DataFrame, CSV path, or list of strings.
    Data is shuffled with `seed` before applying `max_size` to avoid bias from
    ordered sources (e.g. chronological, single-author).

    Args:
        source: HF dataset ID (str), CSV path (str ending in .csv or path exists),
            pandas DataFrame, or List[str].
        text_column: Column name for text (default "text"). Ignored for List[str].
        max_size: Cap on number of items (applied after shuffle). None = no cap.
        split: Dataset split for HF (default "train").
        seed: Random seed for shuffle.
        hf_config: HuggingFace dataset config/subset (e.g. "20220301.en" for wikipedia).
            Only used when source is an HF dataset ID.
        trust_remote_code: Passed to load_dataset when loading from HF. Default False.

    Returns:
        List of strings (texts).
    """
    texts: List[str]
    if isinstance(source, list):
        if not all(isinstance(x, str) for x in source):
            raise TypeError("source as list must contain only strings")
        texts = [str(x).strip() for x in source if str(x).strip()]
    elif isinstance(source, pd.DataFrame):
        if text_column not in source.columns:
            raise ValueError(f"DataFrame missing column '{text_column}'. Columns: {list(source.columns)}")
        texts = source[text_column].dropna().astype(str).str.strip().tolist()
        texts = [x for x in texts if x]
    elif isinstance(source, str):
        texts = _load_from_string_source(
            source, text_column, split, hf_config, trust_remote_code
        )
    else:
        raise TypeError(f"source must be str, DataFrame, or List[str]; got {type(source)}")

    rng = random.Random(seed)
    rng.shuffle(texts)
    if max_size is not None and len(texts) > max_size:
        texts = texts[:max_size]
    logger.debug(f"resolve_base_data: {len(texts)} texts")
    return texts


def _load_from_string_source(
    source: str,
    text_column: str,
    split: str,
    hf_config: Optional[str] = None,
    trust_remote_code: bool = False,
) -> List[str]:
    """Load from HF dataset or CSV path."""
    path = Path(source)
    if path.suffix.lower() == ".csv" and path.exists():
        df = pd.read_csv(source)
        if text_column not in df.columns:
            raise ValueError(f"CSV missing column '{text_column}'. Columns: {list(df.columns)}")
        texts = df[text_column].dropna().astype(str).str.strip().tolist()
        return [x for x in texts if x]

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace datasets required for HF source. pip install datasets") from e

    if hf_config is not None:
        ds = load_dataset(
            source, hf_config, split=split, trust_remote_code=trust_remote_code
        )
    else:
        ds = load_dataset(source, split=split, trust_remote_code=trust_remote_code)
    if hasattr(ds, "to_pandas"):
        df = ds.to_pandas()
    else:
        df = pd.DataFrame(list(ds))
    if text_column not in df.columns:
        raise ValueError(f"HF dataset missing column '{text_column}'. Columns: {list(df.columns)}")
    texts = df[text_column].dropna().astype(str).str.strip().tolist()
    return [x for x in texts if x]
