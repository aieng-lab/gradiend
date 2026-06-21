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

# Hugging Face datasets >= 4 removed trust_remote_code and loading-script datasets.
_DATASETS_V4_OR_NEWER: Optional[bool] = None


def _datasets_v4_or_newer() -> bool:
    global _DATASETS_V4_OR_NEWER
    if _DATASETS_V4_OR_NEWER is None:
        try:
            from datasets import __version__ as ds_version

            _DATASETS_V4_OR_NEWER = int(str(ds_version).split(".", maxsplit=1)[0]) >= 4
        except Exception:
            _DATASETS_V4_OR_NEWER = False
    return _DATASETS_V4_OR_NEWER


def _hf_load_kwargs(
    *,
    split: str,
    trust_remote_code: Optional[bool],
    streaming: bool,
) -> dict:
    kwargs: dict = {"split": split}
    if streaming:
        kwargs["streaming"] = True
    if trust_remote_code is not None and not _datasets_v4_or_newer():
        kwargs["trust_remote_code"] = trust_remote_code
    return kwargs


def resolve_base_data(
    source: Union[str, pd.DataFrame, List[str]],
    text_column: str = "text",
    max_size: Optional[int] = None,
    split: str = "train",
    seed: int = 42,
    hf_config: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
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
        trust_remote_code: Optional value passed to load_dataset when loading from HF.
            None means do not pass the keyword.

    Returns:
        List of strings (texts).
    """
    if not isinstance(text_column, str):
        raise TypeError(f"text_column must be str, got {type(text_column).__name__}")
    if max_size is not None and not isinstance(max_size, int):
        raise TypeError(f"max_size must be int or None, got {type(max_size).__name__}")
    if not isinstance(split, str):
        raise TypeError(f"split must be str, got {type(split).__name__}")
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if hf_config is not None and not isinstance(hf_config, str):
        raise TypeError(f"hf_config must be str or None, got {type(hf_config).__name__}")
    if trust_remote_code is not None and not isinstance(trust_remote_code, bool):
        raise TypeError(f"trust_remote_code must be bool or None, got {type(trust_remote_code).__name__}")

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
            source, text_column, split, hf_config, trust_remote_code, max_size
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
    trust_remote_code: Optional[bool] = None,
    max_size: Optional[int] = None,
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

    use_streaming = max_size is not None
    load_kwargs = _hf_load_kwargs(
        split=split,
        trust_remote_code=trust_remote_code,
        streaming=use_streaming,
    )
    if hf_config is not None:
        ds = load_dataset(source, hf_config, **load_kwargs)
    else:
        ds = load_dataset(source, **load_kwargs)

    if use_streaming:
        texts: List[str] = []
        for row in ds:
            val = row[text_column] if text_column in row else row.get(text_column)
            t = str(val).strip() if val is not None else ""
            if t:
                texts.append(t)
            if len(texts) >= max_size:
                break
        return texts

    if hasattr(ds, "to_pandas"):
        df = ds.to_pandas()
    else:
        df = pd.DataFrame(list(ds))
    if text_column not in df.columns:
        raise ValueError(f"HF dataset missing column '{text_column}'. Columns: {list(df.columns)}")
    texts = df[text_column].dropna().astype(str).str.strip().tolist()
    return [x for x in texts if x]
