"""Persistence helpers for text-prediction data creation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import pandas as pd

from gradiend.data.text.prediction.formats import _to_merged
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _save_data(
    path: str,
    fmt: Literal["csv", "parquet", "hf"],
    df: Optional[pd.DataFrame] = None,
    class_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Save data to path. For training with per_class and fmt hf, pass class_dfs to save as DatasetDict (subsets)."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        data = df if df is not None else (_to_merged(class_dfs) if class_dfs else None)
        if data is not None:
            data.to_csv(path, index=False)
        logger.info("Wrote data to %s (csv)", path)
        return
    if fmt == "parquet":
        data = df if df is not None else (_to_merged(class_dfs) if class_dfs else None)
        if data is not None:
            data.to_parquet(path, index=False)
        logger.info("Wrote data to %s (parquet)", path)
        return
    if fmt == "hf":
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            logger.warning(
                "output_format='hf' requires 'datasets'. Install with: pip install datasets. Falling back to csv."
            )
            ext = path_obj.suffix.lower()
            if not ext or path_obj.suffix == "":
                path = str(path_obj.with_suffix(".csv"))
            _save_data(path, "csv", df=df, class_dfs=class_dfs)
            return
        if class_dfs:
            d = DatasetDict(
                {cid: Dataset.from_pandas(df_c, preserve_index=False) for cid, df_c in class_dfs.items()}
            )
            d.save_to_disk(path)
            logger.info("Wrote data to %s (HuggingFace DatasetDict, subsets per class)", path)
        else:
            data = df if df is not None else None
            if data is not None:
                Dataset.from_pandas(data, preserve_index=False).save_to_disk(path)
                logger.info("Wrote data to %s (HuggingFace Dataset)", path)


def _related_output_path(path: str, suffix: str, fmt: Literal["csv", "parquet", "hf"]) -> str:
    """Return a sibling output path with suffix inserted before the extension."""
    path_obj = Path(path)
    if fmt == "hf":
        return str(path_obj.parent / f"{path_obj.name}_{suffix}")
    if path_obj.suffix:
        return str(path_obj.with_name(f"{path_obj.stem}_{suffix}{path_obj.suffix}"))
    return str(path_obj.with_name(f"{path_obj.name}_{suffix}"))

__all__ = ["_save_data", "_related_output_path"]
