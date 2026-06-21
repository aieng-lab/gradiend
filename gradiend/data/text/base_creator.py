"""
TextDataCreator: base class for text modality data creators.

Provides modality-independent logic: _get_data(), _resolve_output_path(), _load_cached(),
and (via LabelDataCreator) a generalized generate_training_data flow using _label(item) -> dict | None.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from gradiend.data.core import split_dataframe
from gradiend.data.core.base_loader import resolve_base_data
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _save_dataframe(
    path: str,
    fmt: Literal["csv", "parquet"],
    df: pd.DataFrame,
) -> None:
    """Save DataFrame to path. Modality-independent."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    logger.info("Wrote data to %s (%s)", path, fmt)


class TextDataCreator(ABC):
    """Base for text data creators. Shared: _get_data, _resolve_output_path, _load_cached."""

    def __init__(
        self,
        base_data: Union[str, pd.DataFrame, List[str]],
        *,
        text_column: str = "text",
        base_max_size: Optional[int] = None,
        split: str = "train",
        hf_config: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        seed: int = 42,
        output_dir: Optional[str] = None,
        training_basename: str = "training",
        neutral_basename: str = "neutral",
        output_format: Literal["csv", "parquet"] = "csv",
        use_cache: bool = False,
    ) -> None:
        self.base_data = base_data
        self.text_column = text_column
        self.base_max_size = base_max_size
        self.split = split
        self.hf_config = hf_config
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else None
        self.training_basename = training_basename
        self.neutral_basename = neutral_basename
        self.output_format = output_format
        self.use_cache = use_cache
        self._data_cache: Optional[List[str]] = None
        self._last_generation_interrupted = False

    def _get_data(
        self,
        base_override: Optional[Union[str, pd.DataFrame, List[str]]] = None,
    ) -> List[str]:
        """Load base data as list of strings. Modality-independent for text."""
        def _resolve(src: Union[str, pd.DataFrame, List[str]]) -> List[str]:
            is_hf = isinstance(src, str) and not Path(src).suffix.lower().endswith(".csv")
            return resolve_base_data(
                src,
                text_column=self.text_column,
                max_size=self.base_max_size,
                split=self.split,
                seed=self.seed,
                hf_config=self.hf_config if is_hf else None,
                trust_remote_code=self.trust_remote_code if is_hf else None,
            )

        if base_override is not None:
            return _resolve(base_override)
        if self._data_cache is not None:
            return self._data_cache
        self._data_cache = _resolve(self.base_data)
        return self._data_cache

    def _get_extension(self) -> str:
        """File extension for output format. Override for formats like 'hf' (no ext)."""
        if self.output_format == "csv":
            return ".csv"
        if self.output_format == "parquet":
            return ".parquet"
        return ""

    def _resolve_output_path(
        self,
        name: Literal["training", "neutral"],
        explicit: Optional[str],
    ) -> Optional[str]:
        """Resolve output path: explicit, or output_dir + basename + extension."""
        if explicit is not None:
            return explicit
        if self.output_dir is None:
            return None
        basename = self.training_basename if name == "training" else self.neutral_basename
        ext = self._get_extension()
        return str(self.output_dir / f"{basename}{ext}")

    def _load_cached(self, path: str) -> Optional[pd.DataFrame]:
        """Load cached DataFrame from path. Override for format-specific logic (e.g. HF)."""
        p = Path(path)
        if not p.exists():
            return None
        if self.output_format == "csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)


class LabelDataCreator(TextDataCreator):
    """
    Base for label-based creators. generate_training_data uses _get_data() and _label(item).
    Subclasses implement _label(item) -> dict | None (full row or None to skip).
    """

    @abstractmethod
    def _label(self, item: Any) -> Optional[Dict[str, Any]]:
        """Produce a row dict from item, or None to skip. Row keys are modality-specific (e.g. text, label)."""
        ...

    def _build_labeled_dataframe(self) -> pd.DataFrame:
        """Build labeled DataFrame from _get_data() and _label(), no max_size or split. For use before balance/split."""
        data = self._get_data()
        rows: List[Dict[str, Any]] = []
        self._last_generation_interrupted = False
        try:
            for item in data:
                row = self._label(item)
                if row is not None:
                    rows.append(row)
        except KeyboardInterrupt:
            self._last_generation_interrupted = True
            logger.warning("Data generation interrupted by user; keeping %s labeled rows collected so far.", len(rows))
        if not rows and self._last_generation_interrupted:
            return pd.DataFrame()
        if not rows:
            raise ValueError("No labeled rows produced; _label returned None for all items.")
        return pd.DataFrame(rows)

    def _build_training_dataframe(
        self,
        max_size: Optional[int],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        split_col: str,
        min_rows_for_split: int,
    ) -> pd.DataFrame:
        """Build labeled DataFrame from _get_data() and _label(). Modality-independent."""
        data = self._get_data()
        rows: List[Dict[str, Any]] = []
        self._last_generation_interrupted = False
        try:
            for item in data:
                row = self._label(item)
                if row is not None:
                    rows.append(row)
        except KeyboardInterrupt:
            self._last_generation_interrupted = True
            logger.warning("Data generation interrupted by user; keeping %s labeled rows collected so far.", len(rows))
        if not rows and self._last_generation_interrupted:
            return pd.DataFrame()
        if not rows:
            raise ValueError("No labeled rows produced; _label returned None for all items.")
        df = pd.DataFrame(rows)
        if max_size is not None and len(df) > max_size:
            df = df.sample(n=max_size, random_state=self.seed).reset_index(drop=True)
        if len(df) >= min_rows_for_split:
            df = split_dataframe(
                df,
                train_ratio,
                val_ratio,
                test_ratio,
                self.seed,
                split_col=split_col,
                min_rows=0,
            )
        else:
            df = df.copy()
            df[split_col] = "train"
        return df
