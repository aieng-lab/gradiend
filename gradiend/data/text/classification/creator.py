"""
TextClassificationDataCreator: build training and neutral datasets for text classification.

Same interface as TextPredictionDataCreator: generate_training_data and generate_neutral_data.

Implementation notes:

- Uses TextDataCreator for shared caching/output resolution/loading.
- Uses LabelDataCreator for a generalized training-data generation loop:

  _get_data() -> iterable of texts, _label(text) -> row dict or None.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import pandas as pd

from gradiend.data.core import balance_dataframe, cap_dataframe_balanced, split_dataframe
from gradiend.data.text.base_creator import LabelDataCreator, _save_dataframe
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


LabelFnReturn = Union[None, int, str, Dict[str, Any]]


class TextClassificationDataCreator(LabelDataCreator):
    """Create experimental text-classification datasets from base corpora.

    This is a preliminary utility for classification-style experiments. The
    primary, documented GRADIEND data-creation path for v0.2 is still
    :class:`~gradiend.data.text.prediction.creator.TextPredictionDataCreator`.
    Use this class when you explicitly need a label-function based
    classification dataset and are prepared to validate the result for your task.
    """

    def __init__(
        self,
        base_data: Union[str, pd.DataFrame, List[str]],
        label_fn: Callable[[str], LabelFnReturn],
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
        neutral_sentinel: Optional[Union[str, int]] = None,
        neutral_filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """
        Args:
            base_data: HF dataset id, pandas DataFrame, CSV path, or List[str].
            label_fn: Function mapping each string to:

              - a scalar class id/name (int/str), or
              - a dict representing the full row to emit (must include at least 'label' if you want a label), or
              - None to filter out.

            text_column: Column name for text (default "text").
            base_max_size: Cap on base data (after shuffle).
            split: HF split (default "train").
            hf_config: HuggingFace dataset config/subset.
            trust_remote_code: Optional value passed to load_dataset when base_data is HF id.
                None means do not pass the keyword.
            seed: Random seed for shuffle and split.
            output_dir: If set, generate_* methods write here when output= is not passed.
            training_basename: Base name for training output file (default "training").
            neutral_basename: Base name for neutral output file (default "neutral").
            output_format: "csv" (default) or "parquet".
            use_cache: If True and output_dir set, load from existing files when available.
            neutral_sentinel: If set, generate_neutral_data keeps texts where label_fn(text) == neutral_sentinel.
              (Only applies when label_fn returns a scalar, not a dict.)
            neutral_filter_fn: If set, generate_neutral_data keeps only texts where neutral_filter_fn(text) is True.
        """
        if not callable(label_fn):
            raise TypeError(f"label_fn must be callable, got {type(label_fn).__name__}")
        if output_format not in ("csv", "parquet"):
            raise TypeError(f"output_format must be 'csv' or 'parquet', got {output_format!r}")

        super().__init__(
            base_data,
            text_column=text_column,
            base_max_size=base_max_size,
            split=split,
            hf_config=hf_config,
            trust_remote_code=trust_remote_code,
            seed=seed,
            output_dir=output_dir,
            training_basename=training_basename,
            neutral_basename=neutral_basename,
            output_format=output_format,
            use_cache=use_cache,
        )
        self.label_fn = label_fn
        self.neutral_sentinel = neutral_sentinel
        self.neutral_filter_fn = neutral_filter_fn

    def _label(self, item: Any) -> Optional[Dict[str, Any]]:
        text = str(item)
        out = self.label_fn(text)
        if out is None:
            return None
        if isinstance(out, dict):
            row = dict(out)
            row.setdefault("text", text)
            return row
        return {"text": text, "label": out}

    def generate_training_data(
        self,
        max_size: Optional[int] = None,
        output: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_col: str = "split",
        min_rows_for_split: int = 10,
        balance: Union[bool, str] = "try",
        label_col: str = "label",
    ) -> pd.DataFrame:
        """Generate training data by applying label_fn to base texts.

        Args:
            max_size: Cap on number of labeled rows (None = no cap).
            output: If set, save to this path.
            train_ratio, val_ratio, test_ratio: Fractions for split (must sum to 1.0).
            split_col: Name of split column (default "split").
            min_rows_for_split: Minimum rows to perform train/val/test split.
            balance: "try" (default) = oversample minority classes to match the largest;
                "strict" = undersample each class to the smallest class size;
                False = no balancing.
            label_col: Column name for class labels (default "label"), used when balance is not False.

        Returns:
            DataFrame containing at least ``text``, ``label_col``, and
            ``split_col``.

        Raises:
            ValueError: If no labeled rows are produced, there are too few rows
                for splitting, or split ratios are invalid.
        """
        if self.use_cache and self.output_dir is not None:
            out_path = output or self._resolve_output_path("training", None)
            if out_path is not None:
                cached = self._load_cached(out_path)
                if cached is not None:
                    logger.info("Using cached training data from %s", out_path)
                    return cached

        df = self._build_labeled_dataframe()
        interrupted = self._last_generation_interrupted
        if interrupted:
            logger.warning("Skipping balancing after interrupt so only collected rows are saved.")
        elif balance is not False:
            df = balance_dataframe(df, label_col=label_col, balance=balance, seed=self.seed)
        if max_size is not None and len(df) > max_size:
            if balance is not False:
                df = cap_dataframe_balanced(df, max_size, label_col=label_col, seed=self.seed)
            else:
                df = df.sample(n=max_size, random_state=self.seed).reset_index(drop=True)
        if len(df) >= min_rows_for_split or interrupted:
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
            raise ValueError(f"Not enough labeled data to split: {len(df)} rows (need at least {min_rows_for_split}); consider providing more base data.")

        out_path = output or self._resolve_output_path("training", None)
        if out_path is not None:
            _save_dataframe(out_path, self.output_format, df)
        return df

    def generate_neutral_data(
        self,
        base_data: Optional[Union[str, pd.DataFrame, List[str]]] = None,
        max_size: Optional[int] = None,
        output: Optional[str] = None,
        text_column_out: str = "text",
    ) -> pd.DataFrame:
        """Generate neutral text rows via ``neutral_filter_fn`` or ``neutral_sentinel``.

        Args:
            base_data: Optional base-data override. Uses the creator base data
                when omitted.
            max_size: Optional cap on returned rows.
            output: Optional output path. Uses ``output_dir`` when configured and
                this argument is omitted.
            text_column_out: Name of the output text column.

        Returns:
            DataFrame with one text column named by ``text_column_out``.
        """
        if self.use_cache and self.output_dir is not None:
            out_path = output or self._resolve_output_path("neutral", None)
            if out_path is not None:
                cached = self._load_cached(out_path)
                if cached is not None:
                    logger.info("Using cached neutral data from %s", out_path)
                    return cached

        texts = self._get_data(base_override=base_data)
        interrupted = False
        if self.neutral_filter_fn is not None:
            neutral = []
            try:
                for t in texts:
                    if self.neutral_filter_fn(t):
                        neutral.append(t)
            except KeyboardInterrupt:
                interrupted = True
        elif self.neutral_sentinel is not None:
            # Only supported for scalar label_fn outputs.
            neutral = []
            try:
                for t in texts:
                    r = self.label_fn(t)
                    if isinstance(r, dict):
                        continue
                    if r == self.neutral_sentinel:
                        neutral.append(t)
            except KeyboardInterrupt:
                interrupted = True
        else:
            neutral = list(texts)
        if interrupted:
            logger.warning("Neutral data generation interrupted by user; keeping %s rows collected so far.", len(neutral))

        if max_size is not None and len(neutral) > max_size:
            rng = random.Random(self.seed)
            neutral = rng.sample(neutral, max_size)

        df = pd.DataFrame({text_column_out: neutral})
        out_path = output or self._resolve_output_path("neutral", None)
        if out_path is not None:
            _save_dataframe(out_path, self.output_format, df)
        return df

