"""
TextPredictionDataCreator: build training and neutral datasets for text prediction.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

from gradiend.data.core.base_loader import resolve_base_data
from gradiend.data.core import (
    SplitGroupKey,
    normalize_split_group_key,
    split_dataframe_per_group,
)
from gradiend.data.text import (
    SpacyTagSpec,
    TextFilterConfig,
    TextPreprocessConfig,
    iter_sentences_from_texts,
)
from gradiend.data.text.prediction.filter_engine import (
    filter_sentences_multi,
    mask_sentence,
)
from gradiend.data.text.prediction.balancing import _apply_balance
from gradiend.data.text.prediction.formats import _to_merged, _to_minimal, _to_partial_merged
from gradiend.data.text.prediction.ids import _class_id
from gradiend.data.text.prediction.io import _related_output_path, _save_data
from gradiend.data.text.prediction.neutral import _filter_neutral
from gradiend.data.text.prediction.summary import (
    SUMMARY_LOG_CLASS_THRESHOLD,
    SUMMARY_LOG_EXAMPLES,
    _log_training_filter_summary,
)
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

MIN_ROWS_PER_CLASS_FOR_SPLIT = 10
"""Minimum rows per class to perform train/val/test split. Splitting fewer rows yields tiny splits (e.g. 80/10/10 of 5 rows)."""


class TextPredictionDataCreator:
    """Create masked-token text-prediction datasets from base corpora.

    The creator scans text, finds configured target tokens, masks each match, and
    produces either per-class DataFrames or a unified training table consumable by
    :class:`~gradiend.trainer.text.prediction.trainer.TextPredictionTrainer`.
    It can also create neutral evaluation rows by excluding all configured target
    words and optional spaCy tag patterns.

    The main public workflow is:
    ``TextPredictionDataCreator(...).generate_training_data(...)`` and, when
    neutral evaluation data is needed,
    ``TextPredictionDataCreator(...).generate_neutral_data(...)``.
    """

    def __init__(
        self,
        base_data: Union[str, pd.DataFrame, List[str]],
        text_column: str = "text",
        base_max_size: Optional[int] = None,
        split: str = "train",
        hf_config: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        preprocess: Optional[TextPreprocessConfig] = None,
        spacy_model: Optional[str] = None,
        feature_targets: Optional[List[TextFilterConfig]] = None,
        min_left_context_words: int = 10,
        seed: int = 42,
        download_if_missing: bool = True,
        output_dir: Optional[str] = None,
        training_basename: str = "training",
        neutral_basename: str = "neutral",
        output_format: Literal["csv", "parquet", "hf"] = "csv",
        use_cache: bool = False,
        split_group_col: Optional[str] = None,
        split_group_key: SplitGroupKey = None,
    ) -> None:
        """Initialize with shared config for both generate methods.

        Args:
            base_data: HF id, pandas df, csv path, or List[str].
            text_column: Column name for text (default "text").
            base_max_size: Cap on base data (after shuffle, before preprocessing).
            split: HF split (default "train").
            hf_config: HF dataset config/subset (e.g. "20220301.en" for wikipedia).
            trust_remote_code: Optional value passed to load_dataset when base_data is HF id.
                None means do not pass the keyword.
            preprocess: Optional TextPreprocessConfig.
            spacy_model: Spacy model name (e.g. "de_core_news_sm"); lazy-loaded.
            feature_targets: List of TextFilterConfig. Each config's id (or first target) names the class.
            min_left_context_words: Default minimum word-like strings required before a matched target.
                Per-config overrides via TextFilterConfig.min_left_context_words are still supported.
                Mainly useful for decoder-only models where masked targets need left context.
            seed: Random seed for shuffle and sampling.
            download_if_missing: If True, auto-download spacy model when not found.
            output_dir: If set, generate_training_data/generate_neutral_data write to this folder
                when output= is not passed. Default filenames: training_basename + ext, neutral_basename + ext.
            training_basename: Base name for training output (default "training"); extension from output_format.
            neutral_basename: Base name for neutral output (default "neutral").
            output_format: "csv" (default), "parquet", or "hf" (HuggingFace datasets; per_class saves as subsets).
                "hf" requires the datasets library; falls back to csv with a warning if not installed.
            use_cache: If True and output_dir is set, generate_training_data and generate_neutral_data
                load from existing files in output_dir when available instead of regenerating.
            split_group_col: Column used for vocabulary-held-out splits (e.g. ``"label"`` for
                masked target tokens). Each unique value is assigned to exactly one of
                train/validation/test. None = random row split (legacy default).
            split_group_key: Callable or sequence of callables applied to ``split_group_col``
                values before grouping (e.g. ``[str.strip, str.casefold]``).
        """
        if not isinstance(text_column, str):
            raise TypeError(f"text_column must be str, got {type(text_column).__name__}")
        if base_max_size is not None and not isinstance(base_max_size, int):
            raise TypeError(f"base_max_size must be int or None, got {type(base_max_size).__name__}")
        if not isinstance(split, str):
            raise TypeError(f"split must be str, got {type(split).__name__}")
        if hf_config is not None and not isinstance(hf_config, str):
            raise TypeError(f"hf_config must be str or None, got {type(hf_config).__name__}")
        if trust_remote_code is not None and not isinstance(trust_remote_code, bool):
            raise TypeError(f"trust_remote_code must be bool or None, got {type(trust_remote_code).__name__}")
        if spacy_model is not None and not isinstance(spacy_model, str):
            raise TypeError(f"spacy_model must be str or None, got {type(spacy_model).__name__}")
        if not isinstance(min_left_context_words, int):
            raise TypeError(
                f"min_left_context_words must be int, got {type(min_left_context_words).__name__}"
            )
        if min_left_context_words < 0:
            raise ValueError(f"min_left_context_words must be >= 0, got {min_left_context_words}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        if not isinstance(download_if_missing, bool):
            raise TypeError(f"download_if_missing must be bool, got {type(download_if_missing).__name__}")
        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError(f"output_dir must be str or None, got {type(output_dir).__name__}")
        if not isinstance(training_basename, str):
            raise TypeError(f"training_basename must be str, got {type(training_basename).__name__}")
        if not isinstance(neutral_basename, str):
            raise TypeError(f"neutral_basename must be str, got {type(neutral_basename).__name__}")
        if not isinstance(output_format, str):
            raise TypeError(f"output_format must be str, got {type(output_format).__name__}")
        if output_format not in ("csv", "parquet", "hf"):
            raise TypeError(f"output_format must be 'csv', 'parquet', or 'hf', got {output_format!r}")
        if not isinstance(use_cache, bool):
            raise TypeError(f"use_cache must be bool, got {type(use_cache).__name__}")
        if split_group_col is not None and not isinstance(split_group_col, str):
            raise TypeError(f"split_group_col must be str or None, got {type(split_group_col).__name__}")
        normalize_split_group_key(split_group_key)

        self.base_data = base_data
        self.text_column = text_column
        self.base_max_size = base_max_size
        self.split = split
        self.hf_config = hf_config
        self.trust_remote_code = trust_remote_code
        self.preprocess = preprocess
        self.spacy_model = spacy_model
        self.feature_targets = feature_targets or []
        self.min_left_context_words = min_left_context_words
        self.seed = seed
        self.download_if_missing = download_if_missing
        self.output_dir = Path(output_dir) if output_dir else None
        self.training_basename = training_basename
        self.neutral_basename = neutral_basename
        self.output_format = output_format
        self.use_cache = use_cache
        self.split_group_col = split_group_col
        self.split_group_key = split_group_key
        self._texts_cache: Optional[List[str]] = None

    def _resolve_output_path(self, name: Literal["training", "neutral"], explicit: Optional[str]) -> Optional[str]:
        """Resolve output path: explicit path, or output_dir + basename + extension."""
        if explicit is not None:
            return explicit
        if self.output_dir is None:
            return None
        basename = self.training_basename if name == "training" else self.neutral_basename
        if self.output_format == "csv":
            return str(self.output_dir / f"{basename}.csv")
        if self.output_format == "parquet":
            return str(self.output_dir / f"{basename}.parquet")
        # hf: directory, no extension
        return str(self.output_dir / basename)

    def _load_cached_training(
        self,
        path: str,
        format: str,
    ) -> Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]]:
        """Load training data from path when use_cache and output_dir are set. Returns None if path does not exist."""
        p = Path(path)
        if not p.exists():
            return None
        fmt = self.output_format
        if fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "parquet":
            df = pd.read_parquet(path)
        else:
            try:
                from datasets import load_from_disk
            except ImportError:
                return None
            ds = load_from_disk(path)
            if hasattr(ds, "keys"):
                class_dfs = {k: v.to_pandas() for k, v in ds.items()}
                if format == "per_class":
                    return class_dfs
                if format == "minimal":
                    return _to_minimal(class_dfs)
                return _to_merged(class_dfs)
            df = ds.to_pandas()
        if format == "per_class":
            group_col = "label_class" if "label_class" in df.columns else "feature_class_id"
            if group_col not in df.columns:
                return df
            out = {}
            for k, g in df.groupby(group_col, sort=False):
                g = g.drop(columns=[group_col], errors="ignore").copy()
                if "label" in g.columns:
                    g[str(k)] = g["label"]
                out[str(k)] = g
            return out
        if format == "minimal":
            minimal_cols = ["masked", "label", "label_class", "split"]
            if all(c in df.columns for c in minimal_cols):
                return df[minimal_cols].copy()
        return df

    def _load_cached_neutral(self, path: str) -> Optional[pd.DataFrame]:
        """Load neutral data from path when use_cache and output_dir are set. Returns None if path does not exist."""
        p = Path(path)
        if not p.exists():
            return None
        fmt = self.output_format
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "parquet":
            return pd.read_parquet(path)
        try:
            from datasets import load_from_disk
        except ImportError:
            return None
        ds = load_from_disk(path)
        return ds.to_pandas()

    def _get_texts(
        self, base_override: Optional[Union[str, pd.DataFrame, List[str]]] = None
    ) -> List[str]:
        """Load base data as raw texts (no sentence splitting); cache when no override."""
        def _resolve(src: Union[str, pd.DataFrame, List[str]]) -> List[str]:
            is_hf_str = isinstance(src, str) and Path(src).suffix.lower() != ".csv"
            return resolve_base_data(
                src,
                text_column=self.text_column,
                max_size=self.base_max_size,
                split=self.split,
                seed=self.seed,
                hf_config=self.hf_config if is_hf_str else None,
                trust_remote_code=self.trust_remote_code if is_hf_str else None,
            )
        if base_override is not None:
            return _resolve(base_override)
        if self._texts_cache is not None:
            return self._texts_cache
        texts = _resolve(self.base_data)
        self._texts_cache = texts
        return texts

    def generate_training_data(
        self,
        max_size_per_class: Optional[int] = None,
        format: str = "per_class",
        split_name: str = "train",
        balance: Union[bool, str] = "try",
        output: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        min_rows_per_class_for_split: int = MIN_ROWS_PER_CLASS_FOR_SPLIT,
        min_rows_per_target_for_balance: int = 1,
        raise_on_incomplete_classes: bool = False,
        split_group_col: Optional[str] = None,
        split_group_key: SplitGroupKey = None,
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Generate masked training data by filtering configured target tokens.

        Each returned row contains the original ``text``, a ``masked`` version,
        the matched ``label`` token, ``token_count``, and a ``split`` column.
        ``format="per_class"`` returns one DataFrame per feature class. The
        ``"minimal"`` and ``"unified"`` formats return a single DataFrame; the
        unified format contains factual/alternative columns used by GRADIEND
        trainers.

        Args:
            max_size_per_class: Cap per feature class.
            format: Return structure: ``"per_class"`` (dict), ``"unified"``, or
                ``"minimal"``.
            split_name: Value for split column when auto_split is not used (default "train").
            balance: "try" (default) shuffles and caps classes without changing target-token
                counts; False disables all balancing; "strict" exactly balances target-token
                counts within each class, using replacement for targets below
                ``min_rows_per_target_for_balance``.
            output: If set, save the data to this path using ``self.output_format``
                (unified table when format is ``"per_class"``, otherwise the
                returned DataFrame).
            train_ratio: Fraction of each class for train (default 0.8).
            val_ratio: Fraction of each class for validation (default 0.1).
            test_ratio: Fraction of each class for test (default 0.1). Must sum to 1.0 with train_ratio and val_ratio.
            min_rows_per_class_for_split: Minimum rows per class to perform train/val/test split. Splitting fewer rows
                yields meaningless splits (e.g. 80/10/10 of 5 rows). Default 10. Set to 0 to disable this check.
            min_rows_per_target_for_balance: Floor used by ``balance="strict"`` when balancing
                target-token counts within a class. Sparse targets are oversampled with
                replacement up to this count unless ``max_size_per_class`` is too small.
            raise_on_incomplete_classes: If True, raise ValueError when non-empty classes have fewer than
                min_rows_per_class_for_split rows. Main and incomplete-class files are still saved before raising.
                Defaults to False, so incomplete classes are excluded from the main generated training data.
            split_group_col: Override instance ``split_group_col`` for this call. When set (e.g. ``"label"``),
                each unique target token is confined to a single train/validation/test split.
            split_group_key: Override instance ``split_group_key`` (e.g. ``[str.strip, str.casefold]``).

        Returns:
            Per format: dict of DataFrames, or single DataFrame.

        Raises:
            ValueError: If split ratios do not sum to ``1.0``, ``format`` is
                unknown, strict balancing is impossible for the requested cap, or
                ``raise_on_incomplete_classes`` is True and one or more non-empty
                classes have too few rows for splitting.
            TypeError: If lower-level split-key normalization receives an invalid
                ``split_group_key``.
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio, val_ratio, test_ratio must sum to 1.0")
        if self.use_cache and self.output_dir is not None:
            out_path = output or self._resolve_output_path("training", None)
            if out_path is not None:
                cached = self._load_cached_training(out_path, format)
                if cached is not None:
                    logger.info("Using cached training data from %s", out_path)
                    return cached
        texts = self._get_texts()
        configs_with_ids = [
            (_class_id(cfg, i), cfg) for i, cfg in enumerate(self.feature_targets)
        ]
        config_by_id = {cid: cfg for cid, cfg in configs_with_ids}
        stream = iter_sentences_from_texts(
            texts,
            self.preprocess,
            self.spacy_model,
            download_if_missing=self.download_if_missing,
        )
        total_target = (
            len(self.feature_targets) * max_size_per_class
            if max_size_per_class is not None
            else None
        )
        filter_stats: Dict[str, int] = {}
        results_per_class, _ = filter_sentences_multi(
            stream,
            configs_with_ids,
            spacy_model=self.spacy_model,
            download_if_missing=self.download_if_missing,
            max_matches_per_class=max_size_per_class,
            total_target_overall=total_target,
            stats=filter_stats,
            min_left_context_words_default=self.min_left_context_words,
        )
        interrupted = bool(filter_stats.get("interrupted"))
        n_processed = filter_stats.get("sentences_processed", 0)
        sentences_when_cap = filter_stats.get("sentences_when_cap_reached") or {}
        class_dfs = {}
        stats_per_group = {cid: len(matches) for cid, matches in results_per_class.items()}
        total_so_far = sum(stats_per_group.values())
        match_rates = {}
        log_per_class = len(stats_per_group) <= SUMMARY_LOG_CLASS_THRESHOLD
        empty_class_ids = []
        for class_id, matches in results_per_class.items():
            n = len(matches)
            # Use sentences scanned until this class hit its cap when available (more meaningful rate)
            denom = sentences_when_cap.get(class_id) or n_processed
            rate = (n / denom) if denom else 0.0
            match_rates[class_id] = rate
            cap_str = f"/{max_size_per_class}" if max_size_per_class is not None else ""
            if log_per_class:
                logger.info("  %s: %s%s matches (%.2f%% of sentences scanned)", class_id, n, cap_str, 100 * rate)
            cfg = config_by_id[class_id]
            rows = []
            for sent, spans in matches:
                masked = mask_sentence(sent, spans, cfg.mask)
                labels = [m[2] for m in spans]
                label = labels[0] if labels else ""
                rows.append({
                    "text": sent,
                    "masked": masked,
                    "label": label,
                    "token_count": len(spans),
                })
            if not rows:
                empty_class_ids.append(class_id)
                if log_per_class:
                    logger.warning(f"No matches for class '{class_id}'")
                continue
            df = pd.DataFrame(rows)
            df["split"] = split_name
            df[class_id] = df["label"]
            class_dfs[class_id] = df

        if stats_per_group:
            if total_target is not None and total_target > 0:
                pct = 100.0 * total_so_far / total_target
                logger.info("Overall: %s/%s (%.1f%%)", total_so_far, total_target, pct)
            _log_training_filter_summary(
                stats_per_group,
                match_rates,
                total_target=total_target,
                total_so_far=total_so_far,
                max_size_per_class=max_size_per_class,
                n_processed=n_processed,
            )
        if empty_class_ids and not log_per_class:
            logger.warning(
                "No matches for %s/%s classes. Examples: %s",
                len(empty_class_ids),
                len(stats_per_group),
                empty_class_ids[:SUMMARY_LOG_EXAMPLES],
            )

        if interrupted:
            logger.warning("Skipping balancing after interrupt so only collected rows are saved.")
        elif balance is not False:
            class_dfs = _apply_balance(
                class_dfs,
                max_size_per_class,
                balance,
                min_rows_per_target_for_balance,
                self.feature_targets,
                self.seed,
            )
        elif max_size_per_class is not None:
            for class_id in list(class_dfs.keys()):
                df = class_dfs[class_id]
                if len(df) > max_size_per_class:
                    class_dfs[class_id] = df.sample(
                        n=max_size_per_class, random_state=self.seed
                    ).reset_index(drop=True)

        incomplete_class_dfs: Dict[str, pd.DataFrame] = {}
        split_class_dfs = class_dfs
        incomplete_message: Optional[str] = None
        if (
            not interrupted
            and min_rows_per_class_for_split > 0
            and class_dfs
        ):
            too_small = [
                (cid, len(df))
                for cid, df in class_dfs.items()
                if 0 < len(df) < min_rows_per_class_for_split
            ]
            if too_small:
                incomplete_ids = {cid for cid, _ in too_small}
                incomplete_class_dfs = {
                    cid: df.copy()
                    for cid, df in class_dfs.items()
                    if cid in incomplete_ids
                }
                split_class_dfs = {
                    cid: df
                    for cid, df in class_dfs.items()
                    if cid not in incomplete_ids
                }
                incomplete_message = (
                    f"Data split requires at least {min_rows_per_class_for_split} rows per class. "
                    f"Classes with too few rows: {too_small}."
                )
                logger.warning(
                    "%s Excluding these classes from the main generated training data; "
                    "retrieved rows will be saved separately and are not used by GRADIEND training.",
                    incomplete_message,
                )

        split_group_col = self.split_group_col if split_group_col is None else split_group_col
        split_group_key = self.split_group_key if split_group_key is None else split_group_key

        split_dataframe_per_group(
            split_class_dfs,
            train_ratio,
            val_ratio,
            test_ratio,
            self.seed,
            split_col="split",
            min_rows_per_group=0,
            split_group_col=split_group_col,
            split_group_key=split_group_key,
        )

        if format == "per_class":
            result: Union[Dict[str, pd.DataFrame], pd.DataFrame] = split_class_dfs
            df_to_save = _to_partial_merged(split_class_dfs) if interrupted or len(split_class_dfs) < 2 else _to_merged(split_class_dfs)
        elif format == "minimal":
            result = _to_minimal(split_class_dfs)
            df_to_save = result
        elif format == "unified":
            result = _to_partial_merged(split_class_dfs) if interrupted or len(split_class_dfs) < 2 else _to_merged(split_class_dfs)
            df_to_save = result
        else:
            raise ValueError(f"format must be 'per_class', 'unified', or 'minimal'; got {format!r}")

        output = output or self._resolve_output_path("training", None)
        if output is not None:
            fmt = self.output_format
            if fmt == "hf":
                try:
                    from datasets import DatasetDict  # noqa: F401
                except ImportError:
                    fmt = "csv"
            _save_data(
                output,
                fmt,
                df=df_to_save if fmt != "hf" or not isinstance(result, dict) else None,
                class_dfs=split_class_dfs if fmt == "hf" and isinstance(result, dict) else None,
            )
            if incomplete_class_dfs:
                incomplete_output = _related_output_path(output, "incomplete_classes", fmt)
                incomplete_df = _to_partial_merged(incomplete_class_dfs)
                _save_data(
                    incomplete_output,
                    fmt,
                    df=incomplete_df if fmt != "hf" or not isinstance(result, dict) else None,
                    class_dfs=incomplete_class_dfs if fmt == "hf" and isinstance(result, dict) else None,
                )
                logger.warning(
                    "Saved %s incomplete-class rows to %s.",
                    len(incomplete_df),
                    incomplete_output,
                )
        elif incomplete_class_dfs:
            logger.warning(
                "Incomplete classes were excluded from returned training data, but no output path/output_dir "
                "was provided, so their retrieved rows were not saved."
            )

        if incomplete_message and raise_on_incomplete_classes:
            raise ValueError(
                f"{incomplete_message} Incomplete classes were excluded from the main data "
                "and saved separately before raising."
            )

        return result

    def _get_all_target_words(self) -> List[str]:
        """Collect all target strings from feature_targets (for neutral exclusion)."""
        words: List[str] = []
        for cfg in self.feature_targets:
            for t, _ in cfg.flatten_targets_with_tags():
                if t and t not in words:
                    words.append(t)
        return words

    def generate_neutral_data(
        self,
        base_data: Optional[Union[str, pd.DataFrame, List[str]]] = None,
        additional_excluded_words: Optional[List[str]] = None,
        excluded_spacy_tags: Optional[
            Union[SpacyTagSpec, List[SpacyTagSpec]]
        ] = None,
        max_size: Optional[int] = None,
        format: str = "minimal",
        output: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate neutral data by excluding sentences with target tokens.

        Excludes sentences containing:

        - Any token in (target words + additional_excluded_words), deduplicated
        - Any token matching any spec in excluded_spacy_tags

        Use excluded_spacy_tags=[{"pos": "DET"}, {"pos": "PRON", "Person": "3"}]
        to exclude determiners and third-person pronouns.

        Args:
            base_data: Optional override (otherwise uses creator's base).
            additional_excluded_words: Extra words to exclude (in addition to
                target words from feature_targets). E.g. gendered articles or pronouns.
            excluded_spacy_tags: Spacy tag spec(s); exclude if any token matches
                any spec. Use list for multiple: [{"pos": "DET"}, {"pos": "PRON", "Person": "3"}].
            max_size: Global cap for neutral dataset.
            format: Return format ("minimal" = text column for eval).
            output: If set, save neutral data to this path. When output_dir is set on the
                creator and output is None, uses output_dir/neutral_basename + extension.

        Returns:
            DataFrame with at least "text" column.

        Raises:
            ValueError: If ``excluded_spacy_tags`` is set but no ``spacy_model``
                was configured.
        """
        if self.use_cache and self.output_dir is not None:
            out_path = output or self._resolve_output_path("neutral", None)
            if out_path is not None:
                cached = self._load_cached_neutral(out_path)
                if cached is not None:
                    logger.info("Using cached neutral data from %s", out_path)
                    return cached
        texts = self._get_texts(base_override=base_data)
        sentence_stream = iter_sentences_from_texts(
            texts,
            self.preprocess,
            self.spacy_model,
            download_if_missing=self.download_if_missing,
        )
        target_words = self._get_all_target_words()
        extra = list(additional_excluded_words or [])
        excluded_words = list(dict.fromkeys(target_words + extra))
        if not excluded_words and not excluded_spacy_tags:
            if max_size is not None:
                neutral = []
                try:
                    for sentence in sentence_stream:
                        neutral.append(sentence)
                        if len(neutral) >= max_size:
                            break
                except KeyboardInterrupt:
                    logger.warning("Neutral data generation interrupted by user; keeping %s rows collected so far.", len(neutral))
                logger.info("Neutral: no exclusion filters, stopped at max_size=%s (got %s).", max_size, len(neutral))
            else:
                neutral = []
                try:
                    for sentence in sentence_stream:
                        neutral.append(sentence)
                except KeyboardInterrupt:
                    logger.warning("Neutral data generation interrupted by user; keeping %s rows collected so far.", len(neutral))
                logger.info("Neutral: no exclusion filters, kept all %s sentences.", len(neutral))
        else:
            neutral, neutral_stats = _filter_neutral(
                sentence_stream,
                excluded_words,
                excluded_spacy_tags,
                self.spacy_model,
                self.download_if_missing,
                max_size=max_size,
            )
            total_sent = neutral_stats.get("total", 0)
            kept = neutral_stats.get("kept", 0)
            rate = (kept / total_sent) if total_sent else 0.0
            logger.info("Neutral filter stats: %s (success rate: %.2f)", neutral_stats, rate)
        if neutral is None:
            neutral = []
        rows = [{"text": s} for s in neutral]
        df = pd.DataFrame(rows)

        out_path = output or self._resolve_output_path("neutral", None)
        if out_path is not None:
            fmt = self.output_format
            if fmt == "hf":
                try:
                    from datasets import Dataset  # noqa: F401
                except ImportError:
                    fmt = "csv"
            _save_data(out_path, fmt, df=df)

        return df
