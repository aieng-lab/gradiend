"""
Text prediction: MLM/CLM trainers using DataFrames.

This module provides TextPredictionTrainer for handling MLM/CLM data from pandas DataFrames.
It supports per-class datasets and automatically creates training pairs.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Dict, List, Optional, Union, Any, Tuple, Type

from gradiend.trainer import Trainer
from gradiend.trainer.config import TrainerConfig
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.model import ModelWithGradiend
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset, create_masked_pair_from_text
from pathlib import Path
from gradiend.trainer.text.prediction.unified_data import (
    all_subsets_to_mlm_df,
    apply_class_merge_to_merged_df,
    merged_to_unified,
    merge_per_class_dfs,
    per_class_dict_to_unified,
    resolve_dataframe,
    _load_dataframe_from_path,
    load_hf_per_class,
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_ALTERNATIVE,
    UNIFIED_TRANSITION,
    transition_id,
)
from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
from gradiend.trainer.text.prediction.decoder_only_mlm import train_mlm_head
from gradiend.util.logging import get_logger
from gradiend.model.utils import is_decoder_only_model as is_decoder_only_model_from_obj
from gradiend.util import normalize_split_name
from gradiend.util.paths import (
    resolve_encoder_analysis_path,
    resolve_decoder_per_model_cache_path,
    resolve_decoder_mlm_head_dir,
)
from gradiend.evaluator.encoder import encode_dataset_to_rows
from gradiend.evaluator.encoder_metrics import invalidate_encoder_metrics_cache
from gradiend.trainer.text.prediction.decoder_eval_utils import evaluate_probability_shift_score, compute_lms
from gradiend.trainer.text.common.dataset import TextGradientTrainingDataset

logger = get_logger(__name__)


@dataclass
class TextPredictionConfig(TrainerConfig):
    """
    Configuration for TextPredictionTrainer.

    Unified data contract: internal representation uses masked, split, factual_class,
    alternative_class, factual, alternative, transition. Training uses only rows where
    transition ∈ {c1→c2, c2→c1} for the configured pair.

    Data input:
    - data as Dict[str, DataFrame]: per-class format. Key = factual_class; each df has
      masked, split, and columns = class names (factual = df[factual_class], alternative = df[alternative_class]).
      If a df has no column for another class (single-token-per-class, e.g. Gender), target
      is inferred as the other class's token for the configured pair.
    - data as DataFrame: merged format with label_class_col, label_col, and optionally
      target_col, target_class_col. If target columns omitted, pair is required and
      target = other class's token.
    - hf_dataset: load from HuggingFace (merged format), then convert to unified.

    Attributes:
        run_id: Optional run identifier (subdir and display).
        data: Per-class dict (label_class -> DataFrame) or merged DataFrame.
        hf_dataset: HuggingFace dataset ID; loads merged format and converts to unified.
        hf_subset: Subset name(s) to load when using hf_dataset.
        hf_splits: Splits to include (e.g. ['train', 'validation', 'test']).
        target_classes: Target classes for training. Pair is automatically inferred when len(target_classes) == 2.
        all_classes: All classes available in the dataset. If None (default), inferred from data.
            When loading from HuggingFace datasets, None means load all configs/subsets.
        masked_col, split_col: Column names. For merged format also label_col, label_class_col,
            and optionally target_col, target_class_col.
        use_class_names_as_columns: For per-class data, use class name as column name for tokens.
    """

    run_id: Optional[str] = None
    data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], str, Path]] = None
    hf_dataset: Optional[str] = None
    hf_subset: Optional[Union[str, List[str]]] = None
    hf_splits: Optional[List[str]] = None
    target_classes: Optional[List[str]] = None
    """Target classes for training. Pair is automatically inferred when len(target_classes) == 2."""
    
    all_classes: Optional[List[str]] = None
    """All classes available in the dataset. If None (default), inferred from data. 
    When loading from HuggingFace datasets, None means load all configs/subsets."""


    # Column names (merged format)
    masked_col: str = "masked"
    label_col: str = "label"
    label_class_col: str = "label_class"
    split_col: str = "split"
    # Explicit target columns (merged format only; per-class uses class names as columns).
    # Defaults match creator output (label_class, label, alternative, alternative_class) for generated data.
    alternative_col: Optional[str] = "alternative"
    alternative_class_col: Optional[str] = "alternative_class"

    # Per-class: class name = column name for that class's token
    use_class_names_as_columns: bool = True

    # Per-class single-token mode: when alternative is derived from other class df,
    # up to this many unique (case-insensitive) counterfactual tokens per base sentence (default 1).
    max_counterfactuals_per_sentence: int = 1
    # Seed for reproducible weighted sampling of counterfactuals; None = non-deterministic.
    random_state: Optional[int] = None

    # Training options
    n_features: int = 1

    # Decoder evaluation (optional - defaults to training targets/data)
    decoder_eval_targets: Optional[Dict[str, List[str]]] = None
    # e.g., {'white': ['white'], 'black': ['black']}
    # If None, uses training target tokens per class (inferred from target_col/source_col)

    decoder_eval_restrict_to_target_classes: bool = True
    # If True, decoder analysis uses only the target classes (excludes neutral/identity augmenting classes).
    # Set False to evaluate over all classes in decoder_eval_targets.

    decoder_eval_prob_on_other_class: bool = True
    # If True, each target's probability is evaluated on the other class's data only (e.g. P(target1) on target2 rows).
    # Requires eval data to have alternative_id (or data_class_col).

    decoder_eval_ignore_tokens: Optional[List[str]] = None
    # Tokens to ignore in LMS evaluation
    # If None, uses non-neutral terms from trainer (if available)

    decoder_eval_lms_max_samples: Optional[int] = None
    # Max number of texts used for LMS in decoder evaluation. None = use all.

    # Class merging: map base classes to higher-level classes (e.g. {"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]})
    class_merge_map: Optional[Dict[str, List[str]]] = None
    # When set, target_classes and pair use merged names (keys). Values list base classes to merge.
    # Base classes not in any value remain as neutral. When exactly 2 keys, target_classes can be omitted.
    # Optional: limit which base-class transitions are created before merging, by specifying
    # clusters of base classes. Only transitions where BOTH raw classes lie in the same cluster
    # (and are different) are kept (e.g. [["1SG","1PL"], ["3SG","3PL"]] keeps 1SG↔1PL and 3SG↔3PL).
    class_merge_transition_groups: Optional[List[List[str]]] = None

    # Neutral evaluation data
    eval_neutral_data: Optional[Union[pd.DataFrame, str, Path]] = None
    # Optional DataFrame, local path (.csv/.parquet), or HuggingFace dataset id. Paths loaded via resolve_dataframe.

    eval_neutral_max_rows: Optional[int] = None
    # Optional cap on number of rows loaded from neutral HF datasets.


class TextPredictionTrainer(Trainer):
    """
    Trainer for text prediction (MLM/CLM) using DataFrames.

    Handles MLM/CLM data from pandas DataFrames with support for:
    - Per-class datasets (e.g., one DataFrame for "Asian", one for "White")
    - Automatic class pair combination (e.g., Asian<->White)
    - Factual/counterfactual creation
    - Automatic label mapping

    Required DataFrame columns (names configurable via TextPredictionConfig):
    - masked: Text with mask tokens
    - label: Target token (e.g., "he", "He")
    - label_class: Feature class (e.g., "male", "female", "Asian", "White")
    - split: train/val/test

    Optional:
    - correlation_mapping: Dict mapping label_class -> correlation value (default: +1/-1 for binary)
    """

    @property
    def default_model_with_gradiend_cls(self) -> Type[ModelWithGradiend]:
        """
        Default ModelWithGradiend subclass for TextPredictionTrainer.
        
        Returns TextPredictionModelWithGradiend (TextModelWithGradiend).
        """
        return TextPredictionModelWithGradiend

    def get_target_feature_class_ids(self):
        """
        Feature class IDs for target classes (pair transitions only; excludes identity/neutral).
        In create_training_data the pair transitions are assigned 0 and 1; identity classes follow.
        """
        if self.pair is not None:
            return [0, 1]
        return None

    def resolve_custom_prediction_head_dir(self) -> Optional[str]:
        """
        Return the directory path for a trained decoder-only MLM head if it exists.
        
        Uses resolve_decoder_mlm_head_dir to determine the path. When this path exists,
        resolve_model_path will automatically use it instead of the base model.
        DecoderModelWithMLMHead replaces AutoModelForMaskedLM in loading; no special adapter logic.
        """
        experiment_dir = self.experiment_dir
        if experiment_dir is None:
            return None
        return resolve_decoder_mlm_head_dir(experiment_dir)

    def _encoder_cache_path(self, model_path: str, **encoder_kwargs: Any) -> Optional[str]:
        """
        Encoder cache path for analysis CSV.
        Cache under experiment_dir; includes split/max_size in cache key.
        """
        experiment_dir = self.experiment_dir
        split = encoder_kwargs.get("split")
        max_size = encoder_kwargs.get("max_size")
        key_kwargs: Dict[str, Any] = {}
        if split is not None:
            key_kwargs["split"] = split
        if max_size is not None:
            key_kwargs["max_size"] = max_size
        return resolve_encoder_analysis_path(experiment_dir, None, **key_kwargs)

    def __init__(
            self,
            model: Union[str, Any],
            run_id: Optional[str] = None,
            data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], str, Path]] = None,
            correlation_mapping: Optional[Dict[str, float]] = None,
            config: Optional[TextPredictionConfig] = None,
            target_classes: Optional[Union[List[str], Tuple[str, ...]]] = None,
            *,
            args: Optional[TrainingArguments] = None,
            training_args: Optional[TrainingArguments] = None,
            evaluator_class: Optional[Type] = None,
            hf_dataset: Optional[str] = None,
            hf_subset: Optional[Union[str, List[str]]] = None,
            hf_splits: Optional[List[str]] = None,
            all_classes: Optional[List[str]] = None,
            masked_col: Optional[str] = None,
            label_col: Optional[str] = None,
            label_class_col: Optional[str] = None,
            split_col: Optional[str] = None,
            alternative_col: Optional[str] = None,
            alternative_class_col: Optional[str] = None,
            use_class_names_as_columns: Optional[bool] = None,
            max_counterfactuals_per_sentence: Optional[int] = None,
            random_state: Optional[int] = None,
            n_features: Optional[int] = None,
            decoder_eval_targets: Optional[Dict[str, List[str]]] = None,
            decoder_eval_restrict_to_target_classes: Optional[bool] = None,
            decoder_eval_prob_on_other_class: Optional[bool] = None,
            decoder_eval_ignore_tokens: Optional[List[str]] = None,
            decoder_eval_lms_max_samples: Optional[int] = None,
            eval_neutral_data: Optional[Union[pd.DataFrame, str, Path]] = None,
            eval_neutral_max_rows: Optional[int] = None,
            img_format: Optional[str] = None,
            img_dpi: Optional[int] = None,
            class_merge_map: Optional[Dict[str, List[str]]] = None,
            class_merge_transition_groups: Optional[List[List[str]]] = None,
    ):
        """
        Initialize TextPredictionTrainer (Trainer with model at creation time).

        Two usage patterns are supported:

        1) **Config object**: pass a full ``TextPredictionConfig`` as ``config=``.
        2) **Explicit parameters**: pass ``run_id``, ``data``, ``target_classes`` and any
           other config fields as keyword arguments; they are wrapped into an internal
           TextPredictionConfig. Omitted arguments use the config dataclass defaults.

        The number of different counterfactuals paired with the same factual sentence
        (when multiple are available) is controlled by **max_counterfactuals_per_sentence**
        (default 1). Only applies in per-class single-token mode when the alternative
        is derived from the other class's DataFrame.

        Args:
            model: Model identifier (string path) or ModelWithGradiend instance.
            run_id: Optional run identifier (subdir and display).
            data: Training data (DataFrame, dict of DataFrames, or path to .csv/.parquet).
            correlation_mapping: Optional correlation mapping dict.
            config: Optional TextPredictionConfig instance. If given, other config-related
                kwargs are ignored except target_classes.
            target_classes: Target classes for training (e.g. ["3SG", "3PL"]). Pair is
                inferred when len(target_classes) == 2.
            args: Alias for training_args. Training arguments (batch size, steps, etc.).
            training_args: Training arguments. If both args and training_args are set,
                training_args takes precedence.
            evaluator_class: Optional custom Evaluator class.
            hf_dataset: HuggingFace dataset ID when loading from HF instead of data.
            hf_subset: Subset/config name(s) for HF dataset.
            hf_splits: Splits to load (e.g. ["train", "validation"]).
            all_classes: All class names in the dataset; inferred from data if None.
            masked_col: Column name for masked sentences (default "masked").
            label_col: Column name for factual token (default "label").
            label_class_col: Column name for factual class (default "label_class").
            split_col: Column name for split (default "split").
            alternative_col: Column name for alternative token in merged format.
            alternative_class_col: Column name for alternative class in merged format.
            use_class_names_as_columns: Use class name as column for that class's token.
            max_counterfactuals_per_sentence: Max unique counterfactual tokens per base
                sentence when deriving from other class (default 1).
            random_state: Seed for reproducible counterfactual sampling; None = nondeterministic.
            n_features: Number of features (default 1).
            decoder_eval_targets: Per-class token lists for decoder evaluation.
            decoder_eval_restrict_to_target_classes: Restrict decoder eval to target classes.
            decoder_eval_prob_on_other_class: Evaluate target prob on other class's data.
            decoder_eval_ignore_tokens: Tokens to ignore in LMS evaluation.
            decoder_eval_lms_max_samples: Max samples for LMS in decoder eval.
            eval_neutral_data: DataFrame or path for neutral evaluation data.
            eval_neutral_max_rows: Max rows to load from neutral HF datasets.
            img_format: Image format for plots (e.g. 'pdf', 'png'). Default 'pdf'.
            img_dpi: DPI for saved plots (e.g. 600 for publication). None = use visualizer default.
        """
        args_for_super = training_args or args
        if config is None:
            cfg_kwargs: Dict[str, Any] = {
                "run_id": run_id,
                "data": data,
                "hf_dataset": hf_dataset,
                "hf_subset": hf_subset,
                "hf_splits": hf_splits,
                "target_classes": list(target_classes) if isinstance(target_classes, tuple) else target_classes,
                "all_classes": all_classes,
                "masked_col": masked_col,
                "label_col": label_col,
                "label_class_col": label_class_col,
                "split_col": split_col,
                "alternative_col": alternative_col,
                "alternative_class_col": alternative_class_col,
                "use_class_names_as_columns": use_class_names_as_columns,
                "max_counterfactuals_per_sentence": max_counterfactuals_per_sentence,
                "random_state": random_state,
                "n_features": n_features,
                "decoder_eval_targets": decoder_eval_targets,
                "decoder_eval_restrict_to_target_classes": decoder_eval_restrict_to_target_classes,
                "decoder_eval_prob_on_other_class": decoder_eval_prob_on_other_class,
                "decoder_eval_ignore_tokens": decoder_eval_ignore_tokens,
                "decoder_eval_lms_max_samples": decoder_eval_lms_max_samples,
                "eval_neutral_data": eval_neutral_data,
                "eval_neutral_max_rows": eval_neutral_max_rows,
                "img_format": img_format,
                "img_dpi": img_dpi,
                "class_merge_map": class_merge_map,
                "class_merge_transition_groups": class_merge_transition_groups,
            }
            cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if v is not None}
            if target_classes is not None:
                cfg_kwargs["target_classes"] = list(target_classes) if isinstance(target_classes, tuple) else target_classes
            config = TextPredictionConfig(**cfg_kwargs)

        if target_classes:
            config.target_classes = list(target_classes) if isinstance(target_classes, tuple) else target_classes
        elif config.target_classes:
            target_classes = config.target_classes
        elif config.class_merge_map and len(config.class_merge_map) == 2:
            config.target_classes = list(config.class_merge_map.keys())
            target_classes = config.target_classes

        self.config: TextPredictionConfig = config
        super().__init__(
            model=model,
            args=args_for_super,
            run_id=config.run_id,
            n_features=config.n_features,
            evaluator_class=evaluator_class,
            target_classes=target_classes,
        )

        # Public metadata - set target_classes from config
        if config.target_classes is not None:
            tc = list(config.target_classes)
            self._validate_target_classes_unique(tc)
            self._target_classes = tc
        else:
            self._target_classes = None

        # Internal data / mappings (loaded lazily by _ensure_data() when needed)
        self.correlation_mapping = correlation_mapping or {}
        self.data = None
        self.class_datasets = None
        self._combined_data: Optional[pd.DataFrame] = None
        self._data_loaded = False

    def _set_all_classes(self, classes: Optional[List[str]]) -> None:
        """Set the list of all classes in the dataset (including neutral/identity)."""
        if classes is None:
            return
        self._all_classes = classes

    def _ensure_data_for_training(self) -> None:
        """Ensure data is loaded before creating the model for training (so pair is set and from_pretrained can set feature_class_encoding_direction)."""
        self._ensure_data()

    def _ensure_data(self) -> None:
        """Load and normalize data on first use. Idempotent.

        Training data can be specified as:
        - config.hf_dataset: HuggingFace dataset ID (optional subset/splits).
        - config.data: HuggingFace dataset ID (per-class configs), local path (.csv/.parquet),
          per-class dict, or DataFrame in memory. A string is treated as HF id unless it is
          an existing file path.
        """
        if self._data_loaded:
            return
        config = self.config
        # Single HuggingFace gate: hf_dataset => merged-style load; data=str (not a path) => per-class load
        is_hf_id = isinstance(config.data, str) and not Path(config.data).is_file()
        if config.hf_dataset is not None:
            # HF merged-style: one dataset, optional subsets/splits, then optional merge/transition filtering, then merged_to_unified
            raw = self._load_hf_dataset(
                config.hf_dataset,
                config.hf_subset,
                config.hf_splits,
            )
            self.data = raw
            merge_map = config.class_merge_map
            data_df = raw
            if (
                merge_map
                and config.alternative_col
                and config.alternative_class_col
                and config.alternative_col in data_df.columns
                and config.alternative_class_col in data_df.columns
            ):
                data_df = apply_class_merge_to_merged_df(
                    data_df,
                    merge_map,
                    label_class_col=config.label_class_col,
                    target_class_col=config.alternative_class_col,
                    target_classes=config.target_classes,
                    keep_raw=True,
                    transition_groups=getattr(config, "class_merge_transition_groups", None),
                )
            self._combined_data = merged_to_unified(
                data_df,
                masked_col=config.masked_col,
                split_col=config.split_col,
                label_class_col=config.label_class_col,
                label_col=config.label_col,
                target_col=config.alternative_col,
                target_class_col=config.alternative_class_col,
                pair=self.pair,
            )
        elif is_hf_id:
            # HF per-class: load configs/subsets as class_dfs, merge by class_merge_map if set, then per_class_dict_to_unified
            classes_to_load = config.all_classes if config.all_classes is not None else "all"
            class_dfs = load_hf_per_class(
                config.data,
                classes=classes_to_load,
                splits=config.hf_splits,
                masked_col=config.masked_col,
                split_col=config.split_col,
            )
            self.data = class_dfs
            merge_map = config.class_merge_map
            if merge_map:
                class_dfs = merge_per_class_dfs(class_dfs, merge_map, config.target_classes)
                inferred_classes = list(class_dfs.keys())
                pair = tuple(config.target_classes) if config.target_classes and len(config.target_classes) == 2 else tuple(inferred_classes) if len(inferred_classes) == 2 else None
            else:
                inferred_classes = list(class_dfs.keys())
                pair = tuple(config.target_classes) if config.target_classes and len(config.target_classes) == 2 else None
            self.class_datasets = class_dfs
            if config.all_classes is not None:
                self._set_all_classes(config.all_classes)
            else:
                self._set_all_classes(sorted(inferred_classes))
            unified = per_class_dict_to_unified(
                class_dfs,
                classes=inferred_classes,
                masked_col=config.masked_col,
                split_col=config.split_col,
                use_class_names_as_columns=getattr(config, "use_class_names_as_columns", True),
                pair=pair,
                include_identity_rows=False,
                max_counterfactuals_per_sentence=getattr(config, "max_counterfactuals_per_sentence", 1),
                random_state=getattr(config, "random_state", None),
            )
            self._combined_data = unified
        elif config.data is not None:
            if isinstance(config.data, dict):
                # Infer all_classes from data keys; merge by class_merge_map if set
                all_classes_from_data = sorted(list(config.data.keys()))
                self._set_all_classes(all_classes_from_data)
                classes_for_transitions = self.target_classes or all_classes_from_data
                if self._target_classes is None:
                    self._target_classes = classes_for_transitions
                self.data = config.data
                class_dfs = config.data
                merge_map = getattr(config, "class_merge_map", None)
                if merge_map:
                    class_dfs = merge_per_class_dfs(class_dfs, merge_map, self._target_classes)
                    classes_for_transitions = list(class_dfs.keys())
                    pair = tuple(self._target_classes) if self._target_classes and len(self._target_classes) == 2 else (tuple(classes_for_transitions) if len(classes_for_transitions) == 2 else None)
                else:
                    pair = tuple(config.target_classes) if config.target_classes and len(config.target_classes) == 2 else tuple(self._target_classes) if self._target_classes and len(self._target_classes) == 2 else None
                self.class_datasets = class_dfs
                unified = per_class_dict_to_unified(
                    class_dfs,
                    classes=classes_for_transitions,
                    masked_col=config.masked_col,
                    split_col=config.split_col,
                    use_class_names_as_columns=getattr(config, "use_class_names_as_columns", True),
                    pair=pair,
                    include_identity_rows=False,
                    max_counterfactuals_per_sentence=getattr(config, "max_counterfactuals_per_sentence", 1),
                    random_state=getattr(config, "random_state", None),
                )
                self._combined_data = unified
            elif isinstance(config.data, (Path, str)) and Path(config.data).is_file():
                # Local file path: load CSV/Parquet; merge by class_merge_map if set; then merged_to_unified
                data_df = _load_dataframe_from_path(config.data)
                self.data = data_df
                merge_map = getattr(config, "class_merge_map", None)
                if (
                    merge_map
                    and config.alternative_col
                    and config.alternative_class_col
                    and config.alternative_col in data_df.columns
                    and config.alternative_class_col in data_df.columns
                ):
                    data_df = apply_class_merge_to_merged_df(
                        data_df,
                        merge_map,
                        label_class_col=config.label_class_col,
                        target_class_col=config.alternative_class_col,
                        target_classes=config.target_classes,
                        keep_raw=True,
                        transition_groups=getattr(config, "class_merge_transition_groups", None),
                    )
                self._combined_data = merged_to_unified(
                    data_df,
                    masked_col=config.masked_col,
                    split_col=config.split_col,
                    label_class_col=config.label_class_col,
                    label_col=config.label_col,
                    target_col=config.alternative_col,
                    target_class_col=config.alternative_class_col,
                    pair=self.pair,
                )
                if self._combined_data is not None:
                    src = self._combined_data[UNIFIED_FACTUAL_CLASS].unique().tolist()
                    tgt = self._combined_data[UNIFIED_ALTERNATIVE_CLASS].unique().tolist()
                    self._set_all_classes(sorted(set(src) | set(tgt)))
            else:
                # DataFrame in memory; merge by class_merge_map if set
                data_df = config.data
                merge_map = getattr(config, "class_merge_map", None)
                if (
                    merge_map
                    and config.alternative_col
                    and config.alternative_class_col
                    and config.alternative_col in data_df.columns
                    and config.alternative_class_col in data_df.columns
                ):
                    data_df = apply_class_merge_to_merged_df(
                        data_df,
                        merge_map,
                        label_class_col=config.label_class_col,
                        target_class_col=config.alternative_class_col,
                        target_classes=config.target_classes,
                        keep_raw=True,
                        transition_groups=getattr(config, "class_merge_transition_groups", None),
                    )
                self.data = config.data
                self._combined_data = merged_to_unified(
                    data_df,
                    masked_col=config.masked_col,
                    split_col=config.split_col,
                    label_class_col=config.label_class_col,
                    label_col=config.label_col,
                    target_col=config.alternative_col,
                    target_class_col=config.alternative_class_col,
                    pair=self.pair,
                )
                if self._combined_data is not None:
                    src = self._combined_data[UNIFIED_FACTUAL_CLASS].unique().tolist()
                    tgt = self._combined_data[UNIFIED_ALTERNATIVE_CLASS].unique().tolist()
                    self._set_all_classes(sorted(set(src) | set(tgt)))
        self._check_data_non_empty()
        self._data_loaded = True

        if self._all_classes is None and self._combined_data is not None:
            src = self._combined_data[UNIFIED_FACTUAL_CLASS].unique().tolist()
            tgt = self._combined_data[UNIFIED_ALTERNATIVE_CLASS].unique().tolist()
            self._set_all_classes(sorted(set(src) | set(tgt)))
        
        # Infer pair from target_classes when exactly 2 target classes (stored in config.pair)
        if self._target_classes is not None and len(self._target_classes) == 2:
            config.pair = tuple(self._target_classes)
        if self.config.decoder_eval_targets is None and self._combined_data is not None:
            try:
                self.config.decoder_eval_targets = self._infer_decoder_eval_targets()
            except Exception as e:
                logger.warning(f"Could not auto-infer decoder_eval_targets: {e}")

    @property
    def combined_data(self) -> Optional[pd.DataFrame]:
        """Unified training data (lazy-loaded on first access). When class_merge_map is set, already merged at load."""
        self._ensure_data()
        return self._combined_data

    def plot_training_convergence(self, **kwargs: Any) -> Any:
        if "img_format" not in kwargs:
            kwargs["img_format"] = getattr(self.config, "img_format", "pdf")
        if "dpi" not in kwargs and getattr(self.config, "img_dpi", None) is not None:
            kwargs["dpi"] = self.config.img_dpi
        return super().plot_training_convergence(**kwargs)

    def plot_encoder_distributions(self, **kwargs: Any) -> Any:
        if "img_format" not in kwargs:
            kwargs["img_format"] = getattr(self.config, "img_format", "pdf")
        if "dpi" not in kwargs and getattr(self.config, "img_dpi", None) is not None:
            kwargs["dpi"] = self.config.img_dpi
        return super().plot_encoder_distributions(**kwargs)

    def plot_probability_shifts(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        class_ids: Optional[List[str]] = None,
        target_class: Optional[str] = None,
        increase_target_probabilities: bool = True,
        use_cache: Optional[bool] = None,
        **kwargs: Any
    ) -> str:
        """Plot decoder evaluation probability shifts vs learning rate for a target class.
        Uses target_class and increase_target_probabilities (default True = strengthen) to choose which summary config to plot."""
        if "img_format" not in kwargs:
            kwargs["img_format"] = getattr(self.config, "img_format", "pdf")
        if "dpi" not in kwargs and getattr(self.config, "img_dpi", None) is not None:
            kwargs["dpi"] = self.config.img_dpi
        return self.evaluator.plot_probability_shifts(
            decoder_results=decoder_results,
            class_ids=class_ids,
            target_class=target_class,
            increase_target_probabilities=increase_target_probabilities,
            use_cache=use_cache,
            **kwargs
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_data_non_empty(self) -> None:
        """Raise ValueError if any provided training data has length 0."""
        if self.class_datasets is not None:
            empty = [c for c, df in self.class_datasets.items() if len(df) == 0]
            if empty:
                raise ValueError(
                    f"Per-class training data has 0 rows for class(es): {empty}. "
                    "Ensure each class DataFrame has at least one row."
                )
        if self._combined_data is not None and len(self._combined_data) == 0:
            raise ValueError(
                "Unified training data has 0 rows. "
                "Check that input data (DataFrame, per-class dict, or HuggingFace dataset) contains rows."
            )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame has the required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        cfg = self.config
        required = {cfg.masked_col, cfg.label_col, cfg.label_class_col, cfg.split_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in text prediction data: {missing}")

    @staticmethod
    def _load_hf_dataset(
            dataset_name: str,
            subset: Optional[Union[str, List[str]]] = None,
            splits: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a HuggingFace dataset and convert it to a pandas DataFrame.

        This is a convenience method for loading HF datasets with common patterns:
        - Handles multiple subsets (e.g., "white_to_black" and "black_to_white")
        - Adds split column to each split
        - Concatenates all splits into a single DataFrame

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "aieng-lab/gradiend_race_data")
            subset: Optional subset name(s). If str, loads that subset.
                If list, loads multiple subsets and concatenates them.
                If None, loads the default subset.
            splits: Optional list of splits to include (e.g., ['train', 'validation', 'test']).
                If None, includes all available splits.

        Returns:
            Combined pandas DataFrame with all splits, including a 'split' column.

        Example:
            >>> df = TextPredictionTrainer._load_hf_dataset(
            ...     "aieng-lab/gradiend_race_data",
            ...     subset=["white_to_black", "black_to_white"],
            ...     splits=['train', 'validation', 'test']
            ... )
        """
        try:
            from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for HF dataset support. "
                "Install with: pip install datasets "
                "Or install all recommended packages: pip install gradiend[recommended]"
            )

        # Handle subset(s).
        if subset is None:
            # Auto-discover all available configs (subsets) when none is specified.
            try:
                config_names = get_dataset_config_names(dataset_name)
            except Exception:
                config_names = []
            # If configs exist (like fem_nom, masc_nom, ...), load all of them.
            # Otherwise fall back to the default (no-training_args) dataset.
            subsets_to_load = config_names or [None]
        elif isinstance(subset, str):
            subsets_to_load = [subset]
        else:
            subsets_to_load = subset

        # Load datasets for each subset
        datasets_with_split = []
        for sub in subsets_to_load:
            try:
                if sub is None:
                    ds = load_dataset(dataset_name, trust_remote_code=True)
                else:
                    ds = load_dataset(dataset_name, sub, trust_remote_code=True)
            except Exception as e:
                raise ValueError(f"Could not load subset '{sub}' from {dataset_name}: {e}")

            # Handle both DatasetDict and Dataset
            if hasattr(ds, 'items'):  # DatasetDict
                for split_name, split_ds in ds.items():
                    # Normalize split name (e.g., 'val' -> 'validation')
                    normalized_split_name = normalize_split_name(split_name)

                    # Check if this split should be included
                    if splits is None:
                        # Include all splits if none specified
                        include_split = True
                    else:
                        # Normalize requested splits for comparison
                        normalized_splits = [normalize_split_name(s) for s in splits]
                        include_split = normalized_split_name in normalized_splits or split_name in splits

                    if include_split:
                        # Use normalized split name for consistency.
                        # Remove existing "split" column if present to avoid duplicates.
                        if "split" in split_ds.column_names:
                            split_ds = split_ds.remove_columns("split")
                        datasets_with_split.append(
                            split_ds.add_column("split", [normalized_split_name] * len(split_ds))
                        )
            else:  # Single Dataset
                split_name = "train"  # Default split name
                ds_add = ds.remove_columns("split") if "split" in ds.column_names else ds
                datasets_with_split.append(
                    ds_add.add_column("split", [split_name] * len(ds_add))
                )

        if not datasets_with_split:
            raise ValueError(f"No data loaded from {dataset_name} with subset(s) {subsets_to_load}")

        # Concatenate all datasets
        combined = concatenate_datasets(datasets_with_split)
        df = combined.to_pandas()

        logger.info(f"Loaded {len(df)} samples from {dataset_name} (subsets: {subsets_to_load})")
        return df

    def create_training_data(
            self,
            model_or_tokenizer: Any,
            split: str = "train",
            class_pair: Optional[Tuple[str, str]] = None,
            batch_size: Optional[int] = None,
            max_size: Optional[int] = None,
            include_other_classes: bool = False,
            balance_column: Optional[str] = "feature_class_id",
            **kwargs,
    ) -> Any:
        """
        Create training dataset from unified data.
        Training uses only rows where transition in {c1→c2, c2→c1} for the configured pair.
        Accepts model_with_gradiend or tokenizer as first argument.

        When max_size is None, uses train_max_size from training_args if set.
        For text prediction, max_size caps samples per feature_class_id (downsampling).
        Note: Balancing happens automatically via dataset scheduler cycling; this parameter
        primarily reduces total dataset size.
        """
        self._ensure_data()
        max_size = self._default_from_training_args(max_size, "train_max_size")
        tokenizer = getattr(model_or_tokenizer, "tokenizer", model_or_tokenizer)
        if self.combined_data is None:
            raise ValueError("No data provided. Set data in config or override create_training_data().")
        if UNIFIED_TRANSITION not in self.combined_data.columns:
            raise ValueError(
                "combined_data must use unified schema (masked, split, factual_class, alternative_class, factual, alternative, transition).")

        normalized_split = normalize_split_name(split)
        split_data = self.combined_data[
            (self.combined_data[UNIFIED_SPLIT].astype(str).str.lower() == split.lower())
            | (self.combined_data[UNIFIED_SPLIT].astype(str).str.lower() == normalized_split.lower())
            ].copy()

        if len(split_data) == 0:
            available = self.combined_data[UNIFIED_SPLIT].unique().tolist()
            raise ValueError(f"No data for split '{split}' (normalized '{normalized_split}'). Available: {available}.")

        if class_pair is None and self.pair is not None:
            class_pair = self.pair
        if class_pair is None:
            trans = split_data[UNIFIED_TRANSITION].unique().tolist()
            if not trans:
                raise ValueError("No transitions in split data.")
            t = trans[0]
            a, _, b = t.partition("→")
            class_pair = (a, b) if (a and b) else (trans[0], trans[0])
        # Set target_classes from the pair (validation via __setattr__)
        self._target_classes = list(class_pair)
        class_pair = tuple(self._target_classes)

        train_transitions = {transition_id(class_pair[0], class_pair[1]), transition_id(class_pair[1], class_pair[0])}
        if include_other_classes and self.all_classes is not None and len(self.all_classes) > 2:
            pair_data = split_data.copy()
        else:
            pair_data = split_data[split_data[UNIFIED_TRANSITION].isin(train_transitions)].copy()

        if len(pair_data) == 0:
            raise ValueError(f"No data for transitions {train_transitions} in split '{split}'.")

        training_pairs = []
        feature_class_id_map = {}
        next_fcid = 0
        add_identity = bool(getattr(getattr(self, "training_args", None), "add_identity_for_other_classes", False))

        def _feature_class_id(src: str, tgt: str) -> int:
            nonlocal next_fcid
            key = (src, tgt)
            if key not in feature_class_id_map:
                feature_class_id_map[key] = next_fcid
                next_fcid += 1
            return feature_class_id_map[key]

        for _, row in pair_data.iterrows():
            src = row[UNIFIED_FACTUAL_CLASS]
            tgt = row[UNIFIED_ALTERNATIVE_CLASS]
            label = 1 if src == class_pair[0] else (-1 if src == class_pair[1] else 0)
            training_pairs.append({
                "masked": row[UNIFIED_MASKED],
                "factual": row[UNIFIED_FACTUAL],
                "alternative": row[UNIFIED_ALTERNATIVE],
                "factual_id": src,
                "alternative_id": tgt,
                "label": label,
                "feature_class_id": _feature_class_id(src, tgt),
            })

        if add_identity and self.all_classes and len(self.all_classes) > 2:
            neutral_classes = [c for c in self.all_classes if c not in class_pair]
            if neutral_classes:
                neutral_data = split_data[split_data[UNIFIED_FACTUAL_CLASS].isin(neutral_classes)].copy()
                for _, row in neutral_data.iterrows():
                    c = row[UNIFIED_FACTUAL_CLASS]
                    training_pairs.append({
                        UNIFIED_MASKED: row[UNIFIED_MASKED],
                        UNIFIED_FACTUAL: row[UNIFIED_FACTUAL],
                        UNIFIED_ALTERNATIVE: row[UNIFIED_FACTUAL],
                        "factual_id": c,
                        "alternative_id": c,
                        "label": 0,
                        "feature_class_id": _feature_class_id(c, c),
                    })
                # Always add identity rows from per-class datasets for neutral classes missing in unified data.
                if getattr(self, "class_datasets", None):
                    split_col_cfg = self.config.split_col
                    masked_col_cfg = self.config.masked_col
                    for c in neutral_classes:
                        if c not in self.class_datasets:
                            continue
                        df_c = self.class_datasets[c]
                        if split_col_cfg not in df_c.columns:
                            subset = df_c
                        else:
                            subset = df_c[
                                (df_c[split_col_cfg].astype(str).str.lower() == split.lower())
                                | (df_c[split_col_cfg].astype(str).str.lower() == normalized_split.lower())
                                ]
                        if len(subset) == 0:
                            continue
                        factual_col = c if c in subset.columns else ("label" if "label" in subset.columns else None)
                        if factual_col is None:
                            continue
                        for _, row in subset.iterrows():
                            training_pairs.append({
                                "masked": row[masked_col_cfg],
                                "factual": row[factual_col],
                                "alternative": row[factual_col],
                                "factual_id": c,
                                "alternative_id": c,
                                "label": 0,
                                "feature_class_id": _feature_class_id(c, c),
                            })
        elif add_identity and not self.all_classes:
            logger.warning(
                "add_identity_for_other_classes is True but classes are not defined; skipping identity augmentation.")

        training_df = pd.DataFrame(training_pairs)

        # Apply max_size if specified: cap per feature_class_id (downsampling)
        # Note: The dataset's balance_column (set below) cycles through groups, ensuring equal
        # representation via oversampling. This downsampling reduces total dataset size but is
        # not strictly necessary for balancing (the scheduler handles that). It's kept for
        # memory/performance when train_max_size is set.
        if max_size is not None and len(training_df) > max_size:
            training_df = training_df.groupby("feature_class_id").apply(
                lambda x: x.sample(
                    min(len(x), max_size),
                    random_state=42,
                )
            ).reset_index(drop=True)

        # Determine if decoder-only model
        is_decoder_only_model = kwargs.get("is_decoder_only_model")
        if is_decoder_only_model is None:
            is_decoder_only_model = tokenizer.mask_token_id is None

        # Get batch_size
        if batch_size is None:
            batch_size = kwargs.get("batch_size", 1)

        # Create text-specific training dataset
        # Note: balance_column enables oversampling via cycling through groups in __getitem__
        return TextTrainingDataset(
            training_df,
            tokenizer,
            batch_size=batch_size,
            is_decoder_only_model=is_decoder_only_model,
            target_key="label",
            balance_column=balance_column,
        )

    def create_gradient_training_dataset(
        self,
        raw_training_data: Any,
        model_with_gradiend: Any,
        *,
        cache_dir: Optional[str] = None,
        use_cached_gradients: bool = False,
        **kwargs,
    ) -> Any:
        """Wrap raw training data into TextGradientTrainingDataset for gradient creation (text modality).
        source and target are resolved from TrainingArguments (override via kwargs if needed).
        """
        source = kwargs.pop("source", None)
        target = kwargs.pop("target", None)
        args = getattr(self, "training_args", None)
        if source is None and args is not None:
            source = None if getattr(args, "supervised_decoder", False) else getattr(args, "source", "factual")
        if source is None:
            source = "factual"
        if target is None and args is not None:
            target = getattr(args, "target", "diff")
        if target is None:
            target = "diff"
        tokenizer = model_with_gradiend.tokenizer
        dtype = kwargs.pop("dtype", model_with_gradiend.gradiend.torch_dtype)
        device = kwargs.pop("device", model_with_gradiend.gradiend.device_encoder)
        return TextGradientTrainingDataset(
            raw_training_data,
            tokenizer,
            model_with_gradiend.gradient_creator,
            source=source,
            target=target,
            cache_dir=cache_dir,
            use_cached_gradients=use_cached_gradients,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def _infer_decoder_eval_targets(self) -> Dict[str, List[str]]:
        """
        Infer decoder evaluation targets from unified data and, when needed, from per-class datasets.
        For each class, collects tokens used as factual (when factual_class=C) and as alternative (when alternative_class=C).
        When combined_data only has the training pair (e.g. single-token-per-class), classes not in the pair
        get no tokens from combined_data; we then supplement from self.class_datasets when available.
        """
        self._ensure_data()
        if self.combined_data is None:
            raise ValueError("No data available to infer decoder eval targets")
        if UNIFIED_TRANSITION not in self.combined_data.columns:
            raise ValueError("combined_data must use unified schema to infer decoder eval targets")

        targets: Dict[str, List[str]] = {}
        for class_name in self.target_classes or []:
            as_src = self.combined_data[self.combined_data[UNIFIED_FACTUAL_CLASS] == class_name]
            as_tgt = self.combined_data[self.combined_data[UNIFIED_ALTERNATIVE_CLASS] == class_name]
            tokens = set(as_src[UNIFIED_FACTUAL].dropna().astype(str)) | set(
                as_tgt[UNIFIED_ALTERNATIVE].dropna().astype(str))
            # When combined_data only has the training pair, other classes have no rows; use per-class data
            if not tokens and getattr(self, "class_datasets", None) and class_name in self.class_datasets:
                df_c = self.class_datasets[class_name]
                factual_col = class_name if class_name in df_c.columns else (
                    "label" if "label" in df_c.columns else None)
                if factual_col is not None:
                    tokens = set(df_c[factual_col].dropna().astype(str))
            targets[class_name] = list(tokens)

        run_id_label = self.run_id if self.run_id is not None else "default"
        logger.info(f"Inferred decoder eval targets for {run_id_label}: {targets}")
        return targets

    def evaluate_base_model(
            self,
            model: Any,
            tokenizer: Any,
            use_cache: Optional[bool] = None,
            cache_folder: str = '',
            model_id: Optional[str] = None,
            training_like_df: Optional[pd.DataFrame] = None,
            neutral_df: Optional[pd.DataFrame] = None,
            max_size_training_like: Optional[int] = None,
            max_size_neutral: Optional[int] = None,
            eval_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model for decoder evaluation using generic feature score + LMS.

        Probabilities are computed from the passed-in model's forward: for causal/decoder
        models this is next-token (CLM) logits; for encoder MLM, mask-position logits. When
        using a decoder-only MLM head, the trainer injects the base CLM so this receives the
        CLM only (never the MLM head).

        Args:
            model: Model used for probability computation (CLM or full MLM)
            tokenizer: Tokenizer
            use_cache: If True (default), use cached results when available.
            cache_folder: Cache folder suffix
            model_id: Model identifier
            training_like_df: Optional cached training-like DataFrame for probability scoring
            neutral_df: Optional cached neutral DataFrame for LMS scoring
            max_size_training_like: Maximum number of generated training-like rows
            max_size_neutral: Maximum number of generated neutral rows (and LMS text cap)
            eval_batch_size: Common eval batch size used for LMS computation

        Returns:
            Dict with 'feature_score' and 'lms' keys
        """
        if hasattr(model, "gradiend") and hasattr(model, "base_model"):
            base = getattr(model, "base_model", None)
            if base is not None:
                model = base

        use_cache = self._default_from_training_args(use_cache, "use_cache", fallback=False)

        cache_file = resolve_decoder_per_model_cache_path(self.experiment_dir, cache_folder=cache_folder)
        if use_cache and cache_file and os.path.isfile(cache_file):
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                return cached

        # Resolve eval datasets (probability and LMS can use different sources).
        if max_size_training_like is None:
            max_size_training_like = self.config.decoder_eval_lms_max_samples
        if max_size_neutral is None:
            max_size_neutral = self.config.decoder_eval_lms_max_samples
        if training_like_df is None or neutral_df is None:
            training_like_df, neutral_df = self._get_decoder_eval_dataframe(
                tokenizer,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                cached_training_like_df=training_like_df,
                cached_neutral_df=neutral_df,
            )

        # Get targets
        targets = self.config.decoder_eval_targets
        if targets is None:
            targets = self._infer_decoder_eval_targets()

        if not targets:
            run_id_part = f" (run_id={self.run_id})" if self.run_id is not None else ""
            raise ValueError(
                "Could not infer decoder eval targets. "
                "Set config.decoder_eval_targets explicitly or ensure data and target_classes are loaded." + run_id_part
            )

        # Restrict to target classes when requested, excluding neutral/other augmenting classes
        if getattr(self.config, "decoder_eval_restrict_to_target_classes", True) and self.target_classes is not None:
            target_classes_set = frozenset(self.target_classes)
            targets = {k: v for k, v in targets.items() if k in target_classes_set}
            if not targets:
                raise ValueError(
                    f"decoder_eval_restrict_to_target_classes=True but no decoder_eval_targets for target_classes {self.target_classes}. "
                    "Ensure target classes exist in decoder_eval_targets or set decoder_eval_restrict_to_target_classes=False."
                )

        # Compute feature score (target token probabilities)
        # Determine dataset class column for grouping by dataset
        dataset_class_col = None
        if "label_class" in training_like_df.columns:
            dataset_class_col = "label_class"
        elif "factual_id" in training_like_df.columns:
            dataset_class_col = "factual_id"
        
        data_class_col = None
        if getattr(self.config, "decoder_eval_prob_on_other_class", True):
            if "alternative_id" in training_like_df.columns:
                data_class_col = "alternative_id"
        
        # Compute probabilities for all classes on all datasets
        probs_by_dataset = evaluate_probability_shift_score(
            model,
            tokenizer,
            targets=targets,
            eval_data_df=training_like_df,
            key_text=self.config.masked_col,
            dataset_class_col=dataset_class_col,
        )

        # Extract selection metrics:
        # - strengthen: P(target_class) on *other* class datasets (counterfactual)
        # - weaken:     P(class) on its own dataset (factual)
        probs_factual: Dict[str, float] = {}
        if data_class_col and dataset_class_col:
            counterfactual_probs = {}
            for class_name in targets.keys():
                other_classes = [c for c in targets.keys() if c != class_name]
                # Strengthen metric: P(class_name) on datasets of other classes
                if other_classes:
                    vals: List[float] = []
                    for other in other_classes:
                        if other in probs_by_dataset:
                            probs_for_other = probs_by_dataset[other]
                            if class_name in probs_for_other:
                                vals.append(float(probs_for_other[class_name]))
                    if vals:
                        counterfactual_probs[class_name] = float(np.mean(vals))
                # Factual metric: P(class_name) on its own dataset (for weaken)
                if class_name in probs_by_dataset and class_name in probs_by_dataset[class_name]:
                    probs_factual[class_name] = float(probs_by_dataset[class_name][class_name])
            probs = counterfactual_probs if counterfactual_probs else next(iter(probs_by_dataset.values())) if probs_by_dataset else {}
        else:
            probs = next(iter(probs_by_dataset.values())) if probs_by_dataset else {}
            if probs_by_dataset:
                for class_name in targets.keys():
                    if class_name in probs_by_dataset and class_name in probs_by_dataset[class_name]:
                        probs_factual[class_name] = float(probs_by_dataset[class_name][class_name])

        # Compute LMS
        ignore_tokens = self.config.decoder_eval_ignore_tokens
        if ignore_tokens is None and getattr(self.config, "eval_neutral_data", None) is None:
            ignore_set = set()
            for tokens in targets.values():
                if tokens:
                    ignore_set.update([t for t in tokens if t])
            ignore_tokens = sorted(ignore_set)
        if ignore_tokens is None:
            ignore_tokens = []
        if eval_batch_size is None:
            eval_batch_size = self._default_from_training_args(
                None,
                "eval_batch_size",
                fallback=32,
            )
        lms = compute_lms(
            model,
            tokenizer,
            neutral_df['text'].tolist(),
            ignore=ignore_tokens,
            max_texts=max_size_neutral,
            batch_size=eval_batch_size,
        )

        # Return in standard format with new structure
        result = {
            'probs': probs,  # Counterfactual probs for selection (P(other) on metric dataset) for strengthen
            'lms': lms,
        }
        # Add probs_by_dataset if available
        if probs_by_dataset is not None:
            result['probs_by_dataset'] = probs_by_dataset
        # Factual probs P(class) on class dataset for weaken selection
        if probs_factual:
            result['probs_factual'] = probs_factual

        # Save cache
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)

        return result

    def analyze_decoder_for_plotting(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        model_with_gradiend: Optional[Any] = None,
        class_ids: Optional[List[str]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Analyze decoder for plotting: extends decoder results with probabilities for all classes
        evaluated on all datasets.

        Args:
            decoder_results: Decoder evaluation result (summary at top level, e.g. result['3SG'], plus 'grid').
                If None, calls evaluate_decoder() to get base results.
            model_with_gradiend: ModelWithGradiend instance. If None, uses self.get_model().
            class_ids: Classes to evaluate probabilities for. If None, uses all_classes if available,
                else target_classes.
            use_cache: Whether to use cached results when re-evaluating.

        Returns:
            Dict with 'plotting_data' (extended grid with probs_by_dataset) and 'summary' (summary entries from decoder_results).
        """
        if decoder_results is None:
            decoder_results = self.evaluate_decoder(use_cache=use_cache, **kwargs)
        
        grid = decoder_results.get("grid", {})
        _reserved = {"grid", "plot_path", "plot_paths"}
        summary = {k: v for k, v in decoder_results.items() if k not in _reserved}
        
        # Determine classes to evaluate
        if class_ids is None:
            class_ids = self.all_classes if self.all_classes else self.target_classes
        
        if not class_ids:
            raise ValueError("No classes specified for plotting analysis. Provide class_ids or ensure all_classes/target_classes are set.")
        
        # Get model and tokenizer
        if model_with_gradiend is None:
            model_with_gradiend = self.get_model()
        
        if model_with_gradiend is None:
            raise ValueError("No model available. Provide model_with_gradiend or ensure model is loaded.")
        
        base_model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        
        # Get training_like_df for evaluation
        training_like_df, neutral_df = self._get_decoder_eval_dataframe(
            tokenizer,
            cached_training_like_df=None,
            cached_neutral_df=None,
        )
        
        # Build extended targets dict for all classes
        targets = self.config.decoder_eval_targets or self._infer_decoder_eval_targets()
        extended_targets = {}
        for cls in class_ids:
            if cls in targets:
                extended_targets[cls] = targets[cls]
            else:
                # Infer tokens for this class
                # For now, try to infer from data - this is a placeholder
                # TODO: implement _infer_tokens_for_class helper
                logger.warning(f"Class {cls} not in decoder_eval_targets, skipping token inference for now")
                continue
        
        if not extended_targets:
            raise ValueError(f"Could not build targets for classes {class_ids}. Ensure decoder_eval_targets includes these classes.")
        
        # Extend grid entries with probs_by_dataset if missing
        extended_grid = {}
        for candidate_id, entry in grid.items():
            extended_entry = dict(entry)  # Copy entry
            
            # Check if probs_by_dataset already exists
            if "probs_by_dataset" not in extended_entry:
                # Need to re-evaluate this candidate
                if candidate_id == "base":
                    # Evaluate base model
                    base_result = self.evaluate_base_model(
                        base_model,
                        tokenizer,
                        training_like_df=training_like_df,
                        neutral_df=neutral_df,
                        use_cache=use_cache,
                    )
                    if "probs_by_dataset" in base_result:
                        extended_entry["probs_by_dataset"] = base_result["probs_by_dataset"]
                else:
                    # Extract feature_factor and lr from candidate_id
                    if isinstance(candidate_id, tuple) and len(candidate_id) == 2:
                        feature_factor, lr = candidate_id
                    elif isinstance(candidate_id, dict):
                        feature_factor = candidate_id.get("feature_factor")
                        lr = candidate_id.get("learning_rate")
                    else:
                        logger.warning(f"Unknown candidate_id format: {candidate_id}, skipping")
                        extended_grid[candidate_id] = extended_entry
                        continue
                    
                    # Create modified model
                    modified_model = model_with_gradiend.rewrite_base_model(
                        learning_rate=lr,
                        feature_factor=feature_factor,
                        part=getattr(self.config, "decoder_eval_part", "decoder"),
                    )
                    
                    # Evaluate modified model
                    modified_result = self.evaluate_base_model(
                        modified_model,
                        tokenizer,
                        training_like_df=training_like_df,
                        neutral_df=neutral_df,
                        use_cache=use_cache,
                    )
                    
                    if "probs_by_dataset" in modified_result:
                        extended_entry["probs_by_dataset"] = modified_result["probs_by_dataset"]
                    
                    del modified_model
                    torch.cuda.empty_cache()
            
            extended_grid[candidate_id] = extended_entry
        
        return {
            "plotting_data": extended_grid,
            "summary": summary,
        }

    def _encode_training_rows(
            self,
            model_with_gradiend: Any,
            train_eval_data: Any,
            source_type: str,
            max_size: Optional[int],
            encoder_kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Encode training data via gradients and return rows with text, encoded, label, type='training'."""
        logger.info("Encoding training data (max_size=%s, source=%s)", max_size, source_type)
        eval_result = self.evaluator.evaluate_encoder(
            eval_data=train_eval_data,
            use_cache=False,
            split=encoder_kwargs["split"],
            max_size=max_size,
            include_other_classes=encoder_kwargs.get("include_other_classes", True) or False,
            source=source_type,
        )
        training_rows = eval_result.get("training_rows") or []
        if not training_rows:
            def _text_row_extractor(entry: dict) -> dict:
                text = entry.get("input_text") or entry.get("text")
                return {"text": text} if text else {}

            training_rows = encode_dataset_to_rows(
                model_with_gradiend,
                train_eval_data,
                row_extractor=_text_row_extractor,
            )

        rows: List[Dict[str, Any]] = []
        for r in training_rows:
            rows.append({
                "text": r.get("text"),
                "encoded": r["encoded"],
                "label": float(r["label"]),
                "source_id": r.get("source_id"),
                "target_id": r.get("target_id"),
                "type": "training",
            })
        logger.debug(f"Processed {len(rows)} training data entries")
        return rows

    def _encode_neutral_training_masked_rows(
            self,
            model_with_gradiend: Any,
            train_eval_data: Any,
            excluded_tokens: List[str],
            factual_token_key: str,
            alternative_token_key: str,
            max_size: Optional[int],
            torch_dtype: Any,
            device: Any,
    ) -> List[Dict[str, Any]]:
        """Encode neutral variant from training templates with re-masked non-target tokens.

        Uses training templates, replaces [MASK] with factual token, then re-masks a random
        non-excluded token. Returns rows with type='neutral_training_masked'.
        """
        # Excluded tokens must always include at least all target tokens from the training data
        target_tokens_from_data = set()
        for entry in train_eval_data:
            for key in (factual_token_key, alternative_token_key):
                if key in entry and entry[key] is not None:
                    target_tokens_from_data.add(str(entry[key]).strip())
        base_excluded = list(excluded_tokens) if excluded_tokens else []
        excluded_for_masked = list(set(base_excluded) | target_tokens_from_data)

        logger.info("Encoding neutral training masked data (max_size=%s)", max_size)
        if excluded_for_masked:
            logger.debug(f"Excluded tokens: {excluded_for_masked[:10]}..." if len(
                excluded_for_masked) > 10 else f"Excluded tokens: {excluded_for_masked}")

        tokenizer = model_with_gradiend.tokenizer
        is_decoder_only_model = is_decoder_only_model_from_obj(tokenizer)
        mask_token = tokenizer.mask_token if not is_decoder_only_model else None
        logger.debug(f"Tokenizer: is_decoder_only_model={is_decoder_only_model}, mask_token={mask_token}")

        # Collect training data entries and re-mask non-target tokens
        neutral_training_masked_pairs = []
        random.seed(42)  # For reproducibility
        neutral_training_masked_count = 0
        for i, entry in enumerate(train_eval_data):
            if max_size and i >= max_size:
                break

            # Get original masked template (TextTrainingDataset provides "template" with [MASK] placeholder)
            template = entry["template"]

            # Reconstruct unmasked text by replacing [MASK] with the actual factual token
            factual_token = entry[factual_token_key]
            if i == 0:
                logger.debug(
                    f"Sample neutral training masked entry: template={template[:50]}..., "
                    f"factual_token_key={factual_token_key}, factual_token={factual_token}"
                )
            unmasked_text = template.replace("[MASK]", factual_token)

            # For MLM models: mask one random token that is NOT in excluded_tokens
            if not is_decoder_only_model and mask_token:
                tokens = tokenizer.tokenize(unmasked_text)
                if not tokens:
                    continue

                # Filter out special tokens and excluded tokens
                excluded_norm = {
                    str(excl).lower().lstrip("##").lstrip("Ġ").lstrip("▁")
                    for excl in excluded_for_masked
                } if excluded_for_masked else set()
                valid_indices = []
                for idx, token in enumerate(tokens):
                    # Skip special tokens
                    if token.startswith('[') and token.endswith(']'):
                        continue
                    # Skip excluded tokens (target tokens)
                    token_norm = token.lower().lstrip("##").lstrip("Ġ").lstrip("▁")
                    if excluded_norm and token_norm in excluded_norm:
                        continue
                    valid_indices.append(idx)

                if not valid_indices:
                    valid_indices = [
                        idx for idx, token in enumerate(tokens)
                        if not (token.startswith('[') and token.endswith(']'))
                    ]
                if not valid_indices:
                    continue

                # Mask one random non-target token
                mask_idx = random.choice(valid_indices)
                original_token = tokens[mask_idx]  # Token at mask position (neutral)
                tokens[mask_idx] = mask_token
                masked_text = tokenizer.convert_tokens_to_string(tokens)
            else:
                # For decoder-only models, use unmasked text as-is
                masked_text = unmasked_text
                original_token = factual_token  # Use factual as neutral token

            # Build pair for TextTrainingDataset; encoding is done once when iterating gradient_data
            neutral_training_masked_pairs.append({
                UNIFIED_MASKED: masked_text,
                UNIFIED_FACTUAL: original_token,
                UNIFIED_ALTERNATIVE: original_token,
                "factual_id": "neutral",
                "alternative_id": "neutral",
                "label": 0,
                "feature_class_id": 0,
            })

        rows: List[Dict[str, Any]] = []
        if neutral_training_masked_pairs:
            neutral_training_masked_df = pd.DataFrame(neutral_training_masked_pairs)
            neutral_training_masked_dataset = TextTrainingDataset(
                neutral_training_masked_df,
                tokenizer,
                batch_size=1,
                is_decoder_only_model=is_decoder_only_model,
                target_key="label",
                balance_column="feature_class_id",
            )

            # Create TextGradientTrainingDataset for encoding
            neutral_training_masked_gradient_data = TextGradientTrainingDataset(
                neutral_training_masked_dataset,
                tokenizer,
                model_with_gradiend.gradient_creator,
                source="factual",
                target=None,
                dtype=torch_dtype,
                device=device,
            )

            # Encode neutral training masked data
            neutral_ctr = 0
            for i, entry in enumerate(neutral_training_masked_gradient_data):
                if max_size and neutral_ctr >= max_size:
                    break
                neutral_ctr += 1

                grad = entry["source"]
                text = entry["text"]
                encoded_value = model_with_gradiend.encode(grad, return_float=True)
                rows.append({
                    'text': text,
                    'encoded': encoded_value,
                    'label': 0.0,
                    'source_id': "neutral",
                    'target_id': "neutral",
                    'type': 'neutral_training_masked',
                })
                neutral_training_masked_count += 1

        logger.debug(f"Processed {neutral_training_masked_count} neutral training masked entries")
        return rows

    def _encode_neutral_dataset_rows(
            self,
            model_with_gradiend: Any,
            neutral_data_df: Optional[pd.DataFrame],
            encoder_kwargs: Dict[str, Any],
            masked_col_name: str,
            excluded_tokens: List[str],
            max_size: Optional[int],
            torch_dtype: Any,
            device: Any,
    ) -> List[Dict[str, Any]]:
        if neutral_data_df is None or len(neutral_data_df) == 0:
            return []

        logger.info("Encoding neutral dataset data (%s samples)", len(neutral_data_df))
        logger.debug(f"neutral_data_df columns: {list(neutral_data_df.columns)}")

        tokenizer = model_with_gradiend.tokenizer
        is_decoder_only_model = is_decoder_only_model_from_obj(tokenizer)
        mask_token = tokenizer.mask_token if not is_decoder_only_model else None

        # Prepare neutral data: create masked texts with one mask per entry
        # Use provided text_col, or fall back to masked_col_name, or try common defaults
        neutral_text_col = encoder_kwargs.get("text_col") or masked_col_name
        logger.debug(
            f"Trying to use text column: '{neutral_text_col}' (from text_col={encoder_kwargs.get('text_col')}, masked_col={masked_col_name}, training_args.masked_col={self.config.masked_col})")
        if neutral_text_col not in neutral_data_df.columns:
            # Try common alternatives
            if 'text' in neutral_data_df.columns:
                neutral_text_col = 'text'
                logger.debug(f"Column '{neutral_text_col}' not found, falling back to 'text'")
            elif 'masked' in neutral_data_df.columns:
                neutral_text_col = 'masked'
                logger.debug(f"Column '{neutral_text_col}' not found, falling back to 'masked'")
            else:
                raise ValueError(
                    f"neutral_data_df must have '{neutral_text_col}' (from training_args.masked_col), "
                    f"'text', or 'masked' column. Available columns: {list(neutral_data_df.columns)}"
                )
        logger.debug(f"Using text column: '{neutral_text_col}' for neutral dataset")

        # Create neutral eval pairs with one mask per entry
        neutral_pairs = []
        random.seed(42)  # For reproducibility
        neutral_dataset_count = 0
        if max_size:
            neutral_data_df = neutral_data_df.sample(n=max_size, random_state=42).reset_index(drop=True)
        for idx, row in neutral_data_df.iterrows():
            text = str(row[neutral_text_col])
            if idx == 0:
                logger.debug(f"Sample neutral dataset entry: text={text[:50] if text else 'None'}...")
            pair = create_masked_pair_from_text(
                text,
                tokenizer,
                is_decoder_only_model,
                excluded_tokens=excluded_tokens,
                mask_token=mask_token,
                min_prefix_tokens=5,
            )
            if pair is None:
                continue
            masked_text, neutral_token = pair
            neutral_pairs.append({
                UNIFIED_MASKED: masked_text,
                UNIFIED_FACTUAL: neutral_token,  # Use actual token (not empty) for neutral
                UNIFIED_ALTERNATIVE: neutral_token,  # Same token for both (makes diff=0 but factual non-zero)
                "factual_id": "neutral",
                "alternative_id": "neutral",
                "label": 0,
                "feature_class_id": 0,
            })
            neutral_dataset_count += 1

        logger.debug(f"Created {neutral_dataset_count} neutral pairs from {len(neutral_data_df)} rows")
        if not neutral_pairs:
            logger.warning("No valid neutral data entries after masking")
            return []

        neutral_df = pd.DataFrame(neutral_pairs)

        # Create TextTrainingDataset for neutral data
        neutral_dataset = TextTrainingDataset(
            neutral_df,
            tokenizer,
            batch_size=1,
            is_decoder_only_model=is_decoder_only_model,
            target_key="label",
            balance_column="feature_class_id",
        )

        # Create TextGradientTrainingDataset for encoding
        neutral_gradient_data = TextGradientTrainingDataset(
            neutral_dataset,
            tokenizer,
            model_with_gradiend.gradient_creator,
            source="factual",
            target=None,
            dtype=torch_dtype,
            device=device,
        )

        rows: List[Dict[str, Any]] = []
        neutral_encoded_count = 0
        for i, entry in enumerate(neutral_gradient_data):

            grad = entry["source"]
            label = 0.0
            input_text = entry["input_text"]
            if i == 0:
                logger.debug(
                    f"Sample neutral dataset encoded entry: input_text={input_text[:50] if input_text else 'None'}..., "
                    f"label={label}, available keys: {list(entry.keys())}")
            encoded_value = model_with_gradiend.encode(grad, return_float=True)
            rows.append({
                'text': input_text,
                'encoded': encoded_value,
                'label': float(label),
                'source_id': 'neutral',
                'target_id': 'neutral',
                'type': 'neutral_dataset',
            })
            neutral_encoded_count += 1

        logger.debug(f"Encoded {neutral_encoded_count} neutral dataset entries")
        return rows

    def get_decoder_mlm_training_data(
            self,
            split: str = "train",
            max_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get (masked, label) DataFrame for training a decoder-only MLM head.

        When trainer has per-class data (class_datasets), uses *all* subsets of the
        dataset for the given split—including neutral and every class—so the MLM
        head sees the full HuggingFace dataset. When only combined_data is
        available (e.g. single HF dataset), uses combined_data for the split.
        Returned DataFrame has columns 'masked' and 'label'; masked must contain
        [MASK], label must be a single token per row. Target token IDs are derived
        from unique values in 'label'.
        """
        self._ensure_data()
        if getattr(self, "class_datasets", None) is not None:
            out = all_subsets_to_mlm_df(
                self.class_datasets,
                split=split,
                masked_col=self.config.masked_col,
                split_col=self.config.split_col,
                use_class_names_as_columns=getattr(
                    self.config, "use_class_names_as_columns", True
                ),
            )
        else:
            if self.combined_data is None:
                raise ValueError("combined_data is None; trainer has no data.")
            if UNIFIED_MASKED not in self.combined_data.columns or UNIFIED_FACTUAL not in self.combined_data.columns:
                raise ValueError(
                    "combined_data must have unified columns for masked and factual. "
                    "Use a trainer that produces (masked, label) per row."
                )
            from gradiend.util import normalize_split_name
            sn = normalize_split_name(split)
            split_data = self.combined_data[
                (self.combined_data[UNIFIED_SPLIT].astype(str).str.lower() == split.lower())
                | (self.combined_data[UNIFIED_SPLIT].astype(str).str.lower() == sn)
                ]
            if split_data.empty:
                available = self.combined_data[UNIFIED_SPLIT].unique().tolist()
                raise ValueError(f"No data for split '{split}'. Available: {available}")
            out = split_data[[UNIFIED_MASKED, UNIFIED_FACTUAL]].copy()
            out = out.rename(columns={UNIFIED_MASKED: "masked", UNIFIED_FACTUAL: "label"})
        if max_size is not None:
            out = out.groupby('label').apply(
                lambda x: x.sample(
                    min(len(x), max_size),
                    random_state=42,
                )
            ).reset_index(drop=True)
        return out

    def train_decoder_only_mlm_head(
            self,
            model: Union[str, Any],
            output: Optional[str] = None,
            *,
            split: str = "train",
            batch_size: int = 4,
            epochs: int = 5,
            lr: float = 1e-4,
            pooling_length: int = 3,
            max_length: int = 128,
            max_size: Optional[int] = None,
            use_cache: Optional[bool] = None,
            model_use_cache: Optional[bool] = None,
    ) -> str:
        """
        Train a custom MLM head on a decoder-only model. DecoderModelWithMLMHead is a
        drop-in replacement for AutoModelForMaskedLM: loading (e.g. trainer.train())
        automatically uses this path when you pass the base model name (e.g. 'gpt2').

        Use when the target token comes after the mask (e.g. German DE: article
        before noun). The base model (e.g. gpt2) is frozen; only a small classifier head
        is trained to predict the token at the [MASK] position.

        Args:
            model: Base model name or model instance (e.g. 'gpt2', 'meta-llama/Llama-3.2-3B').
            output: Output directory for the saved MLM head. If None, uses
                experiment_dir/cache/decoder_mlm_head when experiment_dir is set.
            split: Dataset split for training (e.g. 'train', 'validation'). Default: 'train'.
            batch_size: Batch size for training. Default: 4.
            epochs: Number of training epochs. Default: 5.
            lr: Learning rate. Default: 1e-4.
            pooling_length: Length of pooling window for the MLM head (context around mask
                position). Default: 3.
            max_length: Maximum sequence length for tokenization. Default: 128.
            max_size: If set, limit training data to this many rows (for faster debugging/trials).
            use_cache: If True, skip training when model already exists at output path.
                Defaults to training args use_cache (fallback False).
            model_use_cache: If False, disable KV cache in model forward (recommended for training).
                Defaults to training args model_use_cache (fallback False). Manual override supported.

        Returns:
            Path (str) to the saved MLM-head model. trainer.train() resolves to this path
            automatically when it exists.
        """
        if output is None:
            experiment_dir = self.experiment_dir
            output = resolve_decoder_mlm_head_dir(experiment_dir) if experiment_dir else None
        if output is None:
            raise ValueError(
                "Output path required for decoder-only MLM head. "
                "Set experiment_dir on TrainingArguments or pass output= explicitly."
            )
        use_cache = self._default_from_training_args(use_cache, "use_cache", fallback=False)
        model_use_cache = self._default_from_training_args(
            model_use_cache, "model_use_cache", fallback=False
        )

        if use_cache and os.path.exists(output) and os.path.exists(os.path.join(output, "training_args.json")):
            logger.info(
                f"Decoder-only MLM head already exists at {output}, skipping training. Use use_cache=False to retrain."
            )
            return output
        train_df = self.get_decoder_mlm_training_data(split=split, max_size=max_size)
        base_model_name = model if isinstance(model, str) else getattr(model, "name_or_path", model)
        trust_remote_code = getattr(getattr(self, "_training_args", None), "trust_remote_code", False)
        train_mlm_head(
            base_model=base_model_name,
            train_df=train_df,
            output_path=output,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            pooling_length=pooling_length,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            use_cache=model_use_cache,
        )
        return output

    def _get_decoder_eval_dataframe(
        self,
        tokenizer: Any,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        cached_training_like_df: Optional[pd.DataFrame] = None,
        cached_neutral_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get DataFrame for decoder evaluation (test split).

        Args:
            tokenizer: Tokenizer
            max_size_training_like: Maximum number of generated training-like samples
            max_size_neutral: Maximum number of generated neutral samples
            cached_training_like_df: Optional cached training-like DataFrame to reuse
            cached_neutral_df: Optional cached neutral DataFrame to reuse

        Returns:
            Tuple (training_like_df, neutral_df)
        """
        training_like_df = cached_training_like_df
        neutral_df = cached_neutral_df

        if training_like_df is None:
            eval_dataset = self.create_training_data(
                tokenizer,
                split='test',
                batch_size=1,
                max_size=max_size_training_like,
            )

            # Extract DataFrame from dataset
            if hasattr(eval_dataset, 'data'):
                training_like_df = eval_dataset.data.copy()
                # Ensure 'masked' column exists (it should from create_training_data)
                if 'masked' not in training_like_df.columns:
                    raise ValueError(
                        f"Dataset DataFrame missing 'masked' column. Available: {list(training_like_df.columns)}"
                    )
            else:
                # Fallback: create from dataset
                rows = []
                for i in range(min(len(eval_dataset), max_size_training_like or len(eval_dataset))):
                    item = eval_dataset[i]
                    template = item["template"]
                    rows.append({
                        UNIFIED_MASKED: template,
                        'text': item["text"],
                    })
                training_like_df = pd.DataFrame(rows)

        if neutral_df is None:
            resolved_neutral_df = resolve_dataframe(
                self.config.eval_neutral_data,
                max_rows=self.config.eval_neutral_max_rows,
            )
            if resolved_neutral_df is not None and len(resolved_neutral_df) > 0:
                neutral_df = resolved_neutral_df.copy()
                if max_size_neutral:
                    neutral_df = neutral_df.sample(
                        n=min(len(neutral_df), max_size_neutral), random_state=42
                    ).reset_index(drop=True)
            else:
                neutral_df = training_like_df.copy()

        neutral_df = self._ensure_decoder_eval_text_columns(neutral_df, tokenizer)

        return training_like_df.reset_index(drop=True), neutral_df.reset_index(drop=True)

    def _ensure_decoder_eval_text_columns(self, df: pd.DataFrame, tokenizer: Any) -> pd.DataFrame:
        """Ensure DataFrame has 'masked' and 'text' columns for decoder evaluation."""
        if "masked" not in df.columns and "text" in df.columns:
            df["masked"] = df["text"]

        if "text" not in df.columns or df["text"].isnull().any():
            mask_token = getattr(tokenizer, "mask_token", None)

            def _label_for_row(row: pd.Series) -> Optional[str]:
                if "factual" in row and pd.notna(row["factual"]):
                    return str(row["factual"])
                if "label" in row and isinstance(row["label"], str):
                    return row["label"]
                if "alternative" in row and pd.notna(row["alternative"]):
                    return str(row["alternative"])
                return None

            def _fill_text(row: pd.Series) -> str:
                masked = str(row.get("masked", ""))
                label = _label_for_row(row)
                if label and mask_token and mask_token in masked:
                    return masked.replace(mask_token, label, 1)
                return masked

            df["text"] = df.apply(_fill_text, axis=1)

        return df

    def _get_decoder_eval_targets(self) -> Dict[str, List[str]]:
        """Get decoder eval targets (delegates to _infer_decoder_eval_targets)."""
        return self._infer_decoder_eval_targets()

    def _analyze_encoder(
        self,
        model_with_gradiend: Optional[Any] = None,
        split: str = "test",
        neutral_data_df: Optional[pd.DataFrame] = None,
        max_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        plot: bool = False,
        include_other_classes: Optional[bool] = True,
        # Column name overrides (defaults to training_args values)
        text_col: Optional[str] = None,
        masked_col: Optional[str] = None,
        factual_token_col: Optional[str] = None,
        alternative_token_col: Optional[str] = None,
        source_id_col: Optional[str] = None,
        target_id_col: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Analyze encoder by encoding gradients from training data and optional neutral data.

        This method processes all variants in a single call:
        1. Training data (always processed)
        2. Neutral variant 1 (if decoder_eval_targets configured)
        3. Neutral variant 2 (if neutral_data_df provided)

        This method handles caching. If cached data exists and use_cache=True, it is loaded and returned.
        Otherwise, the analysis is performed and results are cached.

        Args:
            model_with_gradiend: ModelWithGradiend instance
            split: Dataset split to use
            neutral_data_df: Optional DataFrame with neutral examples (variant 2)
            max_size: Maximum number of samples per variant to encode
            use_cache: If True, use cached encoder analysis when available.
            plot: If True, create encoder distribution plot from analyzed data.
            include_other_classes: If True, include other classes in analysis
            text_col: Column name for text in neutral_data_df (defaults to training_args.masked_col)
            masked_col: Column name for masked text (defaults to training_args.masked_col)
            factual_token_col: Key name for factual token in entries (defaults to "factual_token")
            alternative_token_col: Key name for alternative token in entries (defaults to "alternative_token")
            source_id_col: Key name for source class ID in entries (defaults to "factual_id")
            target_id_col: Key name for target class ID in entries (defaults to "alternative_id")
            **kwargs: Additional arguments passed to create_eval_data

        Returns:
            DataFrame with columns: text, encoded, label, source_id, target_id, type, ...
            The 'type' column indicates the variant: 'training', 'neutral_training_masked', or 'neutral_dataset'
        """

        use_cache = self._default_from_training_args(use_cache, "use_cache", fallback=False)
        if max_size is None:
            max_size = self._default_from_training_args(max_size, "encoder_eval_max_size")
        if model_with_gradiend is None:
            model_with_gradiend = self.get_model()

        # Single encoder_kwargs dict: same keys used for cache path and for logic.
        # Pass the same dict to get_encodings/evaluate_encoder so the cache path matches.
        encoder_kwargs = dict(
            split=split,
            neutral_data_df=neutral_data_df,
            max_size=max_size,
            include_other_classes=include_other_classes,
            text_col=text_col,
            masked_col=masked_col,
            factual_token_col=factual_token_col,
            alternative_token_col=alternative_token_col,
            source_id_col=source_id_col,
            target_id_col=target_id_col,
            **kwargs,
        )

        # Get neutral data if not provided (config first, then training_args); resolve HF id to DataFrame
        neutral_max_rows = getattr(self.config, "eval_neutral_max_rows", None)
        if encoder_kwargs["neutral_data_df"] is None:
            encoder_kwargs["neutral_data_df"] = resolve_dataframe(
                getattr(self.config, "eval_neutral_data", None),
                max_rows=neutral_max_rows,
            )
        if encoder_kwargs["neutral_data_df"] is None and getattr(self, "training_args", None) is not None:
            encoder_kwargs["neutral_data_df"] = resolve_dataframe(
                getattr(self.training_args, "eval_neutral_data", None),
                max_rows=neutral_max_rows,
            )
        neutral_data_df = encoder_kwargs["neutral_data_df"]

        # Excluded tokens for neutral variant 1 (training_args-derived)
        excluded_tokens = None
        if hasattr(self, "training_args") and hasattr(self.config, "decoder_eval_targets"):
            excluded_tokens = self.config.decoder_eval_targets

        output_path = self._encoder_cache_path(model_with_gradiend.name_or_path, **encoder_kwargs)

        # Try to load cached data
        if use_cache and output_path is not None and os.path.exists(output_path):
            logger.info("Using cached encoder analysis")
            df_cached = pd.read_csv(output_path)
            # Check if cached data has the required 'type' column with correct values
            if 'type' not in df_cached.columns:
                logger.warning(
                    f"Cached data missing 'type' column. This may be from old code. "
                    f"Recomputing with use_cache=False to ensure correct structure."
                )
                use_cache = False  # Recompute
            elif not set(df_cached['type'].unique()).issubset(
                    {'training', 'neutral_training_masked', 'neutral_dataset'}):
                logger.warning(
                    f"Cached data has unexpected 'type' values: {df_cached['type'].unique()}. "
                    f"Recomputing with use_cache=False to ensure correct structure."
                )
                use_cache = False  # Recompute
            else:
                return df_cached

        if not use_cache and output_path is not None and os.path.exists(output_path):
            logger.info(f"Recomputing encoder analysis (removing old cache: {output_path})")
            os.remove(output_path)

        training_config = model_with_gradiend.gradiend.kwargs.get('training', {}).get('training_args', {})
        source_type = training_config.get('source', 'factual')

        # create_eval_data only accepts split, source, max_size, include_other_classes, etc.
        # Do not pass column-override or other encoder-only kwargs (text_col, masked_col, ...).
        train_eval_data = self.create_eval_data(
            model_with_gradiend,
            split=encoder_kwargs["split"],
            source=source_type,
            max_size=encoder_kwargs.get("max_size"),
            include_other_classes=encoder_kwargs.get("include_other_classes", True) or False,
        )
        max_size = encoder_kwargs.get("max_size")

        # Perform analysis (will cache automatically if output_path is set)
        logger.info("Computing encoder analysis for all variants")

        # Determine column names from encoder_kwargs or training_args defaults
        masked_col_name = encoder_kwargs.get("masked_col") or self.config.masked_col
        factual_token_key = encoder_kwargs.get("factual_token_col") or "factual_token"
        alternative_token_key = encoder_kwargs.get("alternative_token_col") or "alternative_token"
        source_id_key = encoder_kwargs.get("source_id_col") or "factual_id"
        target_id_key = encoder_kwargs.get("target_id_col") or "alternative_id"

        logger.debug(f"Using column names: masked_col={masked_col_name}, factual_token_key={factual_token_key}, "
                     f"alternative_token_key={alternative_token_key}, source_id_key={source_id_key}, "
                     f"target_id_key={target_id_key}")

        torch_dtype = model_with_gradiend.gradiend.torch_dtype
        device = model_with_gradiend.gradiend.device_encoder
        rows = []

        # Process excluded_tokens
        if excluded_tokens is None:
            excluded_tokens = []
        if isinstance(excluded_tokens, dict):
            # Flatten dict of lists to single list
            excluded_tokens = [token for tokens in excluded_tokens.values() for token in tokens]

        # 1. Training data: use general EncoderEvaluator path (same as Trainer.evaluate_encoder)
        training_rows = self._encode_training_rows(
            model_with_gradiend,
            train_eval_data,
            source_type,
            max_size,
            encoder_kwargs,
        )
        rows.extend(training_rows)
        training_count = len(training_rows)

        # 2. Neutral training masked data (always computed when we have training data)
        neutral_training_masked_rows = self._encode_neutral_training_masked_rows(
            model_with_gradiend,
            train_eval_data,
            excluded_tokens,
            factual_token_key,
            alternative_token_key,
            max_size,
            torch_dtype,
            device,
        )
        rows.extend(neutral_training_masked_rows)
        neutral_training_masked_count = len(neutral_training_masked_rows)

        # 3. Neutral dataset data if provided
        neutral_dataset_rows = self._encode_neutral_dataset_rows(
            model_with_gradiend,
            neutral_data_df,
            encoder_kwargs,
            masked_col_name,
            excluded_tokens,
            max_size,
            torch_dtype,
            device,
        )
        rows.extend(neutral_dataset_rows)
        neutral_encoded_count = len(neutral_dataset_rows)

        logger.debug(f"Total rows collected: {len(rows)} (training: {training_count}, "
                     f"neutral_training_masked: {neutral_training_masked_count}, "
                     f"neutral_dataset: {neutral_encoded_count if neutral_data_df is not None and len(neutral_data_df) > 0 else 0})")
        if not rows:
            raise ValueError("No data to encode")

        df = pd.DataFrame(rows)

        # Save to cache
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            invalidate_encoder_metrics_cache(output_path)
            logger.info(f"Saved encoder analysis results to {output_path}")

        if plot:
            self.plot_encoder_distributions(encoder_df=df)

        return df
