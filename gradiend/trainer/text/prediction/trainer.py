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
from typing import Dict, List, Optional, Union, Any, Tuple, Type, Sequence, Literal

from gradiend.visualizer.plot_delegation import see_implementation

from gradiend.util.tqdm_utils import gradiend_tqdm

from gradiend.data.core import SplitGroupKey, SplitRatiosInput, normalize_split_ratios, resplit_unified_dataframe
from gradiend.trainer.trainer import Trainer, _apply_seed
from gradiend.trainer.config import TrainerConfig
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.model import ModelWithGradiend
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset, create_masked_pair_from_text
from pathlib import Path
from gradiend.trainer.core.unified_data import (
    all_subsets_to_mlm_df,
    apply_class_merge_to_merged_df,
    merged_to_unified,
    merge_per_class_dfs,
    per_class_dict_to_unified,
    resolve_dataframe,
    resolve_training_data_path,
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
from gradiend.trainer.text.prediction.decoder_only_mlm import load_decoder_mlm_head_meta, train_mlm_head
from gradiend.util.logging import get_logger
from gradiend.model.utils import is_decoder_only_model as is_decoder_only_model_from_obj, is_seq2seq_model
from gradiend.util import normalize_split_name
from gradiend.util.encoder_splits import EncoderSplit, encoder_split_cache_key, order_split_names, resolve_encoder_splits
from gradiend.util.split_policy import SplitPolicy, validate_data_split_policy, vocabulary_held_out_viable_for_target_pair
from gradiend.visualizer.encoder_neutral import neutral_encoder_row_metadata
from gradiend.trainer.core.transition_selection import expand_transition_selection, TransitionSpec
from gradiend.util.paths import (
    resolve_encoder_analysis_path,
    resolve_decoder_per_model_cache_path,
    resolve_decoder_row_wise_csv_path,
    resolve_decoder_mlm_head_dir,
    has_saved_decoder_mlm_head,
)
from gradiend.util.encoding_rows import encode_dataset_to_rows, gradient_entry_to_encoder_row
from gradiend.util.paths import invalidate_encoder_metrics_cache
from gradiend.trainer.text.prediction.prediction_objective import (
    resolve_prediction_objective,
    should_use_decoder_mlm_head_for_auto,
)
from gradiend.trainer.text.prediction.seq2seq import tokenize_prediction_label
from gradiend.trainer.text.common.dataset import TextGradientTrainingDataset

logger = get_logger(__name__)

_SPLIT_COL_UNSET = object()


def _sample_up_to_per_group(df: pd.DataFrame, group_col: str, max_size: int, seed: int) -> pd.DataFrame:
    sampled_groups = [
        group.sample(min(len(group), max_size), random_state=seed)
        for _, group in df.groupby(group_col)
    ]
    if not sampled_groups:
        return df.iloc[0:0].copy()
    return pd.concat(sampled_groups).reset_index(drop=True)


def _normalize_mlm_label_strings(series: pd.Series) -> List[str]:
    return sorted({str(v).strip() for v in series.dropna().tolist() if str(v).strip()})


def _ensure_mlm_training_label_coverage(
    train_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    required_labels: Sequence[str],
) -> pd.DataFrame:
    """Ensure each required label appears at least once in train_df (MLM-head class indices)."""
    if not required_labels or train_df.empty or coverage_df.empty:
        return train_df
    present = set(train_df["label"].astype(str).str.strip())
    missing = [lab for lab in required_labels if lab not in present]
    if not missing:
        return train_df
    coverage = coverage_df.copy()
    coverage["label"] = coverage["label"].astype(str).str.strip()
    extra_rows = []
    for lab in missing:
        matches = coverage.loc[coverage["label"] == lab]
        if matches.empty:
            logger.warning(
                "Decoder MLM-head label %r has no (masked, label) row in data; "
                "encoder/decoder eval may fail for that token.",
                lab,
            )
            continue
        extra_rows.append(matches.iloc[[0]])
    if not extra_rows:
        return train_df
    extra = pd.concat(extra_rows, ignore_index=True)
    logger.info(
        "Decoder MLM-head: added %d row(s) covering %d label(s) absent from the requested split sample.",
        len(extra),
        len(missing),
    )
    return pd.concat([train_df, extra], ignore_index=True)


def _generated_incomplete_data_path(path: Union[str, Path]) -> Path:
    """Return the sidecar path written by TextPredictionDataCreator for incomplete classes."""
    p = Path(path)
    if p.suffix:
        return p.with_name(f"{p.stem}_incomplete_classes{p.suffix}")
    return p.with_name(f"{p.name}_incomplete_classes")


def _load_generated_incomplete_classes(path: Union[str, Path], label_class_col: str = "label_class") -> List[str]:
    sidecar = _generated_incomplete_data_path(path)
    if not sidecar.exists() or not sidecar.is_file():
        return []
    try:
        if sidecar.suffix.lower() == ".parquet":
            df = pd.read_parquet(sidecar)
        else:
            df = pd.read_csv(sidecar)
    except Exception as e:
        logger.warning("Could not read generated incomplete-class sidecar %s: %s", sidecar, e)
        return []
    if label_class_col not in df.columns:
        return []
    return sorted({str(value) for value in df[label_class_col].dropna().unique().tolist()})


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
    dataset_trust_remote_code: Optional[bool] = None
    """Optional trust_remote_code value for HuggingFace datasets.load_dataset.
    None means do not pass the keyword. This is separate from
    TrainingArguments.trust_remote_code, which applies to models/tokenizers."""
    target_classes: Optional[List[str]] = None
    """Target classes for training. Pair is automatically inferred when len(target_classes) == 2."""
    
    all_classes: Optional[List[str]] = None
    """All classes available in the dataset. If None (default), inferred from data. 
    When loading from HuggingFace datasets, None means load all configs/subsets."""


    # Column names (merged format)
    masked_col: str = "masked"
    label_col: str = "label"
    label_class_col: str = "label_class"
    split_col: Optional[str] = "split"
    """Dataset split column. ``\"split\"`` (default) uses row-level splits from the data.
    ``None`` assigns vocabulary-held-out splits by factual token (per target class).
    When omitted at trainer construction, vocabulary-held-out is auto-selected only if each
    target class has >=10 rows and enough distinct factual tokens for the configured ratios;
    otherwise existing random row splits are kept."""
    split_group_col: Optional[str] = None
    """Unified column for vocabulary-held-out resplit (default: factual token column)."""
    split_group_key: SplitGroupKey = None
    """Callable or sequence applied to split_group_col before grouping (e.g. [str.strip, str.casefold])."""
    split_ratios: Optional[SplitRatiosInput] = None
    """Train/validation/test fractions as ``(train, val, test)`` or ``{train, validation, test}``."""
    split_train_ratio: float = 0.8
    split_val_ratio: float = 0.1
    split_test_ratio: float = 0.1
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
    decoder_eval_targets: Optional[Union[Dict[str, List[str]], str]] = None
    # None = infer from data (overlap → row-wise); "label" = row-wise P(factual) vs P(alternative) per row;
    # dict = class-based static targets {class_name: [tokens]}.

    decoder_eval_restrict_to_target_classes: bool = True
    # If True, decoder analysis uses only the target classes (excludes neutral/identity augmenting classes).
    # Set False to evaluate over all classes in decoder_eval_targets.

    decoder_eval_prob_on_other_class: bool = True
    # If True, each target's probability is evaluated on the other class's data only (e.g. P(target1) on target2 rows).
    # Requires eval data to have alternative_id (or data_class_col).

    decoder_eval_export_row_wise_csv: bool = False
    # If True and row-wise decoder eval is used (decoder_eval_targets="label" or default with overlap),
    # export full per-row scores to experiment_dir/decoder_row_wise_scores.csv (masked, factual, alternative,
    # factual_id, alternative_id, P_factual, P_alternative, etc.). Default False.

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

    eval_neutral_additional_excluded_words: Optional[List[str]] = None
    # Extra tokens that must not be mask targets for neutral encoder variants
    # (``neutral_training_masked``, ``neutral_dataset``). Mirrors
    # ``generate_neutral_data(additional_excluded_words=...)``.

    eval_neutral_max_rows: Optional[int] = None
    # Optional cap on number of rows loaded from neutral HF datasets.

    def __post_init__(self) -> None:
        """Normalize split-ratio inputs after dataclass initialization."""
        train, val, test = normalize_split_ratios(
            self.split_ratios,
            train=self.split_train_ratio,
            val=self.split_val_ratio,
            test=self.split_test_ratio,
        )
        object.__setattr__(self, "split_train_ratio", train)
        object.__setattr__(self, "split_val_ratio", val)
        object.__setattr__(self, "split_test_ratio", test)

    def __str__(self) -> str:
        return (
            f"TextPredictionConfig(img_format={self.img_format!r}, target_classes={self.target_classes!r}, "
            f"masked_col={self.masked_col!r}, n_features={self.n_features})"
        )


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
        if not should_use_decoder_mlm_head_for_auto(self):
            return None
        experiment_dir = self.experiment_dir
        if experiment_dir is None:
            return None
        return resolve_decoder_mlm_head_dir(experiment_dir)

    def _prediction_objective(self, model_or_tokenizer: Any = None):
        return resolve_prediction_objective(self, model_or_tokenizer)

    def _train(
        self,
        output_dir: str,
        args: Any,
        model: Any,
        model_with_gradiend_cls: Any,
        callbacks: Any,
        runtime_monitor: Any = None,
    ) -> str:
        """
        Ensure mandatory prediction-objective resources before the base Trainer loads
        the training model. In particular, clm_mlm_head must have a trained auxiliary
        head, while probability-shift scoring remains on the original CLM head.

        Args:
            output_dir: Directory where the trained model will be written.
            args: Training arguments used for this run.
            model: Base model argument passed to the trainer.
            model_with_gradiend_cls: ModelWithGradiend subclass used for loading.
            callbacks: Training callbacks forwarded to the base trainer.
            runtime_monitor: Optional runtime monitor for diagnostics.
        """
        train_model = model if model is not None else self._model_arg
        objective = self._prediction_objective()
        objective.ensure_training_resources(self, train_model)
        return super()._train(
            output_dir=output_dir,
            args=args,
            model=model,
            model_with_gradiend_cls=model_with_gradiend_cls,
            callbacks=callbacks,
            runtime_monitor=runtime_monitor,
        )

    def _validate_explicit_decoder_eval_targets(
        self, targets: Dict[Any, List[Any]], *, warn_overlap: bool = True
    ) -> Dict[str, List[str]]:
        """
        Validate and normalize user-provided decoder_eval_targets in **class-based mode**.

        This helper expects a mapping of the form:
            {class_name: [token1, token2, ...], ...}

        and is used after any higher-level mappings (e.g. label-based, (label, class)-based)
        have been resolved into per-class token lists.

        - Ensures keys are strings that correspond to known classes (all_classes/target_classes).
        - Normalizes all tokens to strings and drops Nones.
        - When warn_overlap is True, emits a warning (but does not error) when different
          classes share overlapping tokens. Set warn_overlap=False when overlap is expected
          (e.g. decoder_eval_targets="label" where the same label can appear in multiple classes).
        """
        if not isinstance(targets, dict):
            raise TypeError(
                f"decoder_eval_targets must be a dict mapping class names to token lists; "
                f"got {type(targets).__name__}"
            )

        # Determine known classes from trainer metadata.
        known_classes: set = set()
        if getattr(self, "all_classes", None):
            known_classes.update(self.all_classes)
        if getattr(self, "target_classes", None):
            known_classes.update(self.target_classes)

        # If we have known classes, ensure decoder_eval_targets keys line up to avoid silent typos.
        if known_classes:
            invalid = [k for k in targets.keys() if k not in known_classes]
            if invalid:
                raise ValueError(
                    "decoder_eval_targets contains class keys that are not present in the data: "
                    f"{sorted(invalid)}. Known classes: {sorted(known_classes)}. "
                    "Double-check for typos in class names or adjust target_classes/all_classes."
                )

        # Normalize tokens to strings and drop Nones/empties.
        normalized: Dict[str, List[str]] = {}
        for cls, vals in targets.items():
            if vals is None:
                normalized[cls] = []
                continue
            if not isinstance(vals, (list, tuple)):
                raise TypeError(
                    f"decoder_eval_targets[{cls!r}] must be a list/tuple of tokens, got {type(vals).__name__}"
                )
            cleaned = [str(v) for v in vals if v is not None]
            normalized[cls] = cleaned

        # Warn (but do not error) when user-provided targets have overlapping tokens between classes.
        if warn_overlap:
            token_sets: Dict[str, set] = {cls: set(v) for cls, v in normalized.items()}
            overlapping_details: List[str] = []
            cls_list = list(token_sets.keys())
            for i in range(len(cls_list)):
                for j in range(i + 1, len(cls_list)):
                    c1, c2 = cls_list[i], cls_list[j]
                    inter = token_sets[c1] & token_sets[c2]
                    if inter:
                        overlapping_details.append(f"{c1} ↔ {c2}: {sorted(inter)}")
            if overlapping_details:
                details = "; ".join(overlapping_details)
                logger.warning(
                    "decoder_eval_targets has overlapping tokens between classes: %s. "
                    "This can make per-class decoder evaluation harder to interpret. "
                    "If this is intentional, you can ignore this warning; otherwise, consider "
                    "providing disjoint token sets or using a more fine-grained mapping.",
                    details,
                )

        return normalized

    def _resolve_decoder_eval_targets(
        self, training_like_df: Optional[pd.DataFrame]
    ) -> Tuple[Optional[Dict[str, List[str]]], bool]:
        """
        Resolve decoder eval targets from configuration and/or data.

        Args:
            training_like_df: Optional training-like decoder evaluation DataFrame.
                Present for callers that already resolved eval rows; currently used
                only as part of the resolution contract.

        Returns:
            (targets, use_row_wise). When use_row_wise is True, targets is None and evaluation
            uses row-wise P(factual) vs P(alternative) per row. When False, targets is
            {class_name: [tokens]} for static class-based evaluation.

        Supported config.decoder_eval_targets:
          - **None**: Auto-infer from unified data (factual + alternative per class).
            If inferred targets overlap across classes, fall back to row-wise and log an info message.
          - **"label"**: Row-wise evaluation (P(factual) and P(alternative) per row).
          - **Dict[class_name, List[tokens]]**: Class-based static targets (validated against known classes).
        """
        cfg = self.config
        raw = cfg.decoder_eval_targets

        self._ensure_data()
        cache_key = (cfg.decoder_eval_targets, id(self.combined_data) if self.combined_data is not None else None)
        if raw is not None and getattr(self, "_resolved_decoder_eval_targets_cache_key", None) == cache_key:
            cached = getattr(self, "_resolved_decoder_eval_targets_cache", None)
            cached_row_wise = getattr(self, "_resolved_decoder_eval_use_row_wise", False)
            if cached is not None or cached_row_wise:
                return (cached, cached_row_wise)

        # None: infer; if overlap, use row-wise
        if raw is None:
            inferred, has_overlap = self._infer_decoder_eval_targets()
            if has_overlap:
                logger.info(
                    "Auto-inferred decoder_eval_targets have overlapping tokens between classes; "
                    "using row-wise evaluation (P(factual) vs P(alternative) per row)."
                )
                self._resolved_decoder_eval_targets_cache_key = cache_key
                self._resolved_decoder_eval_targets_cache = None
                self._resolved_decoder_eval_use_row_wise = True
                return (None, True)
            self._resolved_decoder_eval_targets_cache_key = cache_key
            self._resolved_decoder_eval_targets_cache = inferred
            self._resolved_decoder_eval_use_row_wise = False
            return (inferred, False)

        # "label" -> row-wise
        if isinstance(raw, str):
            if raw != "label":
                raise ValueError(
                    "decoder_eval_targets as a string supports only 'label' for row-wise evaluation. "
                    f"Got {raw!r}."
                )
            self._resolved_decoder_eval_targets_cache_key = cache_key
            self._resolved_decoder_eval_targets_cache = None
            self._resolved_decoder_eval_use_row_wise = True
            return (None, True)

        # Class-based dict only
        if not isinstance(raw, dict):
            raise TypeError(
                f"decoder_eval_targets must be None, the string 'label', or a dict of class_name -> list of tokens; "
                f"got {type(raw).__name__}."
            )
        known_classes: set = set()
        if getattr(self, "all_classes", None):
            known_classes.update(self.all_classes)
        if getattr(self, "target_classes", None):
            known_classes.update(self.target_classes)
        keys = list(raw.keys())
        if not keys:
            return ({}, False)
        invalid = set(keys) - known_classes
        if not known_classes or not all(isinstance(k, str) for k in keys) or invalid:
            raise ValueError(
                "decoder_eval_targets must be a dict with class names as keys (matching your data classes). "
                f"Invalid or unknown keys: {sorted(invalid) if invalid else keys}. Known classes: {sorted(known_classes)}."
            )
        result = self._validate_explicit_decoder_eval_targets(raw)
        self._resolved_decoder_eval_targets_cache_key = cache_key
        self._resolved_decoder_eval_targets_cache = result
        self._resolved_decoder_eval_use_row_wise = False
        return (result, False)

    def _encoder_cache_path(self, model_path: str, **encoder_kwargs: Any) -> Optional[str]:
        """
        Encoder cache path for analysis CSV.
        Cache under experiment_dir; includes split/max_size in cache key.

        Args:
            model_path: Model path or identifier associated with this cache lookup.
            **encoder_kwargs: Encoder-analysis options that affect cache keys,
                currently including ``split`` and ``max_size``.
        """
        experiment_dir = self.experiment_dir
        split = encoder_kwargs.get("split")
        max_size = encoder_kwargs.get("max_size")
        key_kwargs: Dict[str, Any] = {}
        if split is not None:
            available = None
            if self.combined_data is not None and UNIFIED_SPLIT in self.combined_data.columns:
                available = self.combined_data[UNIFIED_SPLIT].dropna().astype(str).tolist()
            key_kwargs["split"] = encoder_split_cache_key(split, available=available)
        if max_size is not None:
            key_kwargs["max_size"] = max_size
        return resolve_encoder_analysis_path(experiment_dir, None, **key_kwargs)

    def _cached_encoder_df_matches_request(
        self,
        df: pd.DataFrame,
        *,
        include_other_classes: bool = False,
        use_all_transitions: bool = False,
        transition_selection: Optional[List[Union[TransitionSpec, Tuple[str, str]]]] = None,
    ) -> bool:
        if transition_selection is not None:
            return True
        if not (include_other_classes or use_all_transitions):
            return True
        if not self.all_classes or len(self.all_classes) <= 2:
            return True
        if not self.target_classes or len(self.target_classes) != 2:
            return True
        if "source_id" not in df.columns or "target_id" not in df.columns:
            return False
        training_df = df[df["type"].astype(str) == "training"] if "type" in df.columns else df
        if training_df.empty:
            return False
        target_set = {str(c) for c in self.target_classes}
        source_vals = training_df["source_id"].dropna().astype(str)
        target_vals = training_df["target_id"].dropna().astype(str)
        if source_vals.empty or target_vals.empty:
            return False
        pair_local = source_vals.isin(target_set) & target_vals.isin(target_set)
        return bool((~pair_local).any())

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
            dataset_trust_remote_code: Optional[bool] = None,
            all_classes: Optional[List[str]] = None,
            masked_col: Optional[str] = None,
            label_col: Optional[str] = None,
            label_class_col: Optional[str] = None,
            split_col: Optional[str] = _SPLIT_COL_UNSET,
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
            eval_neutral_additional_excluded_words: Optional[List[str]] = None,
            eval_neutral_max_rows: Optional[int] = None,
            img_format: Optional[str] = None,
            img_dpi: Optional[int] = None,
            class_merge_map: Optional[Dict[str, List[str]]] = None,
            class_merge_transition_groups: Optional[List[List[str]]] = None,
            split_group_col: Optional[str] = None,
            split_group_key: SplitGroupKey = None,
            split_ratios: Optional[SplitRatiosInput] = None,
            split_train_ratio: Optional[float] = None,
            split_val_ratio: Optional[float] = None,
            split_test_ratio: Optional[float] = None,
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
            dataset_trust_remote_code: Optional trust_remote_code value for
                HuggingFace datasets.load_dataset. None means do not pass it.
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
            img_format: Image format for plots (e.g. 'pdf', 'png'). Default 'png'.
            img_dpi: DPI for saved plots (e.g. 600 for publication). None = use visualizer default.
            class_merge_map: Optional mapping from merged class names to raw classes.
            class_merge_transition_groups: Optional raw-class transition groups to
                keep before merging.
            split_group_col: Optional column used to group examples for vocabulary-held-out splitting.
            split_group_key: Optional callable or callable sequence applied before split grouping.
            split_ratios: Optional train/validation/test split ratio specification.
            split_train_ratio: Train split ratio used when split_ratios is omitted.
            split_val_ratio: Validation split ratio used when split_ratios is omitted.
            split_test_ratio: Test split ratio used when split_ratios is omitted.
        """
        # Type checks for key scalar parameters (optional params may be None)
        if run_id is not None and not isinstance(run_id, str):
            raise TypeError(f"run_id must be str or None, got {type(run_id).__name__}")
        if hf_dataset is not None and not isinstance(hf_dataset, str):
            raise TypeError(f"hf_dataset must be str or None, got {type(hf_dataset).__name__}")
        if hf_subset is not None and not isinstance(hf_subset, (str, list)):
            raise TypeError(f"hf_subset must be str, list, or None, got {type(hf_subset).__name__}")
        if dataset_trust_remote_code is not None and not isinstance(dataset_trust_remote_code, bool):
            raise TypeError(
                "dataset_trust_remote_code must be bool or None, "
                f"got {type(dataset_trust_remote_code).__name__}"
            )
        if masked_col is not None and not isinstance(masked_col, str):
            raise TypeError(f"masked_col must be str or None, got {type(masked_col).__name__}")
        if label_col is not None and not isinstance(label_col, str):
            raise TypeError(f"label_col must be str or None, got {type(label_col).__name__}")
        if label_class_col is not None and not isinstance(label_class_col, str):
            raise TypeError(f"label_class_col must be str or None, got {type(label_class_col).__name__}")
        if (
            split_col is not _SPLIT_COL_UNSET
            and split_col is not None
            and not isinstance(split_col, str)
        ):
            raise TypeError(f"split_col must be str or None, got {type(split_col).__name__}")
        if alternative_col is not None and not isinstance(alternative_col, str):
            raise TypeError(f"alternative_col must be str or None, got {type(alternative_col).__name__}")
        if alternative_class_col is not None and not isinstance(alternative_class_col, str):
            raise TypeError(f"alternative_class_col must be str or None, got {type(alternative_class_col).__name__}")
        if use_class_names_as_columns is not None and not isinstance(use_class_names_as_columns, bool):
            raise TypeError(f"use_class_names_as_columns must be bool or None, got {type(use_class_names_as_columns).__name__}")
        if max_counterfactuals_per_sentence is not None and not isinstance(max_counterfactuals_per_sentence, int):
            raise TypeError(f"max_counterfactuals_per_sentence must be int or None, got {type(max_counterfactuals_per_sentence).__name__}")
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError(f"random_state must be int or None, got {type(random_state).__name__}")
        if n_features is not None and not isinstance(n_features, int):
            raise TypeError(f"n_features must be int or None, got {type(n_features).__name__}")
        if decoder_eval_restrict_to_target_classes is not None and not isinstance(decoder_eval_restrict_to_target_classes, bool):
            raise TypeError(f"decoder_eval_restrict_to_target_classes must be bool or None, got {type(decoder_eval_restrict_to_target_classes).__name__}")
        if decoder_eval_prob_on_other_class is not None and not isinstance(decoder_eval_prob_on_other_class, bool):
            raise TypeError(f"decoder_eval_prob_on_other_class must be bool or None, got {type(decoder_eval_prob_on_other_class).__name__}")
        if decoder_eval_lms_max_samples is not None and not isinstance(decoder_eval_lms_max_samples, int):
            raise TypeError(f"decoder_eval_lms_max_samples must be int or None, got {type(decoder_eval_lms_max_samples).__name__}")
        if eval_neutral_max_rows is not None and not isinstance(eval_neutral_max_rows, int):
            raise TypeError(f"eval_neutral_max_rows must be int or None, got {type(eval_neutral_max_rows).__name__}")
        if img_format is not None and not isinstance(img_format, str):
            raise TypeError(f"img_format must be str or None, got {type(img_format).__name__}")
        if img_dpi is not None and not isinstance(img_dpi, int):
            raise TypeError(f"img_dpi must be int or None, got {type(img_dpi).__name__}")

        args_for_super = training_args or args
        split_col_explicit = split_col is not _SPLIT_COL_UNSET
        if config is None:
            cfg_kwargs: Dict[str, Any] = {
                "run_id": run_id,
                "data": data,
                "hf_dataset": hf_dataset,
                "hf_subset": hf_subset,
                "hf_splits": hf_splits,
                "dataset_trust_remote_code": dataset_trust_remote_code,
                "target_classes": list(target_classes) if isinstance(target_classes, tuple) else target_classes,
                "all_classes": all_classes,
                "masked_col": masked_col,
                "label_col": label_col,
                "label_class_col": label_class_col,
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
                "eval_neutral_additional_excluded_words": eval_neutral_additional_excluded_words,
                "eval_neutral_max_rows": eval_neutral_max_rows,
                "img_format": img_format,
                "img_dpi": img_dpi,
                "class_merge_map": class_merge_map,
                "class_merge_transition_groups": class_merge_transition_groups,
                "split_group_col": split_group_col,
                "split_group_key": split_group_key,
                "split_ratios": split_ratios,
                "split_train_ratio": split_train_ratio,
                "split_val_ratio": split_val_ratio,
                "split_test_ratio": split_test_ratio,
            }
            if split_col_explicit:
                cfg_kwargs["split_col"] = split_col
            _preserve_none = {"split_group_key"}
            if split_col_explicit:
                _preserve_none.add("split_col")
            cfg_kwargs = {
                k: v
                for k, v in cfg_kwargs.items()
                if v is not None or k in _preserve_none
            }
            if target_classes is not None:
                cfg_kwargs["target_classes"] = list(target_classes) if isinstance(target_classes, tuple) else target_classes
            config = TextPredictionConfig(**cfg_kwargs)
        else:
            split_col_explicit = True

        # Prefer explicit target_classes argument; otherwise keep any value already on the config.
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
        self._all_classes: Optional[List[str]] = None
        self._data_loaded = False
        self._data_materialized_all_class_transitions = False
        self._combined_data_template: Optional[pd.DataFrame] = None
        self._splits_resplit_seed: Optional[int] = None
        self._split_col_explicit = split_col_explicit

    def _set_all_classes(self, classes: Optional[List[str]]) -> None:
        """Set the list of all classes in the dataset (including neutral/identity)."""
        if classes is None:
            return
        self._all_classes = classes

    @staticmethod
    def _validate_classes_in_data(
        class_dfs: Dict[str, pd.DataFrame],
        classes: List[str],
        param_name: str = "target_classes",
    ) -> None:
        """Raise ValueError if any requested class is absent from per-class data.

        Args:
            class_dfs: Per-class DataFrame mapping.
            classes: Classes expected to be present in ``class_dfs``.
            param_name: Name used in the error message for the checked parameter.
        """
        available = set(class_dfs.keys())
        missing = [c for c in classes if c not in available]
        if missing:
            raise ValueError(
                f"{param_name} {missing} are not present in the data. "
                f"Available classes: {sorted(available)}. "
                "Ensure target_classes (or the classes used for training) match the keys of your per-class data."
            )

    def _exclude_generated_incomplete_target_classes(self, data_path: Union[str, Path]) -> List[str]:
        incomplete = _load_generated_incomplete_classes(data_path, self.config.label_class_col)
        if not incomplete or self._target_classes is None:
            return incomplete
        incomplete_set = set(incomplete)
        kept = [cls for cls in self._target_classes if cls not in incomplete_set]
        removed = [cls for cls in self._target_classes if cls in incomplete_set]
        if removed:
            logger.warning(
                "Excluding generated incomplete classes from GRADIEND target_classes: %s. "
                "These classes are listed in %s and will not be trained.",
                removed,
                _generated_incomplete_data_path(data_path),
            )
            self._target_classes = kept
            self.config.target_classes = kept
            if len(kept) == 2:
                self.config.pair = tuple(kept)
            elif len(kept) < 2:
                self.config.pair = None
        return incomplete

    def _validate_required_splits(
        self,
        combined_data: Optional[pd.DataFrame],
        required_splits: Optional[List[str]] = None,
    ) -> None:
        """Fail early when unified data cannot support the configured workflow.

        Args:
            combined_data: Unified training/evaluation DataFrame.
            required_splits: Optional explicit split names that must be present.
        """
        if combined_data is None or len(combined_data) == 0:
            return
        if UNIFIED_SPLIT not in combined_data.columns:
            return

        available = combined_data[UNIFIED_SPLIT].dropna().astype(str).tolist()
        policy = SplitPolicy.from_available(available)
        do_eval = bool(getattr(getattr(self, "training_args", None), "do_eval", True))
        vocabulary_held_out = getattr(self.config, "split_col", "split") is None

        if required_splits is not None:
            required = [normalize_split_name(split) for split in required_splits]
            available_norm = order_split_names(set(normalize_split_name(str(s)) for s in available))
            missing = [split for split in required if split not in available_norm]
            if missing:
                raise ValueError(
                    "combined_data is missing required split(s): "
                    f"{missing}. Available: {available_norm}."
                )
            return

        validate_data_split_policy(
            policy,
            vocabulary_held_out=vocabulary_held_out,
            do_eval=do_eval,
        )

    def _configured_split_ratios(self) -> Tuple[float, float, float]:
        return normalize_split_ratios(
            self.config.split_ratios,
            train=float(self.config.split_train_ratio),
            val=float(self.config.split_val_ratio),
            test=float(self.config.split_test_ratio),
        )

    def _vocabulary_split_viable_for_targets(self, min_rows_per_class: int = 10) -> bool:
        if self._combined_data is None or self._combined_data.empty:
            return False
        target_classes = self._target_classes or self.config.target_classes
        if not target_classes or len(target_classes) != 2:
            return False
        train_ratio, val_ratio, test_ratio = self._configured_split_ratios()
        group_col = self.config.split_group_col or UNIFIED_FACTUAL
        group_key = self.config.split_group_key
        if group_key is None:
            group_key = [str.strip, str.casefold]
        return vocabulary_held_out_viable_for_target_pair(
            self._combined_data,
            target_classes,
            group_col=group_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            factual_class_col=UNIFIED_FACTUAL_CLASS,
            alternative_class_col=UNIFIED_ALTERNATIVE_CLASS,
            min_rows_per_class=min_rows_per_class,
            group_key=group_key,
        )

    def _validate_explicit_vocabulary_split_or_raise(self) -> None:
        if self._vocabulary_split_viable_for_targets():
            return
        train_ratio, val_ratio, test_ratio = self._configured_split_ratios()
        min_keys = sum(1 for ratio in (train_ratio, val_ratio, test_ratio) if ratio > 0)
        target_classes = self._target_classes or self.config.target_classes
        raise ValueError(
            "split_col=None requires vocabulary-held-out splits with at least 10 rows and "
            f"at least {min_keys} distinct factual token(s) per target class "
            f"{list(target_classes) if target_classes else []}. "
            "Add more target tokens, lower split ratios, or use split_col='split' for random row splits."
        )

    def _resolve_split_col_strategy(self) -> None:
        """Pick random row splits vs vocabulary-held-out when split_col was not set explicitly."""
        if self._split_col_explicit:
            if self.config.split_col is None:
                self._validate_explicit_vocabulary_split_or_raise()
            return
        if self._vocabulary_split_viable_for_targets():
            logger.info(
                "Auto-selected vocabulary-held-out splits (>=10 rows and sufficient distinct "
                "factual tokens per target class)."
            )
            self.config.split_col = None
        else:
            self.config.split_col = "split"

    def _target_pair_transition_mask(self, df: pd.DataFrame) -> pd.Series:
        target_classes = self._target_classes or self.config.target_classes
        if not target_classes or len(target_classes) != 2:
            return pd.Series(True, index=df.index)
        tc_set = {str(c) for c in target_classes}
        return (
            df[UNIFIED_FACTUAL_CLASS].astype(str).isin(tc_set)
            & df[UNIFIED_ALTERNATIVE_CLASS].astype(str).isin(tc_set)
            & (
                df[UNIFIED_FACTUAL_CLASS].astype(str)
                != df[UNIFIED_ALTERNATIVE_CLASS].astype(str)
            )
        )

    def _resplit_seed_for_training(self, seed_value: Optional[int] = None) -> int:
        args = getattr(self, "training_args", None)
        base_seed = int(getattr(args, "seed", 0) or 0) if args is not None else 0
        if args is not None and getattr(args, "split_resplit_per_seed", False) and seed_value is not None:
            return int(seed_value)
        return base_seed

    def _balanced_split_cycle_args(
        self,
        seed_value: Optional[int],
        *,
        split_cycle_index: Optional[int] = None,
        split_cycle_length: Optional[int] = None,
    ) -> Dict[str, int]:
        args = getattr(self, "training_args", None)
        if args is None or not getattr(args, "split_resplit_per_seed", False):
            return {}
        if getattr(args, "split_resplit_strategy", "random") != "balanced_cycle":
            return {}
        if seed_value is None:
            return {}
        base_seed = int(getattr(args, "seed", 0) or 0)
        cycle_index = int(split_cycle_index) if split_cycle_index is not None else int(seed_value) - base_seed
        cycle_length = int(split_cycle_length) if split_cycle_length is not None else int(
            getattr(args, "min_convergent_seeds", None) or getattr(args, "max_seeds", 1) or 1
        )
        return {
            "balanced_cycle_index": cycle_index,
            "balanced_cycle_length": cycle_length,
        }

    def _apply_vocabulary_splits(
        self,
        seed: int,
        *,
        seed_value: Optional[int] = None,
        split_cycle_index: Optional[int] = None,
        split_cycle_length: Optional[int] = None,
    ) -> None:
        if self.config.split_col is not None:
            return
        if self._combined_data_template is None:
            return
        group_col = self.config.split_group_col or UNIFIED_FACTUAL
        group_key = self.config.split_group_key
        if group_key is None:
            group_key = [str.strip, str.casefold]
        train_ratio, val_ratio, test_ratio = self._configured_split_ratios()
        template = self._combined_data_template
        pair_mask = self._target_pair_transition_mask(template)
        if pair_mask.any() and not pair_mask.all():
            pair_df = template[pair_mask].copy()
            other_df = template[~pair_mask].copy()
            resplit_pair = resplit_unified_dataframe(
                pair_df,
                group_col=group_col,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=int(seed),
                group_key=group_key,
                per_feature_class=True,
                feature_class_col=UNIFIED_FACTUAL_CLASS,
                split_col=UNIFIED_SPLIT,
                align_alternatives_with_split_vocab=True,
                **self._balanced_split_cycle_args(
                    seed_value,
                    split_cycle_index=split_cycle_index,
                    split_cycle_length=split_cycle_length,
                ),
            )
            self._combined_data = pd.concat([resplit_pair, other_df], ignore_index=True)
        else:
            self._combined_data = resplit_unified_dataframe(
                template,
                group_col=group_col,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=int(seed),
                group_key=group_key,
                per_feature_class=True,
                feature_class_col=UNIFIED_FACTUAL_CLASS,
                split_col=UNIFIED_SPLIT,
                align_alternatives_with_split_vocab=True,
                **self._balanced_split_cycle_args(
                    seed_value,
                    split_cycle_index=split_cycle_index,
                    split_cycle_length=split_cycle_length,
                ),
            )
        self._splits_resplit_seed = int(seed)

    def _refresh_data_splits_for_seed(
        self,
        seed_value: int,
        args: Any,
        *,
        split_cycle_index: Optional[int] = None,
        split_cycle_length: Optional[int] = None,
    ) -> None:
        if self.config.split_col is not None:
            return
        per_seed = bool(getattr(args, "split_resplit_per_seed", False))
        balanced_cycle = bool(
            per_seed and getattr(args, "split_resplit_strategy", "random") == "balanced_cycle"
        )
        target_seed = (
            self._resplit_seed_for_training(None)
            if balanced_cycle
            else self._resplit_seed_for_training(seed_value if per_seed else None)
        )
        if not per_seed and self._splits_resplit_seed == target_seed:
            return
        self._apply_vocabulary_splits(
            target_seed,
            seed_value=seed_value if per_seed else None,
            split_cycle_index=split_cycle_index,
            split_cycle_length=split_cycle_length,
        )

    def _ensure_data_for_training(self) -> None:
        """Ensure data is loaded before creating the model for training (so pair is set and from_pretrained can set feature_class_encoding_direction)."""
        self._ensure_data()
        self._validate_required_splits(self._combined_data)

    def _ensure_data(self, *, materialize_all_class_transitions: Optional[bool] = None) -> None:
        """Load and normalize data on first use. Idempotent.

        Training data can be specified as:
        - config.hf_dataset: HuggingFace dataset ID (optional subset/splits).
        - config.data: HuggingFace dataset ID (per-class configs), local path (.csv/.parquet),
          per-class dict, or DataFrame in memory. A string is treated as HF id unless it is
          an existing file path.
        """
        if self._data_loaded:
            if materialize_all_class_transitions and not self._data_materialized_all_class_transitions:
                self.data = None
                self.class_datasets = None
                self._combined_data = None
                self._combined_data_template = None
                self._splits_resplit_seed = None
                self._data_loaded = False
            else:
                return
        if self._data_loaded:
            return
        config = self.config

        def _pair_for_unified(classes: List[str], pair: Optional[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
            if pair is None:
                return None
            return pair if set(classes).issubset(set(pair)) else None

        if materialize_all_class_transitions is None:
            materialize_all_class_transitions = bool(
                self._default_from_training_args(None, "include_other_classes", fallback=False)
            )
        else:
            materialize_all_class_transitions = bool(materialize_all_class_transitions)

        # Apply seed from training_args so data build (unified, sampling) is deterministic
        _seed = getattr(config, "seed", None) or (getattr(self, "training_args", None) and getattr(self.training_args, "seed", None))
        if _seed is not None:
            _apply_seed(int(_seed))
        # Single HuggingFace gate: hf_dataset => merged-style load; data=str (not a path) => per-class load
        is_hf_id = isinstance(config.data, str) and not Path(config.data).is_file()
        if config.hf_dataset is not None:
            # HF merged-style: one dataset, optional subsets/splits, then optional merge/transition filtering, then merged_to_unified
            raw = self._load_hf_dataset(
                config.hf_dataset,
                config.hf_subset,
                config.hf_splits,
                dataset_trust_remote_code=config.dataset_trust_remote_code,
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
            if materialize_all_class_transitions and config.all_classes is not None:
                classes_to_load = config.all_classes
            elif config.target_classes is not None:
                classes_to_load = config.target_classes
            else:
                classes_to_load = config.all_classes if config.all_classes is not None else "all"
            class_dfs = load_hf_per_class(
                config.data,
                classes=classes_to_load,
                splits=config.hf_splits,
                masked_col=config.masked_col,
                split_col=config.split_col,
                dataset_trust_remote_code=config.dataset_trust_remote_code,
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
            if config.target_classes is not None:
                self._validate_classes_in_data(class_dfs, config.target_classes, param_name="target_classes")
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
                pair=_pair_for_unified(inferred_classes, pair),
                include_identity_rows=False,
                max_counterfactuals_per_sentence=getattr(config, "max_counterfactuals_per_sentence", 1),
                random_state=getattr(config, "random_state", getattr(config, "seed", None)),
            )
            self._combined_data = unified
        elif config.data is not None:
            if isinstance(config.data, dict):
                # Infer all_classes from data keys; merge by class_merge_map if set
                all_classes_from_data = sorted(list(config.data.keys()))
                configured_all_classes = list(config.all_classes) if config.all_classes is not None else None
                self._set_all_classes(configured_all_classes or all_classes_from_data)
                classes_for_transitions = (
                    configured_all_classes
                    if materialize_all_class_transitions and configured_all_classes is not None
                    else self.target_classes or all_classes_from_data
                )
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
                self._validate_classes_in_data(class_dfs, classes_for_transitions, param_name="all_classes" if configured_all_classes else "target_classes")
                unified = per_class_dict_to_unified(
                    class_dfs,
                    classes=classes_for_transitions,
                    masked_col=config.masked_col,
                    split_col=config.split_col,
                    use_class_names_as_columns=getattr(config, "use_class_names_as_columns", True),
                    pair=_pair_for_unified(classes_for_transitions, pair),
                    include_identity_rows=False,
                    max_counterfactuals_per_sentence=getattr(config, "max_counterfactuals_per_sentence", 1),
                    random_state=getattr(config, "random_state", getattr(config, "seed", None)),
                )
                self._combined_data = unified
            elif isinstance(config.data, pd.DataFrame):
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
            elif isinstance(config.data, (Path, str)):
                data_df = resolve_training_data_path(config.data)
                self._exclude_generated_incomplete_target_classes(config.data)
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
                raise TypeError(
                    f"config.data must be a DataFrame, dict, local path, or HuggingFace dataset id; "
                    f"got {type(config.data).__name__}"
                )
        self._check_data_non_empty()
        self._resolve_split_col_strategy()
        if config.split_col is None:
            self._combined_data_template = self._combined_data.copy()
            self._apply_vocabulary_splits(self._resplit_seed_for_training())
        else:
            self._combined_data_template = None
        self._data_loaded = True
        self._data_materialized_all_class_transitions = materialize_all_class_transitions

        if self._all_classes is None and self._combined_data is not None:
            src = self._combined_data[UNIFIED_FACTUAL_CLASS].unique().tolist()
            tgt = self._combined_data[UNIFIED_ALTERNATIVE_CLASS].unique().tolist()
            self._set_all_classes(sorted(set(src) | set(tgt)))

        # When target_classes were not set explicitly, infer from data when exactly two classes
        # (so decoder evaluation and get_target_feature_classes can derive feature factors).
        if self._target_classes is None and self._all_classes is not None and len(self._all_classes) == 2:
            inferred_tc = sorted(self._all_classes)
            self._target_classes = inferred_tc
            config.target_classes = inferred_tc
            config.pair = tuple(inferred_tc)

        # Infer pair from target_classes when exactly 2 target classes (stored in config.pair)
        if self._target_classes is not None and len(self._target_classes) == 2:
            config.pair = tuple(self._target_classes)
        if self.config.decoder_eval_targets is None and self._combined_data is not None:
            try:
                inferred, has_overlap = self._infer_decoder_eval_targets()
                if not has_overlap:
                    self.config.decoder_eval_targets = inferred
            except Exception as e:
                logger.warning(f"Could not auto-infer decoder_eval_targets: {e}")

    @classmethod
    def peek_unified_training_data(cls, *args: Any, **kwargs: Any) -> Optional[pd.DataFrame]:
        """Load training data into the unified schema without loading a model.

        Uses the same normalization path as :meth:`_ensure_data` (HF per-class,
        merged files, in-memory dicts, etc.). Intended for suite introspection.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.
        """
        init_kwargs = dict(kwargs)
        if args:
            model = args[0]
            extra_args = args[1:]
        else:
            model = init_kwargs.pop("model", "__peek_unified_training_data__")
            extra_args = ()
        instance = cls(model, *extra_args, **init_kwargs)
        return instance.combined_data

    @property
    def combined_data(self) -> Optional[pd.DataFrame]:
        """Unified training data (lazy-loaded on first access). When class_merge_map is set, already merged at load."""
        self._ensure_data()
        return self._combined_data

    def plot_training_convergence(
        self,
        *,
        plot_mean_by_class: bool = True,
        plot_mean_by_feature_class: Optional[bool] = None,
        plot_correlation: bool = True,
        class_spread: Optional[Literal["minmax", "iqr", "ci95"]] = None,
        output: Optional[str] = None,
        show: bool = True,
        title: Union[str, bool] = True,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: Optional[str] = None,
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot text-prediction training convergence with config image defaults.

        Args:
            plot_mean_by_class: Plot mean encoder values by label class.
            plot_mean_by_feature_class: Plot means grouped by feature class.
            plot_correlation: Plot correlation over training steps.
            class_spread: Optional spread band behind class means.
                ``"minmax"`` shades min-max encoded values, ``"iqr"`` shades Q1-Q3,
                ``"ci95"`` shades mean +/- 1.96 standard errors,
                and ``None`` disables spread shading.
            output: Optional explicit output file path.
            show: Whether to display the figure interactively.
            title: Plot title. ``True`` uses the default title, ``False`` omits it.
            figsize: Optional Matplotlib figure size.
            img_format: Optional output format. Defaults to trainer config.
            dpi: Optional output DPI. Defaults to trainer config when available.
            highlight_non_convergence: Override non-convergence markers.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the base trainer.
        """
        if img_format is None:
            img_format = getattr(self.config, "img_format", "png")
        if dpi is None and getattr(self.config, "img_dpi", None) is not None:
            dpi = self.config.img_dpi
        return super().plot_training_convergence(
            plot_mean_by_class=plot_mean_by_class,
            plot_mean_by_feature_class=plot_mean_by_feature_class,
            plot_correlation=plot_correlation,
            class_spread=class_spread,
            output=output,
            show=show,
            title=title,
            figsize=figsize,
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_training_convergence.__doc__ = (
        Trainer.plot_training_convergence.__doc__
    )

    def plot_encoder_distributions(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        output: Optional[str] = None,
        output_dir: Optional[str] = None,
        show: bool = True,
        title: Union[str, bool] = True,
        target_and_neutral_only: bool = True,
        split_plot_mode: str = "facet",
        include_neutral: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: Optional[str] = None,
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot text-prediction encoder distributions with config image defaults.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame.
            output: Optional explicit output file path.
            output_dir: Optional output directory.
            show: Whether to display the figure interactively.
            title: Plot title. ``True`` uses the default title, ``False`` omits it.
            target_and_neutral_only: If True, omit identity/auxiliary rows.
            split_plot_mode: How split-aware data is shown.
            include_neutral: If True, include neutral rows when present.
            figsize: Optional Matplotlib figure size.
            img_format: Optional output format. Defaults to trainer config.
            dpi: Optional output DPI. Defaults to trainer config when available.
            highlight_non_convergence: Override non-convergence markers.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the base trainer.
        """
        if img_format is None:
            img_format = getattr(self.config, "img_format", "png")
        if dpi is None and getattr(self.config, "img_dpi", None) is not None:
            dpi = self.config.img_dpi
        return super().plot_encoder_distributions(
            encoder_df=encoder_df,
            output=output,
            output_dir=output_dir,
            show=show,
            title=title,
            target_and_neutral_only=target_and_neutral_only,
            split_plot_mode=split_plot_mode,
            include_neutral=include_neutral,
            figsize=figsize,
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_encoder_distributions.__doc__ = (
        Trainer.plot_encoder_distributions.__doc__
    )

    def plot_probability_shifts(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        class_ids: Optional[List[str]] = None,
        target_class: Optional[str] = None,
        increase_target_probabilities: bool = True,
        use_cache: Optional[bool] = None,
        *,
        output: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: Optional[str] = None,
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> str:
        """Plot text-prediction decoder probability shifts.

        Args:
            decoder_results: Optional result from ``evaluate_decoder``.
            class_ids: Optional class ids to include in the plot.
            target_class: Optional single target class to plot.
            increase_target_probabilities: True for strengthen plots, False for
                weaken plots.
            use_cache: Whether cached decoder results may be used.
            output: Optional explicit output file path.
            show: Whether to display the Matplotlib figure.
            figsize: Optional Matplotlib figure size.
            img_format: Optional output image format. Defaults to trainer config.
            dpi: Optional output DPI. Defaults to trainer config when available.
            highlight_non_convergence: Override non-convergence markers.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        if img_format is None:
            img_format = getattr(self.config, "img_format", "png")
        if dpi is None and getattr(self.config, "img_dpi", None) is not None:
            dpi = self.config.img_dpi
        return self.evaluator.plot_probability_shifts(
            decoder_results=decoder_results,
            class_ids=class_ids,
            target_class=target_class,
            increase_target_probabilities=increase_target_probabilities,
            use_cache=use_cache,
            output=output,
            show=show,
            figsize=figsize,
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_probability_shifts.__doc__ = (
        """Plot decoder probability shifts over learning-rate/grid candidates.

        Args:
            decoder_results: Optional result from ``evaluate_decoder``.
            class_ids: Optional class ids to include in the plot.
            target_class: Optional single target class to plot.
            increase_target_probabilities: True for strengthen plots, False for
                weaken plots.
            use_cache: Whether cached decoder results may be used.
            output: Optional explicit output file path.
            show: Whether to display the Matplotlib figure.
            figsize: Optional Matplotlib figure size.
            img_format: Optional output image format. Defaults to trainer config.
            dpi: Optional output DPI. Defaults to trainer config when available.
            highlight_non_convergence: Override non-convergence markers.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        + see_implementation("gradiend.visualizer.probability_shifts.plot_probability_shifts")
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
            dataset_trust_remote_code: Optional[bool] = None,
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
            dataset_trust_remote_code: Optional trust_remote_code value forwarded
                to ``datasets.load_dataset``.

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
        load_kwargs = {}
        if dataset_trust_remote_code is not None:
            load_kwargs["trust_remote_code"] = dataset_trust_remote_code
        for sub in subsets_to_load:
            try:
                if sub is None:
                    ds = load_dataset(dataset_name, **load_kwargs)
                else:
                    ds = load_dataset(dataset_name, sub, **load_kwargs)
            except Exception as e:
                raise ValueError(f"Could not load subset '{sub}' from {dataset_name}: {e}")

            # Handle both DatasetDict and Dataset
            if hasattr(ds, 'items'):  # DatasetDict
                for split_name, split_ds in ds.items():
                    # Normalize split name (e.g., 'val' -> 'validation' for HF convention)
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
            split: EncoderSplit = "train",
            class_pair: Optional[Tuple[str, str]] = None,
            batch_size: Optional[int] = None,
            max_size: Optional[int] = None,
            include_other_classes: bool = False,
            use_all_transitions: bool = False,
            transition_selection: Optional[List[Union[TransitionSpec, Tuple[str, str]]]] = None,
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

        Args:
            model_or_tokenizer: Model-with-GRADIEND or tokenizer used to build
                text inputs.
            split: Split name(s) to load. Supports one split, ``"all"``, or a
                sequence of split names.
            class_pair: Optional target class pair. Defaults to the trainer pair.
            batch_size: Optional raw text batch size.
            max_size: Optional per-group cap; defaults to
                ``TrainingArguments.train_max_size`` when omitted.
            include_other_classes:
                Deprecated alias for broadening evaluation/training-like data to
                all available transitions. Prefer ``use_all_transitions``.
            use_all_transitions:
                If True, include every transition available in the split instead
                of restricting to the active target pair. This is primarily
                useful for encoder analysis.
            transition_selection:
                Optional explicit transition specs for evaluation-style probing.
                When provided, these transitions are included in addition to the
                active target pair. Non-target transitions keep label ``0``.
                If set, ``include_other_classes`` / ``use_all_transitions`` are
                ignored.
            balance_column: Column used for balanced dataset scheduling and
                per-group capping.
            **kwargs: Additional dataset-construction options, including
                ``is_decoder_only_model``.
        """
        tokenizer = getattr(model_or_tokenizer, "tokenizer", model_or_tokenizer)
        self._prediction_objective(tokenizer)
        requested_all_transitions = bool(use_all_transitions or include_other_classes or transition_selection)
        self._ensure_data(materialize_all_class_transitions=requested_all_transitions)
        max_size = self._default_from_training_args(max_size, "train_max_size")
        if self.combined_data is None:
            raise ValueError("No data provided. Set data in config or override create_training_data().")
        if UNIFIED_TRANSITION not in self.combined_data.columns:
            raise ValueError(
                "combined_data must use unified schema (masked, split, factual_class, alternative_class, factual, alternative, transition).")

        available_splits = self.combined_data[UNIFIED_SPLIT].dropna().astype(str).tolist()
        resolved_splits = resolve_encoder_splits(split, available=available_splits)
        norm_col = self.combined_data[UNIFIED_SPLIT].astype(str).map(normalize_split_name)
        split_data = self.combined_data[norm_col.isin(resolved_splits)].copy()

        if len(split_data) == 0:
            raise ValueError(
                f"No data for split={split!r} (resolved {resolved_splits}). "
                f"Available: {sorted(set(norm_col.tolist()))}."
            )

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
        transition_edges = None
        if transition_selection:
            if include_other_classes or use_all_transitions:
                logger.warning(
                    "transition_selection was provided; ignoring include_other_classes/use_all_transitions."
                )
            transition_edges = expand_transition_selection(transition_selection)
            selected_transition_ids = {
                transition_id(src, tgt) for src, tgt in transition_edges
            } | train_transitions
            pair_data = split_data[split_data[UNIFIED_TRANSITION].isin(selected_transition_ids)].copy()
        elif (use_all_transitions or include_other_classes) and self.all_classes is not None and len(self.all_classes) > 2:
            pair_data = split_data.copy()
        else:
            pair_data = split_data[split_data[UNIFIED_TRANSITION].isin(train_transitions)].copy()

        if len(pair_data) == 0:
            raise ValueError(f"No data for transitions {train_transitions} in split '{split}'.")

        training_pairs = []
        feature_class_id_map = {
            (class_pair[0], class_pair[1]): 0,
            (class_pair[1], class_pair[0]): 1,
        }
        next_fcid = 2
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
            pair_entry = {
                "masked": row[UNIFIED_MASKED],
                "factual": row[UNIFIED_FACTUAL],
                "alternative": row[UNIFIED_ALTERNATIVE],
                "factual_id": src,
                "alternative_id": tgt,
                "label": label,
                "feature_class_id": _feature_class_id(src, tgt),
            }
            if UNIFIED_SPLIT in row.index and pd.notna(row[UNIFIED_SPLIT]):
                pair_entry[UNIFIED_SPLIT] = row[UNIFIED_SPLIT]
            training_pairs.append(pair_entry)

        # Identity transitions only for classes in all_classes that are not in target_classes
        neutral_classes = [c for c in (self.all_classes or []) if c not in (self.target_classes or [])]
        if add_identity and neutral_classes:
                neutral_data = split_data[split_data[UNIFIED_FACTUAL_CLASS].isin(neutral_classes)].copy()
                for _, row in neutral_data.iterrows():
                    c = row[UNIFIED_FACTUAL_CLASS]
                    identity_entry = {
                        UNIFIED_MASKED: row[UNIFIED_MASKED],
                        UNIFIED_FACTUAL: row[UNIFIED_FACTUAL],
                        UNIFIED_ALTERNATIVE: row[UNIFIED_FACTUAL],
                        "factual_id": c,
                        "alternative_id": c,
                        "label": 0,
                        "feature_class_id": _feature_class_id(c, c),
                    }
                    if UNIFIED_SPLIT in row.index and pd.notna(row[UNIFIED_SPLIT]):
                        identity_entry[UNIFIED_SPLIT] = row[UNIFIED_SPLIT]
                    training_pairs.append(identity_entry)
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
                                df_c[split_col_cfg].astype(str).map(normalize_split_name).isin(resolved_splits)
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

        # Apply max_size if specified: cap per logical balancing group.
        # For encoder eval this may be factual_id / alternative_id (depending on source),
        # while diff-style training naturally uses feature_class_id transitions.
        # The dataset's balance_column (set below) cycles through groups, ensuring equal
        # representation via oversampling. This downsampling reduces total dataset size but is
        # not strictly necessary for balancing (the scheduler handles that). It's kept for
        # memory/performance when train_max_size is set.
        args = getattr(self, "training_args", None)
        seed = getattr(args, "seed", 0) if args is not None else 0
        if seed is None:
            seed = 42
        group_cap_col = (
            balance_column
            if balance_column is not None and balance_column in training_df.columns
            else "feature_class_id"
        )
        if max_size is not None and len(training_df) > max_size:
            training_df = _sample_up_to_per_group(training_df, group_cap_col, max_size, seed)

        # Determine if decoder-only or encoder-decoder model
        is_seq2seq = is_seq2seq_model(tokenizer)
        is_decoder_only_model = kwargs.get("is_decoder_only_model")
        objective = self._prediction_objective(tokenizer)
        objective_name = objective.name
        if is_decoder_only_model is None:
            is_decoder_only_model = False if is_seq2seq else tokenizer.mask_token_id is None
        if objective_name == "clm_sequence_cloze":
            is_decoder_only_model = True
        elif objective_name == "seq2seq_decoder_sequence_cloze":
            is_decoder_only_model = False

        # Get batch_size
        if batch_size is None:
            batch_size = kwargs.get("batch_size", 1)

        mlm_head_target_labels = None
        if objective_name == "clm_mlm_head":
            head_dir = resolve_decoder_mlm_head_dir(self.experiment_dir)
            if head_dir:
                try:
                    meta = load_decoder_mlm_head_meta(head_dir)
                    mlm_head_target_labels = meta.get("target_labels")
                except OSError:
                    mlm_head_target_labels = None

        # Create text-specific training dataset (seed for deterministic batch ordering)
        return TextTrainingDataset(
            training_df,
            tokenizer,
            batch_size=batch_size,
            is_decoder_only_model=is_decoder_only_model,
            is_seq2seq_model=is_seq2seq,
            target_key="label",
            balance_column=balance_column,
            seed=seed,
            prediction_objective=objective_name,
            rhs_window=getattr(getattr(self, "_training_args", None), "decoder_sequence_cloze_rhs_window", -1),
            mlm_head_target_labels=mlm_head_target_labels,
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

        Args:
            raw_training_data: Text training dataset produced by
                ``create_training_data``.
            model_with_gradiend: Model used to create gradients.
            cache_dir: Optional gradient cache directory.
            use_cached_gradients: Whether existing cached gradients may be reused.
            **kwargs: Optional gradient dataset settings such as ``source``,
                ``target``, ``dtype``, and ``device``.
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

    def _infer_decoder_eval_targets(self) -> Tuple[Dict[str, List[str]], bool]:
        """
        Infer decoder evaluation targets from unified data and, when needed, from per-class datasets.
        For each class, collects tokens used as factual (when factual_class=C) and as alternative (when alternative_class=C).
        Returns (targets, has_overlap). When has_overlap is True, the resolver will use row-wise evaluation instead.
        """
        self._ensure_data()
        if self.combined_data is None:
            raise ValueError("No data available to infer decoder eval targets")
        if UNIFIED_TRANSITION not in self.combined_data.columns:
            raise ValueError("combined_data must use unified schema to infer decoder eval targets")
        # Prefer explicit target_classes; fall back to all classes present in data when missing.
        classes: Optional[List[str]] = self.target_classes
        if not classes:
            src_classes = (
                self.combined_data[UNIFIED_FACTUAL_CLASS]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            tgt_classes = (
                self.combined_data[UNIFIED_ALTERNATIVE_CLASS]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            classes = sorted(set(src_classes) | set(tgt_classes))

        targets: Dict[str, List[str]] = {}
        for class_name in classes:
            as_src = self.combined_data[self.combined_data[UNIFIED_FACTUAL_CLASS] == class_name]
            as_tgt = self.combined_data[self.combined_data[UNIFIED_ALTERNATIVE_CLASS] == class_name]
            tokens = set(as_src[UNIFIED_FACTUAL].dropna().astype(str)) | set(
                as_tgt[UNIFIED_ALTERNATIVE].dropna().astype(str)
            )
            # When combined_data only has the training pair, other classes have no rows; use per-class data
            if not tokens and getattr(self, "class_datasets", None) and class_name in self.class_datasets:
                df_c = self.class_datasets[class_name]
                factual_col = (
                    class_name
                    if class_name in df_c.columns
                    else ("label" if "label" in df_c.columns else None)
                )
                if factual_col is not None:
                    tokens = set(df_c[factual_col].dropna().astype(str))
            targets[class_name] = list(tokens)

        # Check for overlapping tokens across classes (used by resolver to decide row-wise fallback).
        token_sets: Dict[str, set] = {cls: set(v) for cls, v in targets.items()}
        overlapping_details: List[str] = []
        cls_list = list(token_sets.keys())
        for i in range(len(cls_list)):
            for j in range(i + 1, len(cls_list)):
                c1, c2 = cls_list[i], cls_list[j]
                inter = token_sets[c1] & token_sets[c2]
                if inter:
                    overlapping_details.append(f"{c1} ↔ {c2}: {sorted(inter)}")
        has_overlap = len(overlapping_details) > 0

        run_id_label = self.run_id if self.run_id is not None else "default"
        if not has_overlap:
            logger.info(f"Inferred decoder eval targets for {run_id_label}: {targets}")
        return targets, has_overlap

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
        objective = self._prediction_objective(model)

        use_cache = self._resolve_artifact_use_cache(use_cache, fallback=False)
        explicit_objective = getattr(getattr(self, "_training_args", None), "prediction_objective", "auto")
        if explicit_objective not in (None, "auto") and cache_folder:
            cache_folder = f"{cache_folder}_{objective.name}"
        elif explicit_objective not in (None, "auto"):
            cache_folder = objective.name

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

        # Resolve targets and whether to use row-wise evaluation (P(factual) vs P(alternative) per row).
        targets, use_row_wise = self._resolve_decoder_eval_targets(training_like_df)

        if not use_row_wise and not targets:
            run_id_part = f" (run_id={self.run_id})" if self.run_id is not None else ""
            raise ValueError(
                "Could not infer decoder eval targets. "
                "Set config.decoder_eval_targets explicitly or ensure data and target_classes are loaded." + run_id_part
            )

        # Restrict to target classes when using static targets
        if not use_row_wise and getattr(self.config, "decoder_eval_restrict_to_target_classes", True) and self.target_classes is not None:
            target_classes_set = frozenset(self.target_classes)
            targets = {k: v for k, v in targets.items() if k in target_classes_set}
            if not targets:
                raise ValueError(
                    f"decoder_eval_restrict_to_target_classes=True but no decoder_eval_targets for target_classes {self.target_classes}. "
                    "Ensure target classes exist in decoder_eval_targets or set decoder_eval_restrict_to_target_classes=False."
                )

        # Compute feature score: row-wise (P(factual), P(alternative) per row) or static targets
        prob_on_other_class = getattr(self.config, "decoder_eval_prob_on_other_class", True)
        label_dataset_col = None
        if "label_class" in training_like_df.columns:
            label_dataset_col = "label_class"
        elif "factual_id" in training_like_df.columns:
            label_dataset_col = "factual_id"

        # probs_by_dataset keys = factual label_class (panel title matches row filter).
        # Do NOT group by alternative_id for panels — same key names, different semantics.
        dataset_class_col = label_dataset_col

        export_row_wise_csv = use_row_wise and getattr(
            self.config, "decoder_eval_export_row_wise_csv", False
        )
        eval_out = objective.score_probability_shift(
            model,
            tokenizer,
            targets=targets or {},
            eval_data_df=training_like_df,
            key_text=self.config.masked_col,
            dataset_class_col=dataset_class_col,
            use_row_wise=use_row_wise,
            return_per_row_df=export_row_wise_csv,
            trainer=self,
        )
        if export_row_wise_csv and isinstance(eval_out, tuple) and len(eval_out) == 2 and hasattr(eval_out[1], "columns"):
            probs_by_dataset, per_row_df = eval_out
            csv_path = resolve_decoder_row_wise_csv_path(self.experiment_dir)
            if csv_path and not per_row_df.empty:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                per_row_df.to_csv(csv_path, index=False)
                logger.info("Saved row-wise decoder eval scores to %s", csv_path)
        else:
            probs_by_dataset = eval_out

        # Class names for selection metrics (from targets or from probs_by_dataset when row-wise)
        if use_row_wise and probs_by_dataset:
            _class_names = set()
            for v in probs_by_dataset.values():
                _class_names.update(v.keys())
            class_names_for_metrics = sorted(_class_names)
        else:
            class_names_for_metrics = list(targets.keys()) if targets else []

        # Strengthen + prob_on_other_class: P(target) on the other class's factual dataset
        # (e.g. P(3SG) on label_class=3PL rows → star on Dataset 3PL, 3SG line).
        # Weaken: P(class) on its own factual dataset.
        probs_factual: Dict[str, float] = {}
        if prob_on_other_class and label_dataset_col and not use_row_wise:
            counterfactual_probs = {}
            for class_name in class_names_for_metrics:
                other_classes = [c for c in class_names_for_metrics if c != class_name]
                for other in other_classes:
                    if other in probs_by_dataset and class_name in probs_by_dataset[other]:
                        counterfactual_probs[class_name] = float(probs_by_dataset[other][class_name])
                        break
                if class_name in probs_by_dataset and class_name in probs_by_dataset[class_name]:
                    probs_factual[class_name] = float(probs_by_dataset[class_name][class_name])
            probs = counterfactual_probs if counterfactual_probs else next(iter(probs_by_dataset.values())) if probs_by_dataset else {}
        else:
            probs = next(iter(probs_by_dataset.values())) if probs_by_dataset else {}
            if probs_by_dataset:
                for class_name in class_names_for_metrics:
                    if class_name in probs_by_dataset and class_name in probs_by_dataset[class_name]:
                        probs_factual[class_name] = float(probs_by_dataset[class_name][class_name])

        # Compute LMS
        ignore_tokens = self.config.decoder_eval_ignore_tokens
        if ignore_tokens is None and getattr(self.config, "eval_neutral_data", None) is None:
            ignore_set = set()
            if use_row_wise and "factual" in training_like_df.columns and "alternative" in training_like_df.columns:
                ignore_set.update(training_like_df["factual"].dropna().astype(str).unique().tolist())
                ignore_set.update(training_like_df["alternative"].dropna().astype(str).unique().tolist())
            elif targets:
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
        lms = objective.compute_lms(
            model,
            tokenizer,
            neutral_df['text'].tolist(),
            trainer=self,
            ignore=ignore_tokens,
            max_texts=max_size_neutral,
            batch_size=eval_batch_size,
        )

        # Return in standard format with new structure
        result = {
            'probs': probs,  # Counterfactual probs for selection (P(target) on other factual dataset)
            'lms': lms,
            '_probs_by_dataset_grouping': 'label_class' if label_dataset_col == 'label_class' else 'factual_id',
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
            **kwargs: Reserved for future decoder plotting options.

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
        
        # Resolve targets and row-wise mode (plotting uses same evaluation path; row-wise needs no static targets).
        targets, use_row_wise = self._resolve_decoder_eval_targets(training_like_df)
        if use_row_wise:
            extended_targets = {cls: [] for cls in class_ids}  # placeholder; row-wise uses per-row factual/alternative
        else:
            extended_targets = {cls: targets[cls] for cls in class_ids if targets and cls in targets}
        if not extended_targets and not use_row_wise:
            raise ValueError(f"Could not build targets for classes {class_ids}. Ensure decoder_eval_targets includes these classes.")
        
        # Always refresh plot panels from full factual-grouped evaluation (never reuse
        # selection-subset or legacy alternative_id-grouped probs_by_dataset).
        extended_grid = {}
        for candidate_id, entry in grid.items():
            extended_entry = dict(entry)

            if candidate_id == "base":
                eval_model = base_model
                modified_model = None
            else:
                if isinstance(candidate_id, tuple) and len(candidate_id) == 2:
                    feature_factor, lr = candidate_id
                elif isinstance(candidate_id, dict):
                    feature_factor = candidate_id.get("feature_factor")
                    lr = candidate_id.get("learning_rate")
                else:
                    logger.warning(f"Unknown candidate_id format: {candidate_id}, skipping")
                    extended_grid[candidate_id] = extended_entry
                    continue

                modified_model = model_with_gradiend.rewrite_base_model(
                    learning_rate=lr,
                    feature_factor=feature_factor,
                    part=getattr(self.config, "decoder_eval_part", "decoder"),
                )
                eval_model = modified_model

            eval_result = self.evaluate_base_model(
                eval_model,
                tokenizer,
                training_like_df=training_like_df,
                neutral_df=neutral_df,
                use_cache=False,
            )
            if "probs_by_dataset" in eval_result:
                extended_entry["probs_by_dataset"] = eval_result["probs_by_dataset"]
            grouping = eval_result.get("_probs_by_dataset_grouping")
            if grouping:
                extended_entry["_probs_by_dataset_grouping"] = grouping

            if modified_model is not None:
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
        """Encode training data via gradients and return rows with type='training'.

        Args:
            model_with_gradiend: Model used to encode gradient rows.
            train_eval_data: Evaluation dataset built from training-like rows.
            source_type: Encoder source type, such as ``"factual"`` or ``"alternative"``.
            max_size: Optional row cap for logging/metadata consistency.
            encoder_kwargs: Encoder evaluation options, including ``split``.
        """
        logger.info("Encoding training data (max_size=%s, source=%s, size=%s)", max_size, source_type, len(train_eval_data) if train_eval_data is not None else 0)
        eval_result = self.evaluator.evaluate_encoder(
            eval_data=train_eval_data,
            use_cache=False,
            split=encoder_kwargs["split"],
            max_size=max_size,
            include_other_classes=self._default_from_training_args(
                encoder_kwargs.get("include_other_classes"), "include_other_classes", fallback=False
            ),
            source=source_type,
            model_with_gradiend=model_with_gradiend,
        )
        training_rows = eval_result.get("training_rows") or []
        if not training_rows:
            def _text_row_extractor(entry: dict) -> dict:
                extra: Dict[str, Any] = {}
                text = entry.get("input_text") or entry.get("text")
                if text:
                    extra["text"] = text
                for token_key in ("factual_token", "alternative_token"):
                    if entry.get(token_key) is not None:
                        extra[token_key] = entry[token_key]
                if entry.get("source_token") is not None:
                    extra["source_token"] = entry["source_token"]
                return extra

            training_rows = encode_dataset_to_rows(
                model_with_gradiend,
                train_eval_data,
                row_extractor=_text_row_extractor,
            )

        rows: List[Dict[str, Any]] = []
        resolved_eval_group = getattr(self, "eval_group", None)
        for r in training_rows:
            if resolved_eval_group == "factual_id":
                eval_group_value = r.get("source_id")
            elif resolved_eval_group == "feature_class_id":
                eval_group_value = r.get("feature_class_id")
            else:
                raise ValueError(
                    f"Unsupported eval_group {resolved_eval_group!r}. "
                    "Expected one of: 'factual_id', 'feature_class_id'."
                )
            rows.append({
                "text": r.get("text"),
                "encoded": r["encoded"],
                "label": float(r["label"]),
                "source_id": r.get("source_id"),
                "target_id": r.get("target_id"),
                "feature_class_id": r.get("feature_class_id"),
                "eval_group": eval_group_value,
                "type": "training",
                "source_token": r.get("source_token"),
                "factual_token": r.get("factual_token"),
                "alternative_token": r.get("alternative_token"),
                "data_split": r.get("data_split"),
            })
        logger.debug(f"Processed {len(rows)} training data entries")
        return rows

    @staticmethod
    def _neutral_encoder_prediction_objective(objective_name: str) -> str:
        """Objective for neutral encoder rows.

        The auxiliary decoder MLM head only defines gradients for its trained target
        labels, so neutral encoder analysis falls back to CLM next-token gradients.
        """
        if objective_name == "clm_mlm_head":
            return "clm_next_token"
        return objective_name

    def _neutral_encoder_gradient_creator(self, model_with_gradiend: Any) -> Any:
        """Gradient creator for neutral encoder rows under clm_mlm_head training."""
        objective_name = self._prediction_objective(
            getattr(model_with_gradiend, "base_model", getattr(model_with_gradiend, "tokenizer", None))
        ).name
        if objective_name == "clm_mlm_head":
            creator = getattr(model_with_gradiend, "forward_clm_gradients", None)
            if callable(creator):
                return creator
        return model_with_gradiend.gradient_creator

    def _resolve_encoder_neutral_excluded_tokens(self) -> List[str]:
        """Tokens that must not be selected as neutral mask targets.

        Matches ``generate_neutral_data`` exclusion: all feature target tokens plus
        ``eval_neutral_additional_excluded_words`` (and optional decoder LMS ignores).
        """
        excluded: List[str] = []
        try:
            self._ensure_data()
            inferred, _ = self._infer_decoder_eval_targets()
            for tokens in inferred.values():
                excluded.extend(str(t) for t in tokens)
        except (ValueError, AttributeError, TypeError):
            pass
        raw_targets = getattr(self.config, "decoder_eval_targets", None)
        if isinstance(raw_targets, dict):
            for tokens in raw_targets.values():
                excluded.extend(str(t) for t in tokens)
        additional = getattr(self.config, "eval_neutral_additional_excluded_words", None) or []
        excluded.extend(str(w) for w in additional)
        ignore = getattr(self.config, "decoder_eval_ignore_tokens", None) or []
        excluded.extend(str(w) for w in ignore)
        seen: set = set()
        out: List[str] = []
        for word in excluded:
            key = word.lower()
            if key not in seen:
                seen.add(key)
                out.append(word)
        return out

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

        Args:
            model_with_gradiend: Model used to encode gradient rows.
            train_eval_data: Training-like evaluation dataset.
            excluded_tokens: Tokens that must not be selected as neutral mask targets.
            factual_token_key: Key containing the factual token in each dataset item.
            alternative_token_key: Key containing the alternative token in each dataset item.
            max_size: Optional row cap.
            torch_dtype: Torch dtype used when tensors are created.
            device: Device used for tensors and model execution.
        """
        # Excluded tokens must always include at least all target tokens from the training data
        target_tokens_from_data = set()
        for entry in train_eval_data:
            for key in (factual_token_key, alternative_token_key):
                if key in entry and entry[key] is not None:
                    target_tokens_from_data.add(str(entry[key]).strip())
        base_excluded = list(excluded_tokens) if excluded_tokens else []
        excluded_for_masked = list(set(base_excluded) | target_tokens_from_data)

        logger.info("Encoding neutral training masked data (max_size=%s, size=%s)", max_size, len(train_eval_data) if train_eval_data is not None else 0)
        if excluded_for_masked:
            logger.debug(f"Excluded tokens: {excluded_for_masked[:10]}..." if len(
                excluded_for_masked) > 10 else f"Excluded tokens: {excluded_for_masked}")

        tokenizer = model_with_gradiend.tokenizer
        is_seq2seq_model_value = bool(getattr(model_with_gradiend, "is_seq2seq_model", False))
        is_decoder_only_model = is_decoder_only_model_from_obj(tokenizer)
        mask_token = tokenizer.mask_token if not is_decoder_only_model else None
        objective_name = self._prediction_objective(getattr(model_with_gradiend, "base_model", tokenizer)).name
        neutral_objective_name = self._neutral_encoder_prediction_objective(objective_name)
        rhs_window = getattr(getattr(self, "_training_args", None), "decoder_sequence_cloze_rhs_window", -1)
        logger.debug(f"Tokenizer: is_decoder_only_model={is_decoder_only_model}, mask_token={mask_token}")

        # Collect training data entries and re-mask non-target tokens
        neutral_training_masked_pairs = []
        _seed = getattr(getattr(self, "training_args", None), "seed", 0) or 0
        random.seed(_seed)
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

            pair = create_masked_pair_from_text(
                unmasked_text,
                tokenizer,
                is_decoder_only_model=is_decoder_only_model,
                excluded_tokens=excluded_for_masked,
                mask_token=mask_token,
                min_prefix_tokens=5,
            )
            if pair is None:
                continue
            masked_text, original_token = pair

            if is_seq2seq_model_value and not tokenize_prediction_label(tokenizer, original_token):
                logger.debug(
                    "Skipping neutral training masked token that cannot be used as a seq2seq label: %r",
                    original_token,
                )
                continue

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
                is_seq2seq_model=is_seq2seq_model_value,
                prediction_objective=neutral_objective_name,
                rhs_window=rhs_window,
                target_key="label",
                balance_column="feature_class_id",
            )

            # Create TextGradientTrainingDataset for encoding
            neutral_training_masked_gradient_data = TextGradientTrainingDataset(
                neutral_training_masked_dataset,
                tokenizer,
                self._neutral_encoder_gradient_creator(model_with_gradiend),
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
                encoded_value = model_with_gradiend.encode(grad, return_float=True)
                rows.append(
                    gradient_entry_to_encoder_row(
                        entry,
                        encoded=encoded_value,
                        input_type="factual",
                        overrides={
                            "label": 0.0,
                            "source_id": "neutral",
                            "target_id": "neutral",
                            "type": "neutral_training_masked",
                            **neutral_encoder_row_metadata("neutral_training_masked"),
                        },
                    )
                )
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

        logger.info("Encoding neutral dataset data (size=%s)", len(neutral_data_df))
        logger.debug(f"neutral_data_df columns: {list(neutral_data_df.columns)}")

        tokenizer = model_with_gradiend.tokenizer
        is_seq2seq_model_value = bool(getattr(model_with_gradiend, "is_seq2seq_model", False))
        is_decoder_only_model = is_decoder_only_model_from_obj(tokenizer)
        mask_token = tokenizer.mask_token if not is_decoder_only_model else None
        objective_name = self._prediction_objective(getattr(model_with_gradiend, "base_model", tokenizer)).name
        neutral_objective_name = self._neutral_encoder_prediction_objective(objective_name)
        rhs_window = getattr(getattr(self, "_training_args", None), "decoder_sequence_cloze_rhs_window", -1)

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
        _seed = getattr(getattr(self, "training_args", None), "seed", 0) or 0
        random.seed(_seed)
        neutral_dataset_count = 0
        if max_size:
            neutral_data_df = neutral_data_df.sample(n=max_size, random_state=_seed).reset_index(drop=True)
        for idx, row in gradiend_tqdm(
            neutral_data_df.iterrows(),
            total=len(neutral_data_df),
            desc="Processing neutral dataset",
            leave=False,
        ):
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
            if is_seq2seq_model_value and not tokenize_prediction_label(tokenizer, neutral_token):
                logger.debug(
                    "Skipping neutral dataset token that cannot be used as a seq2seq label: %r",
                    neutral_token,
                )
                continue
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
            is_seq2seq_model=is_seq2seq_model_value,
            prediction_objective=neutral_objective_name,
            rhs_window=rhs_window,
            target_key="label",
            balance_column="feature_class_id",
        )

        # Create TextGradientTrainingDataset for encoding
        neutral_gradient_data = TextGradientTrainingDataset(
            neutral_dataset,
            tokenizer,
            self._neutral_encoder_gradient_creator(model_with_gradiend),
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
            rows.append(
                gradient_entry_to_encoder_row(
                    entry,
                    encoded=encoded_value,
                    input_type="factual",
                    overrides={
                        "label": 0.0,
                        "source_id": "neutral",
                        "target_id": "neutral",
                        "type": "neutral_dataset",
                        **neutral_encoder_row_metadata("neutral_dataset"),
                    },
                )
            )
            neutral_encoded_count += 1

        logger.debug(f"Encoded {neutral_encoded_count} neutral dataset entries")
        return rows

    def _collect_decoder_mlm_required_labels(self) -> List[str]:
        """All factual/alternative token strings that may appear during GRADIEND train/eval."""
        self._ensure_data()
        if self.combined_data is not None:
            if UNIFIED_FACTUAL not in self.combined_data.columns:
                return []
            labels = set(_normalize_mlm_label_strings(self.combined_data[UNIFIED_FACTUAL]))
            if UNIFIED_ALTERNATIVE in self.combined_data.columns:
                labels |= set(_normalize_mlm_label_strings(self.combined_data[UNIFIED_ALTERNATIVE]))
            return sorted(labels)
        if getattr(self, "class_datasets", None):
            tokens: set[str] = set()
            for df in self.class_datasets.values():
                for col in df.columns:
                    if col in ("masked", "split", "text"):
                        continue
                    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                        tokens |= set(_normalize_mlm_label_strings(df[col]))
            return sorted(tokens)
        return []

    def _decoder_mlm_coverage_dataframe(self) -> pd.DataFrame:
        """(masked, label) rows from all splits for supplementing MLM-head training."""
        self._ensure_data()
        if (
            self.combined_data is not None
            and UNIFIED_MASKED in self.combined_data.columns
            and UNIFIED_FACTUAL in self.combined_data.columns
        ):
            factual_rows = self.combined_data[[UNIFIED_MASKED, UNIFIED_FACTUAL]].rename(
                columns={UNIFIED_MASKED: "masked", UNIFIED_FACTUAL: "label"}
            )
            frames = [factual_rows]
            if UNIFIED_ALTERNATIVE in self.combined_data.columns:
                alt_rows = self.combined_data[[UNIFIED_MASKED, UNIFIED_ALTERNATIVE]].rename(
                    columns={UNIFIED_MASKED: "masked", UNIFIED_ALTERNATIVE: "label"}
                )
                frames.append(alt_rows)
            out = pd.concat(frames, ignore_index=True)
            out["label"] = out["label"].astype(str).str.strip()
            return out.drop_duplicates(subset=["masked", "label"]).reset_index(drop=True)
        if getattr(self, "class_datasets", None):
            frames: List[pd.DataFrame] = []
            masked_col = self.config.masked_col
            split_col = self.config.split_col or "split"
            use_class_cols = getattr(self.config, "use_class_names_as_columns", True)
            for split_name in order_split_names(["train", "validation", "test", "all"]):
                try:
                    frames.append(
                        all_subsets_to_mlm_df(
                            self.class_datasets,
                            split=split_name,
                            masked_col=masked_col,
                            split_col=split_col,
                            use_class_names_as_columns=use_class_cols,
                        )
                    )
                except ValueError:
                    continue
            if frames:
                out = pd.concat(frames, ignore_index=True)
                out["label"] = out["label"].astype(str).str.strip()
                return out.drop_duplicates(subset=["masked", "label"]).reset_index(drop=True)
        return pd.DataFrame(columns=["masked", "label"])

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
        [MASK]. Multi-token label strings are supported: each unique label maps to one
        classifier output on the auxiliary MLM head.

        Args:
            split: Data split(s) to use.
            max_size: Optional cap on returned rows.
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
            _seed = getattr(getattr(self, "training_args", None), "seed", 0) or 0
            out = _sample_up_to_per_group(out, 'label', max_size, _seed)
        required_labels = self._collect_decoder_mlm_required_labels()
        coverage_df = self._decoder_mlm_coverage_dataframe()
        out = _ensure_mlm_training_label_coverage(out, coverage_df, required_labels)
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
            pooling_length: Union[int, Sequence[int]] = 3,
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
                position). Default: 3. Pass a sequence (e.g. ``range(1, 7)``) to run a
                validation grid search and save ablation JSON/CSV/plot automatically.
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
        use_cache = self._resolve_artifact_use_cache(use_cache, fallback=False)
        model_use_cache = self._default_from_training_args(
            model_use_cache, "model_use_cache", fallback=False
        )

        if use_cache and has_saved_decoder_mlm_head(output):
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
                dataset_trust_remote_code=self.config.dataset_trust_remote_code,
            )
            if resolved_neutral_df is not None and len(resolved_neutral_df) > 0:
                neutral_df = resolved_neutral_df.copy()
                if max_size_neutral:
                    _seed = getattr(getattr(self, "config", None), "seed", None) or getattr(getattr(self, "training_args", None), "seed", 0) or 0
                    neutral_df = neutral_df.sample(
                        n=min(len(neutral_df), max_size_neutral), random_state=_seed
                    ).reset_index(drop=True)
            else:
                neutral_df = training_like_df.copy()

        neutral_df = self._ensure_decoder_eval_text_columns(neutral_df, tokenizer)

        return training_like_df.reset_index(drop=True), neutral_df.reset_index(drop=True)

    def _ensure_decoder_eval_text_columns(self, df: pd.DataFrame, tokenizer: Any) -> pd.DataFrame:
        """Ensure DataFrame has 'masked' and 'text' columns for decoder evaluation.

        Args:
            df: Decoder evaluation DataFrame to normalize.
            tokenizer: Tokenizer whose mask token is used to reconstruct text.
        """
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
        """Get decoder eval targets (delegates to _infer_decoder_eval_targets). Returns dict only; overlap implies row-wise at eval time."""
        targets, _ = self._infer_decoder_eval_targets()
        return targets

    def _analyze_encoder(
        self,
        model_with_gradiend: Optional[Any] = None,
        split: EncoderSplit = "test",
        neutral_data_df: Optional[pd.DataFrame] = None,
        max_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        plot: bool = False,
        include_other_classes: Optional[bool] = None,
        use_all_transitions: bool = False,
        transition_selection: Optional[List[Union[TransitionSpec, Tuple[str, str]]]] = None,
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
            include_other_classes: Coarse shortcut for including all transitions
                already present in the selected split. Ignored when
                ``transition_selection`` is provided.
            use_all_transitions: If True, include all transitions available in
                the selected split during encoder analysis.
            transition_selection: Optional explicit transition specs added to
                the target-pair transitions during encoder analysis. Non-target
                probe transitions keep label ``0``.
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

        use_cache = self._resolve_artifact_use_cache(use_cache, fallback=False)
        if max_size is None:
            max_size = self._default_from_training_args(max_size, "encoder_eval_max_size")
        include_other_classes = self._default_from_training_args(
            include_other_classes, "include_other_classes", fallback=False
        )
        if model_with_gradiend is None:
            model_with_gradiend = self.get_model()

        # Single encoder_kwargs dict: same keys used for cache path and for logic.
        # Pass the same dict to get_encodings/evaluate_encoder so the cache path matches.
        encoder_kwargs = dict(
            split=split,
            neutral_data_df=neutral_data_df,
            max_size=max_size,
            include_other_classes=include_other_classes,
            use_all_transitions=use_all_transitions,
            transition_selection=transition_selection,
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
                dataset_trust_remote_code=getattr(self.config, "dataset_trust_remote_code", None),
            )
        if encoder_kwargs["neutral_data_df"] is None and getattr(self, "training_args", None) is not None:
            encoder_kwargs["neutral_data_df"] = resolve_dataframe(
                getattr(self.training_args, "eval_neutral_data", None),
                max_rows=neutral_max_rows,
                dataset_trust_remote_code=getattr(self.training_args, "dataset_trust_remote_code", None),
            )
        neutral_data_df = encoder_kwargs["neutral_data_df"]

        # Excluded tokens for neutral encoder variants (target words + additional exclusions)
        excluded_tokens = self._resolve_encoder_neutral_excluded_tokens()

        output_path = self._encoder_cache_path(model_with_gradiend.name_or_path, **encoder_kwargs)

        # Try to load cached data
        if use_cache and output_path is not None and os.path.exists(output_path):
            logger.info("Using cached encoder analysis")
            try:
                df_cached = pd.read_csv(output_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                logger.warning(
                    "Cached encoder analysis at %s is unreadable (%s). Deleting and recomputing.",
                    output_path,
                    exc,
                )
                try:
                    os.remove(output_path)
                except OSError:
                    logger.warning("Failed to remove unreadable encoder cache at %s", output_path, exc_info=True)
                invalidate_encoder_metrics_cache(output_path)
                use_cache = False
            else:
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
                elif not self._cached_encoder_df_matches_request(
                    df_cached,
                    include_other_classes=include_other_classes,
                    use_all_transitions=use_all_transitions,
                    transition_selection=transition_selection,
                ):
                    logger.info(
                        "Cached encoder analysis at %s is pair-local but this request needs full-eval "
                        "transitions; recomputing into the same cache path.",
                        output_path,
                    )
                    invalidate_encoder_metrics_cache(output_path)
                    use_cache = False
                else:
                    return df_cached

        if not use_cache and output_path is not None and os.path.exists(output_path):
            logger.info(
                "Recomputing encoder analysis; existing cache will be overwritten after "
                "successful recomputation: %s",
                output_path,
            )

        training_config = model_with_gradiend.gradiend.kwargs.get('training', {}).get('training_args', {})
        source_type = training_config.get('source', 'factual')

        # create_eval_data only accepts split, source, max_size, include_other_classes, etc.
        # Do not pass column-override or other encoder-only kwargs (text_col, masked_col, ...).
        train_eval_data = self.create_eval_data(
            model_with_gradiend,
            split=encoder_kwargs["split"],
            source=source_type,
            max_size=encoder_kwargs.get("max_size"),
            include_other_classes=self._default_from_training_args(
                encoder_kwargs.get("include_other_classes"), "include_other_classes", fallback=False
            ),
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

        # Process excluded_tokens (already a flat list from _resolve_encoder_neutral_excluded_tokens)
        if excluded_tokens is None:
            excluded_tokens = []
        elif isinstance(excluded_tokens, dict):
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

        neutral_training_masked_count = 0
        resolved_eval_splits = resolve_encoder_splits(
            encoder_kwargs.get("split", "test"),
            available=getattr(self, "combined_data", pd.DataFrame()).get(UNIFIED_SPLIT, pd.Series(dtype=str)).tolist()
            if getattr(self, "combined_data", None) is not None
            else None,
        )
        if "test" in resolved_eval_splits:
            test_eval_data = self.create_eval_data(
                model_with_gradiend,
                split="test",
                source=source_type,
                max_size=encoder_kwargs.get("max_size"),
                include_other_classes=self._default_from_training_args(
                encoder_kwargs.get("include_other_classes"), "include_other_classes", fallback=False
            ),
            )
            neutral_training_masked_rows = self._encode_neutral_training_masked_rows(
                model_with_gradiend,
                test_eval_data,
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
            tmp_output_path = f"{output_path}.tmp.{os.getpid()}"
            try:
                df.to_csv(tmp_output_path, index=False)
                os.replace(tmp_output_path, output_path)
            finally:
                if os.path.exists(tmp_output_path):
                    try:
                        os.remove(tmp_output_path)
                    except OSError:
                        logger.warning("Failed to remove temporary encoder cache file %s", tmp_output_path, exc_info=True)
            invalidate_encoder_metrics_cache(output_path)
            logger.info(f"Saved encoder analysis results to {output_path}")

        if plot:
            self.plot_encoder_distributions(encoder_df=df)

        return df
