"""
Split roles and validation for flexible train / validation / test workflows.

The library no longer assumes all three splits exist. A :class:`SplitPolicy` is
derived from splits present in unified data and drives which split is used for
training, in-training encoder eval, decoder eval, and split-generalization metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from gradiend.data.core.split_group_key import apply_split_group_key
from gradiend.data.core.split_ratios import min_vocabulary_keys_for_split_ratios as _min_vocabulary_keys_for_split_ratios
from gradiend.data.core.split_validation import validate_vocabulary_group_split_coverage
from gradiend.util import normalize_split_name
from gradiend.util.encoder_splits import order_split_names

SplitRole = str  # "train" | "eval" | "decoder"


@dataclass(frozen=True)
class SplitPolicy:
    """Resolved split roles given splits present in unified / encoder data."""

    available: Tuple[str, ...]

    @classmethod
    def from_available(cls, splits: Sequence[str]) -> SplitPolicy:
        normalized = [normalize_split_name(str(s)) for s in splits if str(s).strip()]
        if not normalized:
            raise ValueError("No splits available in data.")
        return cls(available=tuple(order_split_names(set(normalized))))

    def split_for_role(self, role: SplitRole) -> str:
        """Pick the best available split for a workflow role."""
        preferences = {
            "train": ["train"],
            "eval": ["validation", "test", "train"],
            "decoder": ["test", "validation", "train"],
        }
        if role not in preferences:
            raise ValueError(f"Unknown split role {role!r}. Expected one of: {list(preferences)}")
        for candidate in preferences[role]:
            if candidate in self.available:
                return candidate
        return self.available[0]

    def generalization_pair(self) -> Optional[Tuple[str, str]]:
        """Splits compared for split_generalization, or None when not defined."""
        if "train" in self.available and "test" in self.available:
            return ("train", "test")
        if len(self.available) >= 2:
            ordered = order_split_names(self.available)
            return (ordered[0], ordered[-1])
        return None

    def required_at_load(
        self,
        *,
        vocabulary_held_out: bool = False,
        do_eval: bool = True,
    ) -> Tuple[str, ...]:
        """Minimum splits that must exist before training / evaluation."""
        required: List[str] = []
        if "train" in self.available:
            required.append("train")
        elif self.available:
            required.append(self.available[0])
        if vocabulary_held_out and "test" not in self.available:
            required.append("test")
        if do_eval and "validation" not in self.available and "test" not in self.available:
            # Caller should set do_eval=False; still list what is missing for error text.
            required.append("validation_or_test")
        return tuple(dict.fromkeys(required))


def resolve_split_policy(available: Sequence[str]) -> SplitPolicy:
    return SplitPolicy.from_available(available)


def validate_data_split_policy(
    policy: SplitPolicy,
    *,
    vocabulary_held_out: bool = False,
    do_eval: bool = True,
) -> None:
    """Fail early when data splits cannot support the requested workflow."""
    available = set(policy.available)
    if "train" not in available and not available:
        raise ValueError("No splits found in data.")
    if "train" not in available:
        raise ValueError(
            f"Training requires a 'train' split. Available splits: {order_split_names(policy.available)}"
        )
    if vocabulary_held_out and "test" not in available:
        raise ValueError(
            "split_col=None uses vocabulary-held-out splits and requires a held-out 'test' split. "
            f"Available: {order_split_names(policy.available)}. "
            "Add more target tokens or adjust split ratios so test is non-empty."
        )
    if do_eval and "validation" not in available and "test" not in available:
        raise ValueError(
            "do_eval=True requires a held-out split ('validation' or 'test') for in-training encoder "
            f"monitoring. Available: {order_split_names(policy.available)}. "
            "Set TrainingArguments(do_eval=False) for train-only data, or provide validation/test rows."
        )


def pair_transition_mask(
    df: pd.DataFrame,
    target_classes: Sequence[str],
) -> pd.Series:
    """Return rows that represent direct non-identity transitions within a target pair."""
    classes = [str(c) for c in target_classes]
    if len(classes) < 2:
        return pd.Series(False, index=df.index)
    class_set = set(classes)
    return (
        (df["label"].astype(float) != 0)
        & df["source_id"].astype(str).isin(class_set)
        & df["target_id"].astype(str).isin(class_set)
    )


def min_vocabulary_keys_for_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> int:
    """Minimum distinct vocabulary keys required for non-empty train/val/test buckets."""
    return _min_vocabulary_keys_for_split_ratios(train_ratio, val_ratio, test_ratio)


def vocabulary_held_out_viable_for_target_pair(
    combined_data: pd.DataFrame,
    target_classes: Sequence[str],
    *,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    factual_class_col: str = "factual_class",
    alternative_class_col: str = "alternative_class",
    min_rows_per_class: int = 10,
    group_key=None,
) -> bool:
    """
    Whether vocabulary-held-out splits are viable for the trained target pair.

    Requires at least ``min_rows_per_class`` rows and enough distinct ``group_col``
    values per target class to fill every non-zero split bucket.
    """
    if combined_data is None or combined_data.empty:
        return False
    if not target_classes or len(target_classes) != 2:
        return False
    if group_col not in combined_data.columns or factual_class_col not in combined_data.columns:
        return False

    min_keys = min_vocabulary_keys_for_split_ratios(train_ratio, val_ratio, test_ratio)
    tc_set = {str(c) for c in target_classes}
    pair_df = combined_data[combined_data[factual_class_col].astype(str).isin(tc_set)]
    if alternative_class_col in combined_data.columns:
        pair_df = pair_df[
            pair_df[alternative_class_col].astype(str).isin(tc_set)
            & (
                pair_df[factual_class_col].astype(str)
                != pair_df[alternative_class_col].astype(str)
            )
        ]

    for cls in (str(target_classes[0]), str(target_classes[1])):
        cls_df = pair_df[pair_df[factual_class_col].astype(str) == cls]
        if len(cls_df) < min_rows_per_class:
            return False
        keys = {
            apply_split_group_key(value, group_key)
            for value in cls_df[group_col].dropna().astype(str)
        }
        if len(keys) < min_keys:
            return False
    return True


def validate_target_class_vocabulary_coverage(
    combined_data: pd.DataFrame,
    target_classes: Sequence[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    vocabulary_held_out: bool = False,
    factual_class_col: str = "factual_class",
    factual_col: str = "factual",
    split_col: str = "split",
    group_key=None,
) -> None:
    """Backward-compatible wrapper; prefer split-time validation in splitting.py."""
    if not vocabulary_held_out or combined_data is None or combined_data.empty:
        return
    if not target_classes or len(target_classes) < 2:
        return
    for cls in (str(target_classes[0]), str(target_classes[1])):
        cls_df = combined_data[combined_data[factual_class_col].astype(str) == cls]
        if cls_df.empty:
            raise ValueError(
                "Target-class vocabulary split coverage is insufficient for "
                f"vocabulary-held-out evaluation:\n  {cls!r} has no rows in unified data"
            )
        validate_vocabulary_group_split_coverage(
            cls_df,
            group_col=factual_col,
            split_col=split_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            group_key=group_key,
            class_id=cls,
        )


def validate_target_pair_encoder_split_coverage(
    encoder_df: pd.DataFrame,
    target_classes: Sequence[str],
    compared_splits: Tuple[str, str],
    *,
    split_col: str = "data_split",
    type_col: str = "type",
) -> None:
    """
  Raise when the trained target pair lacks pair-transition encoder rows in every
  compared split (e.g. no christian tokens in test after vocabulary-held-out resplit).
    """
    if encoder_df is None or encoder_df.empty:
        return
    if split_col not in encoder_df.columns:
        return
    if not target_classes or len(target_classes) < 2:
        return

    training = encoder_df[encoder_df[type_col].astype(str) == "training"]
    if training.empty:
        return

    pair_df = training[pair_transition_mask(training, target_classes)]
    if pair_df.empty:
        raise ValueError(
            "split_generalization requires encoder rows for the trained target pair transitions "
            f"({target_classes[0]!r}↔{target_classes[1]!r}), but none were found. "
            "Check include_other_classes / evaluate_encoder split selection."
        )

    problems: List[str] = []
    for cls in (str(target_classes[0]), str(target_classes[1])):
        cls_rows = pair_df[pair_df["source_id"].astype(str) == cls]
        present = set(cls_rows[split_col].dropna().astype(str).map(normalize_split_name).tolist())
        for sp in compared_splits:
            norm_sp = normalize_split_name(sp)
            if norm_sp not in present:
                problems.append(
                    f"{cls!r} has no target-pair encoder rows in split {norm_sp!r} "
                    f"(present in: {order_split_names(present) or ['<none>']})"
                )
    if problems:
        raise ValueError(
            "Target pair split coverage is insufficient for split_generalization:\n  "
            + "\n  ".join(problems)
        )
