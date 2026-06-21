"""Core data loading, splitting, and data creator protocols."""

from gradiend.data.core.balancing import (
    balance_dataframe,
    balance_dataframe_per_target,
    balance_dataframe_per_target_with_floor,
    cap_dataframe_balanced,
)
from gradiend.data.core.base_loader import resolve_base_data
from gradiend.data.core.dataframe_splitting import (
    split_dataframe,
    split_dataframe_by_group_key,
    split_dataframe_per_group,
)
from gradiend.data.core.data_creator_protocol import DataCreator
from gradiend.data.core.split_group_key import (
    SplitGroupKey,
    apply_split_group_key,
    normalize_split_group_key,
)
from gradiend.data.core.split_ratios import (
    SplitRatiosInput,
    min_vocabulary_keys_for_split_ratios,
    normalize_split_ratios,
)
from gradiend.data.core.split_validation import (
    validate_row_split_coverage,
    validate_vocabulary_group_split_coverage,
)
from gradiend.data.core.unified_splitting import (
    align_unified_alternatives_with_split_vocab,
    resplit_unified_dataframe,
)

__all__ = [
    "resolve_base_data",
    "DataCreator",
    "SplitGroupKey",
    "SplitRatiosInput",
    "normalize_split_group_key",
    "apply_split_group_key",
    "normalize_split_ratios",
    "min_vocabulary_keys_for_split_ratios",
    "validate_vocabulary_group_split_coverage",
    "validate_row_split_coverage",
    "split_dataframe",
    "split_dataframe_by_group_key",
    "split_dataframe_per_group",
    "resplit_unified_dataframe",
    "align_unified_alternatives_with_split_vocab",
    "balance_dataframe",
    "balance_dataframe_per_target",
    "balance_dataframe_per_target_with_floor",
    "cap_dataframe_balanced",
]
