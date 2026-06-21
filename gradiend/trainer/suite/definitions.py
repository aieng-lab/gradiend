"""
TrainerSuite: shared orchestration for many pairwise trainer runs.

The suite mirrors the underlying trainer constructor as closely as possible:
- one trainer class
- shared constructor args / kwargs
- one child trainer per unordered pair of target classes

Pairwise studies can differ in their semantics:
- positive pairs (e.g. true vs false property classes)
- symmetric pairs (e.g. x <-> y variable contrasts)

This module keeps the shared lifecycle in TrainerSuite and exposes semantic
subclasses for analysis methods that are not valid for every pair notion.
"""

from __future__ import annotations

import copy
import gc
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type, Union

import torch
import pandas as pd

from gradiend.comparison import (
    compute_anchor_aligned_encoding_matrix,
    compute_similarity_matrix,
    compute_grouped_similarity_matrices,
    compute_cross_encoding_matrix,
    normalize_cross_encoding_rows_by_diagonal,
)
from gradiend.trainer.core.unified_data import (
    _load_dataframe_from_path,
    load_hf_per_class,
    resolve_dataframe,
)
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_SPLIT,
)
from gradiend.trainer.trainer import Trainer
from gradiend.trainer.core.multi_seed import load_seed_model_group, resolve_default_seed_selection
from gradiend.util.logging import get_logger
from gradiend.visualizer.heatmaps import plot_comparison_heatmap, plot_similarity_heatmap
from gradiend.visualizer.topk.pairwise_heatmap import plot_topk_overlap_heatmap


logger = get_logger(__name__)


def _default_child_id(pair: Tuple[str, str]) -> str:
    def slug(value: str) -> str:
        """Return a filesystem-safe class-id fragment for ``value``."""
        text = str(value).strip().replace("\\", "").replace("/", "_")
        text = re.sub(r"[^0-9A-Za-z_+\-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "class"

    return f"{slug(pair[0])}__{slug(pair[1])}"


def _default_child_label(pair: Tuple[str, str]) -> str:
    return f"{pair[0]} <-> {pair[1]}"


def _generated_incomplete_data_path(path: Any) -> Optional[str]:
    if not isinstance(path, (str, os.PathLike)):
        return None
    p = os.fspath(path)
    root, ext = os.path.splitext(p)
    return f"{root}_incomplete_classes{ext}" if ext else f"{p}_incomplete_classes"


def _load_generated_incomplete_classes_from_kwargs(trainer_kwargs: Dict[str, Any]) -> List[str]:
    config = trainer_kwargs.get("config")
    data_path = trainer_kwargs.get("data")
    label_class_col = "label_class"
    if data_path is None and config is not None:
        data_path = getattr(config, "data", None)
        label_class_col = getattr(config, "label_class_col", label_class_col)
    elif config is not None:
        label_class_col = getattr(config, "label_class_col", label_class_col)
    sidecar = _generated_incomplete_data_path(data_path)
    if sidecar is None or not os.path.isfile(sidecar):
        return []
    try:
        if str(sidecar).lower().endswith(".parquet"):
            df = pd.read_parquet(sidecar)
        else:
            df = pd.read_csv(sidecar)
    except Exception as e:
        logger.warning("Could not read generated incomplete-class sidecar %s: %s", sidecar, e)
        return []
    if label_class_col not in df.columns:
        return []
    return sorted({str(value) for value in df[label_class_col].dropna().unique().tolist()})


@dataclass(frozen=True)
class SuitePairDefinition:
    """Declarative definition of one suite child trainer.

    target_classes are the effective classes seen by the child trainer. When
    class_merge_map is set, the existing trainer-level class merge mechanism is
    used to merge multiple raw feature classes from the shared dataset into
    these effective target classes before training/evaluation.
    positive_class is only meaningful for PositiveTrainerSuite analytics.
    """

    target_classes: Tuple[str, str]
    positive_class: Optional[str] = None
    child_id: Optional[str] = None
    label: Optional[str] = None
    class_merge_map: Optional[Dict[str, List[str]]] = None
    class_merge_transition_groups: Optional[List[List[str]]] = None


@dataclass(frozen=True)
class PositiveFeatureDefinition:
    """One positive-suite feature with its orthogonal true/false feature classes."""

    positive_feature_class: str
    negative_feature_class: str
    feature_class_group: Optional[str] = None
    label: Optional[str] = None


def _infer_positive_class_for_pair(pair: Tuple[str, str]) -> str:
    a, b = pair
    if a == f"non_{b}" or a == f"non-{b}":
        return b
    if b == f"non_{a}" or b == f"non-{a}":
        return a
    return a


_normalize_cross_encoding_rows_by_diagonal = normalize_cross_encoding_rows_by_diagonal


def _resolved_shared_model_key(trainer: Trainer) -> Optional[str]:
    try:
        base_model_path = trainer.base_model_path
    except Exception:
        base_model_path = None
    if isinstance(base_model_path, str) and base_model_path.strip():
        try:
            resolved = trainer.resolve_model_path(base_model_path)
        except Exception:
            resolved = base_model_path
        if isinstance(resolved, str) and resolved.strip():
            return resolved
    model_arg = getattr(trainer, "_base_model_arg", None)
    if isinstance(model_arg, str) and model_arg.strip():
        return model_arg
    return None


def _normalize_target_pairs(value: Optional[Sequence[Sequence[Any]]]) -> Optional[List[Tuple[str, str]]]:
    if value is None:
        return None
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise TypeError("target_pairs must be a sequence of 2-item class pairs")
        pair = (str(item[0]), str(item[1]))
        if pair[0] == pair[1]:
            raise ValueError(f"target_pairs cannot contain identical classes: {pair!r}")
        pair_key = tuple(sorted(pair))
        if pair_key in seen:
            raise ValueError(f"Duplicate target pair in target_pairs: {pair!r}")
        seen.add(pair_key)
        pairs.append(pair)
    return pairs


def _normalize_class_merge_map(
    value: Optional[Dict[str, Sequence[Any]]],
    *,
    target_classes: Tuple[str, str],
) -> Optional[Dict[str, List[str]]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("class_merge_map must be a dict[str, Sequence[str]] or None")
    expected = set(target_classes)
    actual = {str(key) for key in value.keys()}
    if actual != expected:
        raise ValueError(
            "class_merge_map keys must exactly match target_classes. "
            f"Expected {sorted(expected)!r}, got {sorted(actual)!r}."
        )
    normalized: Dict[str, List[str]] = {}
    used_base_classes = set()
    for key, raw_values in value.items():
        if isinstance(raw_values, (str, bytes)):
            raise TypeError("class_merge_map values must be sequences of class ids, not strings")
        classes = [str(item) for item in raw_values]
        if not classes:
            raise ValueError(f"class_merge_map[{key!r}] must contain at least one raw class")
        duplicates = set(classes) & used_base_classes
        if duplicates:
            raise ValueError(
                f"class_merge_map assigns raw classes to multiple merged targets: {sorted(duplicates)!r}"
            )
        used_base_classes.update(classes)
        normalized[str(key)] = classes
    return normalized


def _normalize_class_merge_transition_groups(
    value: Optional[Any],
) -> Optional[List[List[str]]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("class_merge_transition_groups must be a list of class clusters or None")
    normalized: List[List[str]] = []
    for cluster in value:
        if isinstance(cluster, (str, bytes)):
            raise TypeError("class_merge_transition_groups must be a list of class clusters, not strings")
        if not isinstance(cluster, (list, tuple)):
            raise TypeError("class_merge_transition_groups clusters must be sequences of class ids")
        classes = [str(item) for item in cluster]
        if not classes:
            raise ValueError("class_merge_transition_groups clusters must not be empty")
        normalized.append(classes)
    return normalized


def _normalize_pair_definitions(
    value: Optional[Sequence[Any]],
    *,
    pair_id_fn: Callable[[Tuple[str, str]], str],
    pair_label_fn: Callable[[Tuple[str, str]], str],
) -> Optional[List[SuitePairDefinition]]:
    if value is None:
        return None
    normalized: List[SuitePairDefinition] = []
    seen_ids = set()
    for item in value:
        if isinstance(item, SuitePairDefinition):
            target_classes = tuple(str(v) for v in item.target_classes)
            if len(target_classes) != 2 or target_classes[0] == target_classes[1]:
                raise ValueError(f"Invalid SuitePairDefinition target_classes: {item.target_classes!r}")
            child_id = str(item.child_id) if item.child_id is not None else pair_id_fn((target_classes[0], target_classes[1]))
            label = str(item.label) if item.label is not None else pair_label_fn((target_classes[0], target_classes[1]))
            positive_class = str(item.positive_class) if item.positive_class is not None else None
            merge_map = _normalize_class_merge_map(item.class_merge_map, target_classes=(target_classes[0], target_classes[1]))
            transition_groups = _normalize_class_merge_transition_groups(item.class_merge_transition_groups)
        elif isinstance(item, dict):
            raw_target_classes = item.get("target_classes")
            if not isinstance(raw_target_classes, Sequence) or isinstance(raw_target_classes, (str, bytes)) or len(raw_target_classes) != 2:
                raise TypeError("pair_definitions entries must contain target_classes as a 2-item sequence")
            target_classes = (str(raw_target_classes[0]), str(raw_target_classes[1]))
            if target_classes[0] == target_classes[1]:
                raise ValueError(f"pair_definitions cannot contain identical target classes: {target_classes!r}")
            child_id = str(item.get("child_id")) if item.get("child_id") is not None else pair_id_fn(target_classes)
            label = str(item.get("label")) if item.get("label") is not None else pair_label_fn(target_classes)
            positive_class = str(item.get("positive_class")) if item.get("positive_class") is not None else None
            merge_map = _normalize_class_merge_map(item.get("class_merge_map"), target_classes=target_classes)
            transition_groups = _normalize_class_merge_transition_groups(
                item.get("class_merge_transition_groups")
            )
        else:
            raise TypeError("pair_definitions must contain dicts or SuitePairDefinition instances")
        if child_id in seen_ids:
            raise ValueError(f"Duplicate child_id in pair_definitions: {child_id!r}")
        seen_ids.add(child_id)
        normalized.append(
            SuitePairDefinition(
                target_classes=(target_classes[0], target_classes[1]),
                positive_class=positive_class,
                child_id=child_id,
                label=label,
                class_merge_map=merge_map,
                class_merge_transition_groups=transition_groups,
            )
        )
    return normalized


def _normalize_positive_feature_definitions(
    value: Optional[Sequence[Any]],
) -> Optional[List[PositiveFeatureDefinition]]:
    if value is None:
        return None
    normalized: List[PositiveFeatureDefinition] = []
    seen_positive = set()
    for item in value:
        if isinstance(item, PositiveFeatureDefinition):
            positive_feature_class = str(item.positive_feature_class)
            negative_feature_class = str(item.negative_feature_class)
            feature_class_group = str(item.feature_class_group) if item.feature_class_group is not None else None
            label = str(item.label) if item.label is not None else positive_feature_class
        elif isinstance(item, dict):
            if "positive_feature_class" not in item or "negative_feature_class" not in item:
                raise TypeError(
                    "positive_feature_definitions entries must contain positive_feature_class and negative_feature_class"
                )
            positive_feature_class = str(item["positive_feature_class"])
            negative_feature_class = str(item["negative_feature_class"])
            feature_class_group = str(item["feature_class_group"]) if item.get("feature_class_group") is not None else None
            label = str(item.get("label")) if item.get("label") is not None else positive_feature_class
        else:
            raise TypeError("positive_feature_definitions must contain dicts or PositiveFeatureDefinition instances")
        if positive_feature_class in seen_positive:
            raise ValueError(f"Duplicate positive_feature_class in positive_feature_definitions: {positive_feature_class!r}")
        seen_positive.add(positive_feature_class)
        normalized.append(
            PositiveFeatureDefinition(
                positive_feature_class=positive_feature_class,
                negative_feature_class=negative_feature_class,
                feature_class_group=feature_class_group,
                label=label,
            )
        )
    return normalized


def _normalize_target_classes(value: Optional[Sequence[Any]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError("target_classes must be a sequence of class ids, not a string")
    classes = [str(v) for v in value]
    if len(set(classes)) != len(classes):
        raise ValueError("target_classes must be unique")
    return classes


def _suite_data_reference(trainer_kwargs: Dict[str, Any]) -> Any:
    config = trainer_kwargs.get("config")
    data = trainer_kwargs.get("data")
    if data is None and config is not None:
        data = getattr(config, "data", None)
    return data, config


def _feature_classes_from_unified(df: pd.DataFrame) -> List[str]:
    classes: List[str] = []
    seen = set()
    for column_name in (UNIFIED_FACTUAL_CLASS, UNIFIED_ALTERNATIVE_CLASS):
        if column_name not in df.columns:
            continue
        for value in df[column_name].dropna().astype(str).tolist():
            if value not in seen:
                seen.add(value)
                classes.append(value)
    return classes


def _feature_classes_from_tabular(
    data: pd.DataFrame,
    *,
    config: Optional[Any],
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    column_candidates = []
    for source in (trainer_kwargs, config):
        if source is None:
            continue
        getter = source.get if isinstance(source, dict) else lambda name, default=None: getattr(source, name, default)
        for attr_name in (
            "label_class_col",
            "alternative_class_col",
            "factual_cls_col",
            "alternative_cls_col",
        ):
            value = getter(attr_name)
            if isinstance(value, str) and value.strip():
                column_candidates.append(value)
    column_candidates.extend(
        [
            UNIFIED_FACTUAL_CLASS,
            UNIFIED_ALTERNATIVE_CLASS,
            "label_class",
            "alternative_class",
            "label_cls",
            "counterfactual_cls",
            "factual_cls",
            "alternative_cls",
            "label_feature_cls",
            "counterfactual_feature_cls",
            "factual_feature_cls",
            "alternative_feature_cls",
        ]
    )
    classes: List[str] = []
    seen = set()
    for column_name in column_candidates:
        if column_name not in data.columns:
            continue
        for value in data[column_name].dropna().astype(str).tolist():
            if value not in seen:
                seen.add(value)
                classes.append(value)
    return classes


def _resolve_suite_training_view(
    *,
    trainer_cls: Type[Trainer],
    trainer_args: Tuple[Any, ...],
    trainer_kwargs: Dict[str, Any],
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame], None]:
    """Resolve suite trainer data to an in-memory view for class discovery."""
    data, config = _suite_data_reference(trainer_kwargs)
    if isinstance(data, dict):
        return data
    if isinstance(data, pd.DataFrame):
        return data

    peek = getattr(trainer_cls, "peek_unified_training_data", None)
    if peek is not None:
        try:
            unified = peek(*trainer_args, **trainer_kwargs)
            if unified is not None and len(unified) > 0:
                return unified
        except (ValueError, FileNotFoundError, TypeError, AttributeError) as exc:
            logger.debug("TrainerSuite could not peek unified training data: %s", exc)

    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_file():
            return _load_dataframe_from_path(path)

    if isinstance(data, str) and not Path(data).is_file():
        classes_to_load = trainer_kwargs.get("all_classes")
        if classes_to_load is None and config is not None:
            classes_to_load = getattr(config, "all_classes", None)
        masked_col = trainer_kwargs.get("masked_col")
        if masked_col is None and config is not None:
            masked_col = getattr(config, "masked_col", None)
        split_col = trainer_kwargs.get("split_col")
        if split_col is None and config is not None:
            split_col = getattr(config, "split_col", None)
        dataset_trust_remote_code = trainer_kwargs.get("dataset_trust_remote_code")
        if dataset_trust_remote_code is None and config is not None:
            dataset_trust_remote_code = getattr(config, "dataset_trust_remote_code", None)
        hf_splits = trainer_kwargs.get("hf_splits")
        if hf_splits is None and config is not None:
            hf_splits = getattr(config, "hf_splits", None)
        import gradiend.trainer.suite as suite_api

        return suite_api.load_hf_per_class(
            data,
            classes=classes_to_load if classes_to_load is not None else "all",
            splits=hf_splits,
            masked_col=masked_col or "masked",
            split_col=split_col or "split",
            dataset_trust_remote_code=dataset_trust_remote_code,
        )

    hf_dataset = trainer_kwargs.get("hf_dataset")
    if hf_dataset is None and config is not None:
        hf_dataset = getattr(config, "hf_dataset", None)
    if hf_dataset is not None:
        dataset_trust_remote_code = trainer_kwargs.get("dataset_trust_remote_code")
        if dataset_trust_remote_code is None and config is not None:
            dataset_trust_remote_code = getattr(config, "dataset_trust_remote_code", None)
        return resolve_dataframe(
            hf_dataset,
            dataset_trust_remote_code=dataset_trust_remote_code,
        )

    return None


def _infer_target_classes_from_pair_inputs(
    *,
    pair_definitions: Optional[Sequence[Any]] = None,
    target_pairs: Optional[Sequence[Sequence[Any]]] = None,
) -> Optional[List[str]]:
    """Collect feature classes declared by suite pair inputs without loading data."""
    classes: List[str] = []
    seen: set = set()

    def add(value: Any) -> None:
        """Append ``value`` once while preserving input order."""
        text = str(value)
        if text not in seen:
            seen.add(text)
            classes.append(text)

    if target_pairs is not None:
        for item in target_pairs:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                continue
            add(item[0])
            add(item[1])

    if pair_definitions is not None:
        for item in pair_definitions:
            if isinstance(item, SuitePairDefinition):
                for cls in item.target_classes:
                    add(cls)
                if item.class_merge_map:
                    for raw_values in item.class_merge_map.values():
                        for cls in raw_values:
                            add(cls)
            elif isinstance(item, dict):
                raw_target_classes = item.get("target_classes")
                if isinstance(raw_target_classes, Sequence) and not isinstance(raw_target_classes, (str, bytes)):
                    for cls in raw_target_classes:
                        add(cls)
                merge_map = item.get("class_merge_map")
                if isinstance(merge_map, dict):
                    for raw_values in merge_map.values():
                        if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
                            for cls in raw_values:
                                add(cls)

    if len(classes) < 2:
        return None
    return classes


def _infer_target_classes_from_inputs(
    *,
    target_classes: Optional[Sequence[Any]],
    trainer_kwargs: Dict[str, Any],
    trainer_cls: Optional[Type[Trainer]] = None,
    trainer_args: Tuple[Any, ...] = (),
    pair_definitions: Optional[Sequence[Any]] = None,
    target_pairs: Optional[Sequence[Sequence[Any]]] = None,
) -> List[str]:
    explicit = _normalize_target_classes(target_classes)
    if explicit is not None:
        return explicit

    config = trainer_kwargs.get("config")
    config_target_classes = getattr(config, "target_classes", None) if config is not None else None
    inferred = _normalize_target_classes(config_target_classes)
    if inferred:
        return inferred

    all_classes = trainer_kwargs.get("all_classes")
    if all_classes is None and config is not None:
        all_classes = getattr(config, "all_classes", None)
    inferred = _normalize_target_classes(all_classes)
    if inferred:
        return inferred

    inferred = _normalize_target_classes(
        _infer_target_classes_from_pair_inputs(
            pair_definitions=pair_definitions,
            target_pairs=target_pairs,
        )
    )
    if inferred:
        return inferred

    data, config = _suite_data_reference(trainer_kwargs)
    if isinstance(data, dict):
        inferred = _normalize_target_classes(list(data.keys()))
        if inferred:
            return inferred
    if isinstance(data, pd.DataFrame):
        inferred = _normalize_target_classes(
            _feature_classes_from_tabular(data, config=config, trainer_kwargs=trainer_kwargs)
        )
        if inferred:
            return inferred

    if trainer_cls is not None:
        resolved = _resolve_suite_training_view(
            trainer_cls=trainer_cls,
            trainer_args=trainer_args,
            trainer_kwargs=trainer_kwargs,
        )
        if isinstance(resolved, dict):
            inferred = _normalize_target_classes(list(resolved.keys()))
            if inferred:
                return inferred
        if isinstance(resolved, pd.DataFrame):
            if UNIFIED_FACTUAL_CLASS in resolved.columns:
                inferred = _normalize_target_classes(_feature_classes_from_unified(resolved))
            else:
                inferred = _normalize_target_classes(
                    _feature_classes_from_tabular(
                        resolved,
                        config=config,
                        trainer_kwargs=trainer_kwargs,
                    )
                )
            if inferred:
                return inferred

    raise ValueError(
        "TrainerSuite could not determine feature classes. Pass target_classes explicitly, "
        "set config.target_classes or all_classes, or provide data containing feature-class columns."
    )


def _infer_transition_columns(config: Optional[Any], data: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    source_col = None
    target_col = None
    split_col = None
    if config is not None:
        for left_name, right_name in (
            ("label_class_col", "alternative_class_col"),
            ("factual_cls_col", "alternative_cls_col"),
        ):
            left = getattr(config, left_name, None)
            right = getattr(config, right_name, None)
            if isinstance(left, str) and isinstance(right, str):
                source_col = left
                target_col = right
                break
        split_value = getattr(config, "split_col", None)
        if isinstance(split_value, str):
            split_col = split_value
    if hasattr(data, "columns"):
        fallback_pairs = (
            ("label_cls", "counterfactual_cls"),
            ("factual_cls", "alternative_cls"),
            ("label_feature_cls", "counterfactual_feature_cls"),
            ("factual_feature_cls", "alternative_feature_cls"),
        )
        if source_col is None or target_col is None:
            for left, right in fallback_pairs:
                if left in data.columns and right in data.columns:
                    source_col = left
                    target_col = right
                    break
        if split_col is None and "split" in data.columns:
            split_col = "split"
    return source_col, target_col, split_col


def _validate_pair_definitions_against_data(
    *,
    trainer_kwargs: Dict[str, Any],
    pair_definitions: Sequence[SuitePairDefinition],
    trainer_cls: Optional[Type[Trainer]] = None,
    trainer_args: Tuple[Any, ...] = (),
) -> None:
    config = trainer_kwargs.get("config")
    data, _ = _suite_data_reference(trainer_kwargs)
    if isinstance(data, str) and not Path(data).is_file():
        logger.debug(
            "Skipping TrainerSuite pair-definition validation for unresolved dataset id %s.",
            data,
        )
        return
    if trainer_cls is not None and not isinstance(data, pd.DataFrame):
        try:
            resolved = _resolve_suite_training_view(
                trainer_cls=trainer_cls,
                trainer_args=trainer_args,
                trainer_kwargs=trainer_kwargs,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            logger.warning(
                "TrainerSuite could not resolve training data for pair-definition validation; skipping check: %s",
                exc,
            )
            return
        if isinstance(resolved, pd.DataFrame):
            data = resolved
    if not isinstance(data, pd.DataFrame):
        return

    source_col, target_col, split_col = _infer_transition_columns(config, data)
    if (
        (source_col is None or target_col is None)
        and UNIFIED_FACTUAL_CLASS in data.columns
        and UNIFIED_ALTERNATIVE_CLASS in data.columns
    ):
        source_col = UNIFIED_FACTUAL_CLASS
        target_col = UNIFIED_ALTERNATIVE_CLASS
        if split_col is None and UNIFIED_SPLIT in data.columns:
            split_col = UNIFIED_SPLIT
    if source_col is None or target_col is None or source_col not in data.columns or target_col not in data.columns:
        return

    train_df = data
    if split_col is not None and split_col in data.columns:
        train_mask = data[split_col].astype(str).str.lower() == "train"
        if bool(train_mask.any()):
            train_df = data[train_mask]

    observed = set(
        tuple(sorted((str(src), str(tgt))))
        for src, tgt in zip(train_df[source_col].astype(str), train_df[target_col].astype(str))
        if str(src) != str(tgt)
    )
    missing: List[str] = []
    for definition in pair_definitions:
        left_name, right_name = definition.target_classes
        merge_map = definition.class_merge_map or {
            left_name: [left_name],
            right_name: [right_name],
        }
        left_raw = [str(value) for value in merge_map[left_name]]
        right_raw = [str(value) for value in merge_map[right_name]]
        found = any(tuple(sorted((left, right))) in observed for left in left_raw for right in right_raw if left != right)
        if not found:
            missing.append(f"{left_name} <-> {right_name}")
    if missing:
        missing_preview = ", ".join(missing[:8])
        raise ValueError(
            "TrainerSuite was asked to build pair definitions that are not present in the training data. "
            f"Missing train pairs: {missing_preview}. "
            "Pass pair_definitions=... with only the transitions you want to train, or adjust the input data."
        )


def _filter_generated_incomplete_pair_definition(
    definition: SuitePairDefinition,
    incomplete: set,
) -> Optional[SuitePairDefinition]:
    target_classes = tuple(str(value) for value in definition.target_classes)
    if any(value in incomplete for value in target_classes):
        return None
    merge_map = copy.deepcopy(definition.class_merge_map)
    if merge_map:
        filtered: Dict[str, List[str]] = {}
        for key, values in merge_map.items():
            kept = [str(value) for value in values if str(value) not in incomplete]
            if not kept:
                return None
            filtered[str(key)] = kept
        merge_map = filtered
    return SuitePairDefinition(
        target_classes=target_classes,
        positive_class=definition.positive_class,
        child_id=definition.child_id,
        label=definition.label,
        class_merge_map=merge_map,
        class_merge_transition_groups=copy.deepcopy(definition.class_merge_transition_groups),
    )




__all__ = [name for name in globals() if not name.startswith("__")]
