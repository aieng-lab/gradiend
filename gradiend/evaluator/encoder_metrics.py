"""
Encoder metrics: compute and cache correlation/accuracy from encoder CSV outputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from gradiend.util.metrics import accuracy_score

from gradiend.util import json_loads
from gradiend.util.encoder_splits import order_split_names
from gradiend.util.logging import get_logger
from gradiend.util.split_policy import (
	pair_transition_mask,
	validate_target_pair_encoder_split_coverage,
)

logger = get_logger(__name__)


def get_correlation(
	df: pd.DataFrame,
	encoded_col: str = "encoded",
	label_col: str = "label",
) -> float:
	"""
	Compute correlation between encoded values and labels.

	Args:
		df: DataFrame with encoded and label columns.
		encoded_col: Column name for encoded values.
		label_col: Column name for labels.

	Returns:
		Pearson correlation coefficient.
	"""
	if len(df) < 2:
		return 0.0
	if df[label_col].std() == 0 or df[encoded_col].std() == 0:
		return 0.0
	x = np.asarray(df[label_col], dtype=float)
	y = np.asarray(df[encoded_col], dtype=float)
	corr = np.corrcoef(x, y)[0, 1]
	if np.isnan(corr):
		return 0.0
	return float(corr)


def invalidate_encoder_metrics_cache(encoded_values_csv_path: str) -> None:
	"""Delete cached metrics JSON for a given encoded-values CSV path.

	Args:
		encoded_values_csv_path: Path to the encoder CSV whose sibling JSON cache
			should be removed.
	"""
	json_path = encoded_values_csv_path.replace(".csv", ".json")
	try:
		if os.path.exists(json_path):
			os.remove(json_path)
	except OSError as e:
		logger.warning("Failed to delete metrics cache at %s (%s)", json_path, e)


def _should_use_metrics_cache(csv_path: str, json_path: str) -> bool:
	if not os.path.exists(json_path):
		return False
	try:
		return os.path.getmtime(json_path) >= os.path.getmtime(csv_path)
	except OSError:
		return False


def _split_pair_for_agreement(splits: List[str]) -> Optional[tuple[str, str]]:
	if "train" in splits and "test" in splits:
		return ("train", "test")
	if len(splits) >= 2:
		return (splits[0], splits[-1])
	return None


def _normalize_eval_group_id(value: Any) -> str:
	"""Stable string id for eval_group values (avoids ``0.0`` vs ``0``)."""
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return "unknown"
	if isinstance(value, (int, np.integer)):
		return str(int(value))
	if isinstance(value, (float, np.floating)):
		f = float(value)
		if f == int(f):
			return str(int(f))
		return str(f)
	return str(value)


def _transition_display_name(source_id: str, target_id: str) -> str:
	if source_id == target_id:
		return f"{source_id} (identity)"
	return f"{source_id}->{target_id}"


def _infer_eval_group_basis(df: pd.DataFrame) -> Optional[str]:
	"""Return ``feature_class_id``, ``factual_id``, or None."""
	if "eval_group" not in df.columns:
		return None
	mask = df["eval_group"].notna()
	if not mask.any():
		return None
	eg = df.loc[mask, "eval_group"]
	if "feature_class_id" in df.columns:
		fc = df.loc[mask, "feature_class_id"]
		if fc.notna().all() and all(
			_normalize_eval_group_id(a) == _normalize_eval_group_id(b) for a, b in zip(eg, fc)
		):
			return "feature_class_id"
	if "source_id" in df.columns:
		sid = df.loc[mask, "source_id"]
		if sid.notna().all() and all(str(a) == str(b) for a, b in zip(eg, sid)):
			return "factual_id"
	return None


def _build_eval_group_id2name(df: pd.DataFrame, basis: Optional[str]) -> Dict[str, str]:
	"""Map normalized eval_group ids to human-readable names."""
	if "eval_group" not in df.columns or basis is None:
		return {}
	id2name: Dict[str, str] = {}
	for gid, grp in df.groupby("eval_group", dropna=True):
		key = _normalize_eval_group_id(gid)
		if basis == "factual_id":
			sources = (
				grp["source_id"].dropna().astype(str).unique().tolist()
				if "source_id" in grp.columns
				else []
			)
			id2name[key] = sources[0] if len(sources) == 1 else key
		elif basis == "feature_class_id" and "source_id" in grp.columns and "target_id" in grp.columns:
			sources = grp["source_id"].dropna().astype(str).unique().tolist()
			targets = grp["target_id"].dropna().astype(str).unique().tolist()
			if len(sources) == 1 and len(targets) == 1:
				id2name[key] = _transition_display_name(sources[0], targets[0])
			else:
				id2name[key] = key
		else:
			id2name[key] = key
	return id2name


def _compute_split_generalization(
	df_for_means: pd.DataFrame,
	encoded_col: str = "encoded",
	*,
	target_classes: Optional[Sequence[str]] = None,
	compared_splits: Optional[Tuple[str, str]] = None,
) -> Optional[Dict[str, Any]]:
	"""Cross-split class-mean agreement for the trained target pair when ``data_split`` is present."""
	if "data_split" not in df_for_means.columns or "source_id" not in df_for_means.columns:
		return None
	if len(df_for_means) == 0:
		return None

	df_sg = df_for_means
	if target_classes and len(target_classes) >= 2:
		mask = pair_transition_mask(df_sg, target_classes)
		df_sg = df_sg[mask]
		if df_sg.empty:
			raise ValueError(
				"split_generalization requires encoder rows for the trained target pair, "
				f"but none were found for target_classes={list(target_classes)!r}."
			)

	splits = order_split_names(df_sg["data_split"].dropna().astype(str).unique().tolist())
	if len(splits) < 2:
		return None

	pair = compared_splits if compared_splits is not None else _split_pair_for_agreement(splits)
	if pair is None:
		return None

	if target_classes and len(target_classes) >= 2:
		validate_target_pair_encoder_split_coverage(
			df_sg.assign(type="training"),
			target_classes,
			pair,
		)

	encoded_vals = df_sg[encoded_col].astype(float)
	if len(encoded_vals) > 1:
		scale = float(encoded_vals.std())
	else:
		scale = 1.0
	epsilon = max(1e-8, 1e-6 * max(1.0, scale))

	mean_by_fc_by_split: Dict[str, Dict[str, float]] = {}
	shift_by_fc: Dict[str, float] = {}
	smd_by_fc: Dict[str, float] = {}
	agreement_by_fc: Dict[str, float] = {}

	for fc, grp in df_sg.groupby(df_sg["source_id"].astype(str)):
		by_split: Dict[str, float] = {}
		for sp, sub in grp.groupby(grp["data_split"].astype(str)):
			if len(sub) > 0:
				by_split[str(sp)] = float(sub[encoded_col].astype(float).mean())
		if len(by_split) < 2:
			continue
		mean_by_fc_by_split[str(fc)] = by_split

	pair = _split_pair_for_agreement(splits)
	if pair is None:
		return None
	s_a, s_b = pair

	for fc, by_split in mean_by_fc_by_split.items():
		mu_a = by_split.get(s_a)
		mu_b = by_split.get(s_b)
		if mu_a is None or mu_b is None:
			continue
		shift = abs(mu_a - mu_b)
		shift_by_fc[fc] = float(shift)
		denom = abs(mu_a) + abs(mu_b) + epsilon
		agreement = 1.0 - (shift / denom)
		agreement_by_fc[fc] = float(max(0.0, min(1.0, agreement)))
		sub_fc = df_sg[df_sg["source_id"].astype(str) == fc]
		pooled_std = float(sub_fc[encoded_col].astype(float).std())
		if pooled_std > 0:
			smd_by_fc[fc] = float((mu_a - mu_b) / pooled_std)
		else:
			smd_by_fc[fc] = 0.0

	if not agreement_by_fc:
		return None

	agreement_values = list(agreement_by_fc.values())
	return {
		"agreement": float(np.mean(agreement_values)),
		"agreement_worst": float(np.min(agreement_values)),
		"agreement_by_feature_class": agreement_by_fc,
		"mean_by_feature_class_by_split": mean_by_fc_by_split,
		"shift_by_feature_class": shift_by_fc,
		"smd_by_feature_class": smd_by_fc,
		"epsilon": float(epsilon),
		"splits_compared": [s_a, s_b],
		"splits_present": splits,
	}


def _compute_metrics_from_df(
	df_all: pd.DataFrame,
	*,
	neg_boundary: float = -0.5,
	pos_boundary: float = 0.5,
	neutral_boundary: float = 0.0,
	target_classes: Optional[Sequence[str]] = None,
	generalization_splits: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
	"""
	Compute encoder metrics (correlation, accuracy) from an encoder DataFrame.

	Expects columns: encoded, label, and optionally type, source_id, target_id.
	Returns the same structure as get_model_metrics.
	"""
	if len(df_all) == 0:
		raise ValueError("DataFrame is empty, cannot compute metrics")

	# Make a copy to avoid mutating the original
	df_all = df_all.copy()

	if "type" not in df_all.columns:
		# label == 0 is neutral, label != 0 is training
		df_all["type"] = df_all["label"].apply(lambda x: "neutral" if x == 0 else "training")

	def _classify_value(val: float) -> int:
		if val <= neg_boundary:
			return -1
		if val >= pos_boundary:
			return 1
		return 0

	def _classify_binary(val: float) -> int:
		return 1 if val >= neutral_boundary else -1

	n_samples = len(df_all)
	by_type = df_all["type"].astype(str).value_counts()
	sample_counts: Dict[str, Any] = {"by_type": by_type.to_dict()}

	# training_only: exclude neutral types (neutral_training_masked, neutral_dataset);
	# includes identity transitions (label 0). Ternary classification (-1, 0, 1).
	non_neutral_mask = ~df_all["type"].astype(str).str.contains("neutral", case=False, na=False)
	df_training = df_all[non_neutral_mask]

	# Fail fast: encoder correlation is undefined for invalid eval setups
	if len(df_training) == 0:
		raise ValueError(
			"All encoder eval data is neutral. Encoder correlation requires non-neutral (training-like) "
			"examples. Add eval data with distinct feature classes (e.g. 3SG vs 3PL)."
		)
	if len(df_training) < 2:
		raise ValueError(
			f"Encoder eval has only {len(df_training)} non-neutral instance(s). "
			"Correlation requires at least 2 non-neutral samples."
		)
	if df_training["label"].astype(float).std() == 0:
		raise ValueError(
			"Encoder eval has only one label class (all labels identical). "
			"Correlation requires at least two distinct classes in eval data."
		)

	# target_classes_only: same as training_only but exclude identity (label 0). Binary classification.
	target_only_mask = non_neutral_mask & (df_all["label"] != 0)
	target_only_df = df_all[target_only_mask]

	eval_group_basis: Optional[str] = None
	eval_group_id2name: Dict[str, str] = {}
	if "eval_group" in df_all.columns:
		if len(df_training) > 0:
			eval_group_basis = _infer_eval_group_basis(df_training)
			eval_group_id2name = _build_eval_group_id2name(df_training, eval_group_basis)
			by_group = df_training["eval_group"].dropna().value_counts()
			sample_counts["training_by_eval_group"] = {
				eval_group_id2name.get(_normalize_eval_group_id(gid), _normalize_eval_group_id(gid)): int(count)
				for gid, count in by_group.items()
			}
	elif "source_id" in df_all.columns:
		if len(df_training) > 0:
			by_class = df_training["source_id"].value_counts()
			sample_counts["training_by_feature_class"] = by_class.astype(int).to_dict()
			if "target_id" in df_all.columns:
				by_transition = df_training.apply(
					lambda r: (r["source_id"], r["target_id"]),
					axis=1,
				).value_counts()
				sample_counts["training_by_transition"] = {
					str(k): int(v) for k, v in by_transition.items()
				}
	else:
		target_only_df = df_all[
			df_all["type"].astype(str).str.lower().str.contains("training", na=False)
			& ~df_all["type"].astype(str).str.contains("neutral", case=False, na=False)
			& (df_all["label"] != 0)
		]

	# Multi-dimensional encodings: "encoded" may be JSON list or float
	if pd.api.types.is_float_dtype(df_all["encoded"]):
		df_all["encoded_list"] = df_all["encoded"].apply(lambda x: [x])
	else:
		df_all["encoded_list"] = df_all["encoded"].apply(json_loads)

	first_len = len(df_all["encoded_list"].iloc[0])
	all_dimension_scores: Dict[int, Dict[str, Any]] = {}

	for dim in range(first_len):
		df_all["encoded"] = df_all["encoded_list"].apply(lambda x: x[dim])

		if "label" not in df_all.columns:
			raise ValueError("DataFrame must have 'label' column for metrics computation")

		if len(df_all) > 0:
			pearson_all = get_correlation(df_all)
			df_all_labels = df_all["label"].astype(float).tolist()
			df_all_preds = df_all["encoded"].astype(float).tolist()
			actual_all = [_classify_value(v) for v in df_all_labels]
			pred_all = [_classify_value(v) for v in df_all_preds]
			acc_all = accuracy_score(actual_all, pred_all)
		else:
			pearson_all = 0.0
			acc_all = 0.0

		if len(df_training) > 0:
			pearson_training = get_correlation(df_training)
			df_training_labels = df_training["label"].astype(float).tolist()
			df_training_preds = df_training["encoded"].astype(float).tolist()
			actual_training = [_classify_value(v) for v in df_training_labels]
			pred_training = [_classify_value(v) for v in df_training_preds]
			acc_training = accuracy_score(actual_training, pred_training)
		else:
			pearson_training = 0.0
			acc_training = 0.0

		if len(target_only_df) > 0:
			pearson_target_only = get_correlation(target_only_df)
			labels_target_only = target_only_df["label"].astype(float).tolist()
			preds_target_only = target_only_df["encoded"].astype(float).tolist()
			actual_target_only = [_classify_binary(v) for v in labels_target_only]
			pred_target_only = [_classify_binary(v) for v in preds_target_only]
			acc_target_only = accuracy_score(actual_target_only, pred_target_only)
		else:
			pearson_target_only = 0.0
			acc_target_only = 0.0

		all_dimension_scores[dim] = {
			"all_data": {
				"correlation": float(pearson_all),
				"accuracy": float(acc_all),
			},
			"training_only": {
				"correlation": float(pearson_training),
				"accuracy": float(acc_training),
			},
			"target_classes_only": {
				"correlation": float(pearson_target_only),
				"accuracy": float(acc_target_only),
			},
		}

	# Per-class and per-type aggregates (using first dimension for multi-dim)
	# Include identity classes (label 0, type "training") for convergence plot.
	df_dim0 = df_all.copy()
	df_dim0["encoded"] = df_dim0["encoded_list"].apply(lambda x: x[0])
	mean_aggregate_mask = ~df_all["type"].astype(str).str.contains("neutral", case=False, na=False)
	df_for_means = df_dim0[mean_aggregate_mask]
	neutral_types = ("neutral_training_masked", "neutral_dataset")

	mean_by_class: Dict[float, float] = {}
	if len(df_for_means) > 0 and "label" in df_for_means.columns:
		for lbl, grp in df_for_means.groupby("label"):
			if len(grp) > 0:
				mean_by_class[float(lbl)] = float(grp["encoded"].astype(float).mean())

	mean_by_type: Dict[str, float] = {}
	abs_mean_by_type: Dict[str, float] = {}
	for t, grp in df_dim0.groupby("type"):
		if len(grp) > 0:
			mean_by_type[str(t)] = float(grp["encoded"].astype(float).mean())
			abs_mean_by_type[str(t)] = float(grp["encoded"].astype(float).abs().mean())


	neutral_mean_by_type: Dict[str, float] = {nt: mean_by_type[nt] for nt in neutral_types if nt in mean_by_type}

	mean_by_feature_class: Dict[str, float] = {}
	mean_by_eval_group: Dict[str, float] = {}
	label_value_to_class_name: Dict[float, str] = {}
	if "eval_group" in df_for_means.columns and len(df_for_means) > 0:
		if eval_group_basis is None:
			eval_group_basis = _infer_eval_group_basis(df_for_means)
			eval_group_id2name = _build_eval_group_id2name(df_for_means, eval_group_basis)
		for group_value, grp in df_for_means.groupby("eval_group"):
			if len(grp) > 0:
				group_id = _normalize_eval_group_id(group_value)
				group_name = eval_group_id2name.get(group_id, group_id)
				mean_by_eval_group[group_name] = float(grp["encoded"].astype(float).mean())
	if "source_id" in df_for_means.columns and len(df_for_means) > 0:
		for fc, grp in df_for_means.groupby("source_id"):
			if len(grp) > 0:
				mean_by_feature_class[str(fc)] = float(grp["encoded"].astype(float).mean())
		# Map label -> class name from source_id (factual class). When multiple
		# feature classes share the same label (e.g. include_other_classes=True),
		# avoid pretending the aggregate +/-1 mean belongs to one specific class.
		for lbl, grp in df_for_means.groupby("label"):
			lv = float(lbl)
			if lv == 0 or lv == 0.0:
				label_value_to_class_name[lv] = "neutral"
				continue
			source_ids = [
				str(fc)
				for fc in grp.get("source_id", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
			]
			source_ids = [fc for fc in source_ids if fc]
			if len(source_ids) == 1 and not source_ids[0].isdigit():
				label_value_to_class_name[lv] = source_ids[0]
			elif len(source_ids) > 1:
				label_value_to_class_name[lv] = "positive" if lv > 0 else "negative"
			else:
				label_value_to_class_name[lv] = str(lv)

	# Top-level correlation = training_only (for callbacks, best-checkpoint selection)
	corr_training = all_dimension_scores[0]["training_only"]["correlation"]

	result: Dict[str, Any] = {
		"n_samples": n_samples,
		"sample_counts": sample_counts,
		"all_data": all_dimension_scores[0]["all_data"],
		"training_only": all_dimension_scores[0]["training_only"],
		"target_classes_only": all_dimension_scores[0]["target_classes_only"],
		"boundaries": {
			"neg_boundary": neg_boundary,
			"pos_boundary": pos_boundary,
			"neutral_boundary": neutral_boundary,
		},
		"correlation": corr_training,
		"mean_by_class": mean_by_class,
		"mean_by_type": mean_by_type,
		"abs_mean_by_type": abs_mean_by_type,
	}
	if neutral_mean_by_type:
		result["neutral_mean_by_type"] = neutral_mean_by_type
	if mean_by_feature_class:
		result["mean_by_feature_class"] = mean_by_feature_class
	if mean_by_eval_group:
		result["mean_by_eval_group"] = mean_by_eval_group
	if eval_group_basis:
		result["eval_group_basis"] = eval_group_basis
	if label_value_to_class_name:
		result["label_value_to_class_name"] = label_value_to_class_name

	mean_by_target: Dict[str, Dict[str, float]] = {}
	if (
		"factual_token" in df_for_means.columns
		and "source_id" in df_for_means.columns
		and len(df_for_means) > 0
	):
		token_mask = df_for_means["factual_token"].notna() & (
			df_for_means["factual_token"].astype(str).str.strip() != ""
		)
		df_targets = df_for_means[token_mask]
		for (fc, tok), grp in df_targets.groupby(
			[df_targets["source_id"].astype(str), df_targets["factual_token"].astype(str)],
			sort=False,
		):
			if len(grp) > 0:
				mean_by_target.setdefault(str(fc), {})[str(tok)] = float(
					grp["encoded"].astype(float).mean()
				)
	if mean_by_target:
		result["mean_by_target"] = mean_by_target

	split_generalization = _compute_split_generalization(
		df_for_means,
		target_classes=target_classes,
		compared_splits=generalization_splits,
	)
	if split_generalization:
		result["split_generalization"] = split_generalization

	if len(all_dimension_scores) > 1:
		result.update(all_dimension_scores)

	return result


def get_encoder_metrics_from_dataframe(
	df: pd.DataFrame,
	*,
	neg_boundary: Optional[float] = -0.5,
	pos_boundary: Optional[float] = 0.5,
	neutral_boundary: float = 0.0,
	target_classes: Optional[Sequence[str]] = None,
	generalization_splits: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
	"""
	Compute unified encoder metrics from an encoder DataFrame. Single source of truth for
	evaluate_encoder and get_encoder_metrics.

	Args:
		df: DataFrame with columns: encoded, label; optionally type, source_id, target_id.
		neg_boundary: Lower class boundary for ternary labels (defaults to -0.5).
		pos_boundary: Upper class boundary for ternary labels (defaults to 0.5).
		neutral_boundary: Center point for neutral labels (defaults to 0.0).
		target_classes: Optional target class ids used to identify target-class
			transitions and label-value display names.
		generalization_splits: Optional pair of split names to compare for
			split-generalization metrics, typically ``("train", "test")`` or
			``("validation", "test")``. When set and the DataFrame contains
			``data_split`` plus target/group columns, the result may include a
			``split_generalization`` block.

	Returns:
		Dict with: n_samples, sample_counts, all_data, training_only, target_classes_only,
		boundaries, correlation (top-level = training_only.correlation), mean_by_class,
		mean_by_type; optionally neutral_mean_by_type, mean_by_feature_class,
		mean_by_target (feature class -> target token -> mean encoded),
		split_generalization, label_value_to_class_name. See get_model_metrics
		for subset metric interpretation.
	"""
	neg_val = neg_boundary if neg_boundary is not None else (neutral_boundary - 0.5)
	pos_val = pos_boundary if pos_boundary is not None else (neutral_boundary + 0.5)
	return _compute_metrics_from_df(
		df,
		neg_boundary=neg_val,
		pos_boundary=pos_val,
		neutral_boundary=neutral_boundary,
		target_classes=target_classes,
		generalization_splits=generalization_splits,
	)


def get_model_metrics(
	encoded_values_csv_path: str,
	*,
	use_cache: Optional[bool] = None,
	neg_boundary: Optional[float] = -0.5,
	pos_boundary: Optional[float] = 0.5,
	neutral_boundary: float = 0.0,
) -> Any:
	"""
	Compute encoder metrics from CSV(s): correlation (Pearson) and accuracy over all data and training-only.

	Note: "correlation" refers to Pearson correlation.
	Note: Neutral examples are identified by label == 0 (independent of neutral_boundary).

	Pass one or more paths to encoder CSV files. If the corresponding .json
	cache exists, it is loaded; otherwise metrics are computed and written to
	<path>.replace('.csv', '.json').

	Args:
		encoded_values_csv_path: Path to CSV file(s) with encoded values.
		use_cache: If True, always use cached metrics when present. If False, recompute.
			If None, use cache only when it is newer than the CSV.
		neg_boundary: Lower class boundary for ternary labels (defaults to -0.5).
		pos_boundary: Upper class boundary for ternary labels (defaults to 0.5).
		neutral_boundary: Center point for neutral labels (defaults to 0.0).

	Returns:
		- Single path: dict with keys:
		  - n_samples: total number of rows in the CSV.
		  - sample_counts: breakdown by "type" and (when available) by feature class.
		  - all_data: metrics over all rows; contains "correlation" (Pearson) and "accuracy"
		    using ternary classification with neg/pos boundaries.
		  - training_only: metrics over training-like data (excludes type neutral, includes identity transitions
		    with label 0). Uses ternary classification (-1, 0, 1) with neg/pos boundaries.
		  - target_classes_only: metrics over target class transitions only (excludes type neutral and label 0).
		    Uses binary classification (pred ≥ neutral_boundary → 1, else -1).
		  - per-dimension entries: when encodings are multi-dimensional, integer keys
		    0, 1, ... each mapping to a dict with "all_data", "training_only", and
		    "target_classes_only" metrics.
		- Multiple paths: dict mapping path -> metrics dict (same schema as above).
	"""
	csv_path = encoded_values_csv_path
	json_path = csv_path.replace(".csv", ".json")

	if use_cache is True and os.path.exists(json_path):
		with open(json_path, "r") as f:
			return json.load(f)
	if use_cache is None and _should_use_metrics_cache(csv_path, json_path):
		with open(json_path, "r") as f:
			return json.load(f)

	logger.info("Computing model metrics for %s", csv_path)
	df_all = pd.read_csv(csv_path)

	if len(df_all) == 0:
		raise ValueError(f"CSV file {csv_path} is empty, cannot compute metrics")

	neg_boundary_val = neg_boundary if neg_boundary is not None else (neutral_boundary - 0.5)
	pos_boundary_val = pos_boundary if pos_boundary is not None else (neutral_boundary + 0.5)

	result = _compute_metrics_from_df(
		df_all,
		neg_boundary=neg_boundary_val,
		pos_boundary=pos_boundary_val,
		neutral_boundary=neutral_boundary,
	)

	os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
	with open(json_path, "w") as f:
		json.dump(result, f, indent=2)

	return result



__all__ = [
	"get_model_metrics",
	"get_encoder_metrics_from_dataframe",
	"get_correlation",
	"invalidate_encoder_metrics_cache",
]
