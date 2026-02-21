"""
Encoder metrics: compute and cache correlation/accuracy from encoder CSV outputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score

from gradiend.util import json_loads
from gradiend.util.logging import get_logger

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
	corr, _ = stats.pearsonr(df[label_col], df[encoded_col])
	return float(corr)


def invalidate_encoder_metrics_cache(encoded_values_csv_path: str) -> None:
	"""Delete cached metrics JSON for a given encoded values CSV path."""
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


def _compute_metrics_from_df(
	df_all: pd.DataFrame,
	*,
	neg_boundary: float = -0.5,
	pos_boundary: float = 0.5,
	neutral_boundary: float = 0.0,
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

	if "source_id" in df_all.columns and "target_id" in df_all.columns:
		if len(df_training) > 0:
			keys = df_training.apply(
				lambda r: (r["source_id"], r["target_id"])
				if r["source_id"] != r["target_id"]
				else r["source_id"],
				axis=1,
			)
			by_class = keys.value_counts()
			sample_counts["training_by_feature_class"] = {
				str(k): int(v) for k, v in by_class.items()
			}
	elif "source_id" in df_all.columns:
		if len(df_training) > 0:
			by_class = df_training["source_id"].value_counts()
			sample_counts["training_by_feature_class"] = by_class.astype(int).to_dict()
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
	for t, grp in df_dim0.groupby("type"):
		if len(grp) > 0:
			mean_by_type[str(t)] = float(grp["encoded"].astype(float).mean())

	neutral_mean_by_type: Dict[str, float] = {nt: mean_by_type[nt] for nt in neutral_types if nt in mean_by_type}

	mean_by_feature_class: Dict[str, float] = {}
	label_value_to_class_name: Dict[float, str] = {}
	if "source_id" in df_for_means.columns and len(df_for_means) > 0:
		for fc, grp in df_for_means.groupby("source_id"):
			if len(grp) > 0:
				mean_by_feature_class[str(fc)] = float(grp["encoded"].astype(float).mean())
		# Map label -> class name from source_id (factual class)
		for lbl, grp in df_for_means.groupby("label"):
			lv = float(lbl)
			fc = grp["source_id"].iloc[0] if "source_id" in grp.columns else None
			if lv == 0 or lv == 0.0:
				label_value_to_class_name[lv] = "neutral"
			elif fc is not None and isinstance(fc, str) and not fc.isdigit():
				label_value_to_class_name[lv] = str(fc)
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
	}
	if neutral_mean_by_type:
		result["neutral_mean_by_type"] = neutral_mean_by_type
	if mean_by_feature_class:
		result["mean_by_feature_class"] = mean_by_feature_class
	if label_value_to_class_name:
		result["label_value_to_class_name"] = label_value_to_class_name

	if len(all_dimension_scores) > 1:
		result.update(all_dimension_scores)

	return result


def get_encoder_metrics_from_dataframe(
	df: pd.DataFrame,
	*,
	neg_boundary: Optional[float] = -0.5,
	pos_boundary: Optional[float] = 0.5,
	neutral_boundary: float = 0.0,
) -> Dict[str, Any]:
	"""
	Compute unified encoder metrics from an encoder DataFrame. Single source of truth for
	evaluate_encoder and get_encoder_metrics.

	Args:
		df: DataFrame with columns: encoded, label; optionally type, source_id, target_id.
		neg_boundary: Lower class boundary for ternary labels (defaults to -0.5).
		pos_boundary: Upper class boundary for ternary labels (defaults to 0.5).
		neutral_boundary: Center point for neutral labels (defaults to 0.0).

	Returns:
		Dict with: n_samples, sample_counts, all_data, training_only, target_classes_only,
		boundaries, correlation (top-level = training_only.correlation), mean_by_class,
		mean_by_type; optionally neutral_mean_by_type, mean_by_feature_class,
		label_value_to_class_name. See get_model_metrics for subset metric interpretation.
	"""
	neg_val = neg_boundary if neg_boundary is not None else (neutral_boundary - 0.5)
	pos_val = pos_boundary if pos_boundary is not None else (neutral_boundary + 0.5)
	return _compute_metrics_from_df(df, neg_boundary=neg_val, pos_boundary=pos_val, neutral_boundary=neutral_boundary)


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
