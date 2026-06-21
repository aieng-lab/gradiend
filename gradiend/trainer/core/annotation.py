"""
Base-model annotation helpers for Trainer.

The mixin keeps annotation I/O, probability-column naming, and summary creation
out of the core training/orchestration class.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from gradiend.trainer.text.prediction.decoder_eval_utils import annotate_text_probability_rows
from gradiend.util.logging import get_logger
from gradiend.util.paths import resolve_annotated_data_csv_path, resolve_annotated_data_json_path

logger = get_logger(__name__)


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("Failed to remove temporary JSON file %s", tmp_path, exc_info=True)


class TrainerAnnotationMixin:
    """Mixin implementing ``Trainer.annotate_data`` and its helper methods."""

    @staticmethod
    def _make_safe_token_map(
        tokens: List[str],
        string_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        safe_to_token: Dict[str, str] = {}
        used: set = set()
        normalized_map = dict(string_map or {})
        for token in tokens:
            raw = str(token)
            mapped = raw
            for source in sorted(normalized_map.keys(), key=len, reverse=True):
                mapped = mapped.replace(source, str(normalized_map[source]))
            base = re.sub(r"[^0-9A-Za-z]+", "_", mapped.strip()).strip("_").lower()
            if not base:
                base = "tok"
            digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
            candidate = base if base not in used else f"{base}_{digest}"
            if candidate in used:
                candidate = f"tok_{digest}"
            safe_to_token[candidate] = raw
            used.add(candidate)
        return safe_to_token

    def _load_base_annotation_model(self) -> Any:
        if self.model_path == self.base_model_path and self._model_instance is not None:
            return self._model_instance
        kwargs: Dict[str, Any] = {"definition": self}
        if self._training_args is not None:
            kwargs["training_args"] = self._training_args
            kwargs["trust_remote_code"] = getattr(self._training_args, "trust_remote_code", False)
        return super().create_model_with_gradiend(self.base_model_path, **kwargs)

    @staticmethod
    def _mean_map(df: pd.DataFrame, columns: Dict[str, str], mask: Optional[pd.Series] = None) -> Dict[str, float]:
        if mask is not None:
            df = df.loc[mask.fillna(False)]
        return {
            key: TrainerAnnotationMixin._rounded_mean(df[col]) if col in df.columns and len(df) > 0 else 0.0
            for key, col in columns.items()
        }

    @staticmethod
    def _rounded_mean(series: pd.Series) -> float:
        if series is None or len(series) == 0:
            return 0.0
        value = series.mean()
        if pd.isna(value):
            return 0.0
        return round(float(value), 12)

    def _build_annotation_summary(
        self,
        annotated_df: pd.DataFrame,
        *,
        target_token_map: Dict[str, str],
        targets_by_class: Dict[str, List[str]],
        token_columns: Dict[str, str],
        class_columns: Dict[str, str],
        unicode_string_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        factual_token_col = "factual" if "factual" in annotated_df.columns else None
        alternative_token_col = "alternative" if "alternative" in annotated_df.columns else None
        factual_class_col = None
        for candidate in ("factual_id", "label_class", "factual_cls"):
            if candidate in annotated_df.columns:
                factual_class_col = candidate
                break
        alternative_class_col = None
        for candidate in ("alternative_id", "alternative_cls"):
            if candidate in annotated_df.columns:
                alternative_class_col = candidate
                break

        summary = {
            "n_rows": int(len(annotated_df)),
            "trainer_type": type(self).__name__,
            "base_model_path": self.base_model_path,
            "target_classes": list(self.target_classes or []),
            "targets_by_class": {str(k): [str(v) for v in vals] for k, vals in targets_by_class.items()},
            "target_token_map": dict(target_token_map),
            "mean_p_target": self._mean_map(annotated_df, token_columns),
            "mean_p_class": self._mean_map(annotated_df, class_columns),
        }
        if unicode_string_map:
            summary["unicode_string_map"] = {str(k): str(v) for k, v in unicode_string_map.items()}

        if factual_token_col is not None:
            summary["mean_p_target_factual"] = {
                safe: self._rounded_mean(annotated_df.loc[
                    annotated_df[factual_token_col].astype(str) == token,
                    column,
                ]) if column in annotated_df.columns else 0.0
                for safe, token in target_token_map.items()
                for column in [token_columns[safe]]
            }
        if alternative_token_col is not None:
            summary["mean_p_target_alternative"] = {
                safe: self._rounded_mean(annotated_df.loc[
                    annotated_df[alternative_token_col].astype(str) == token,
                    column,
                ]) if column in annotated_df.columns else 0.0
                for safe, token in target_token_map.items()
                for column in [token_columns[safe]]
            }
        if factual_class_col is not None:
            summary["mean_p_class_factual"] = {
                class_name: self._rounded_mean(annotated_df.loc[
                    annotated_df[factual_class_col].astype(str) == class_name,
                    column,
                ]) if column in annotated_df.columns else 0.0
                for class_name, column in class_columns.items()
            }
        if alternative_class_col is not None:
            summary["mean_p_class_alternative"] = {
                class_name: self._rounded_mean(annotated_df.loc[
                    annotated_df[alternative_class_col].astype(str) == class_name,
                    column,
                ]) if column in annotated_df.columns else 0.0
                for class_name, column in class_columns.items()
            }

        token_factual_col = factual_token_col or factual_class_col
        token_alternative_col = alternative_token_col or alternative_class_col
        if token_factual_col is not None:
            summary["mean_p_target_factual"] = {
                safe: self._rounded_mean(annotated_df.loc[
                    annotated_df[token_factual_col].astype(str) == token,
                    column,
                ]) if column in annotated_df.columns else 0.0
                for safe, token in target_token_map.items()
                for column in [token_columns[safe]]
            }
            summary["mean_p_target_expected"] = dict(summary["mean_p_target_factual"])
        if token_alternative_col is not None:
            summary["mean_p_target_alternative"] = {
                safe: self._rounded_mean(annotated_df.loc[
                    annotated_df[token_alternative_col].astype(str) == token,
                    column,
                ]) if column in annotated_df.columns else 0.0
                for safe, token in target_token_map.items()
                for column in [token_columns[safe]]
            }
        if factual_class_col is not None:
            summary["mean_p_class_expected"] = dict(summary["mean_p_class_factual"])

        return summary

    @staticmethod
    def _add_expected_probability_columns(
        annotated_df: pd.DataFrame,
        *,
        target_token_map: Dict[str, str],
        token_columns: Dict[str, str],
        class_columns: Dict[str, str],
    ) -> pd.DataFrame:
        df = annotated_df.copy()

        token_to_column = {
            original_token: token_columns[safe_token]
            for safe_token, original_token in target_token_map.items()
            if safe_token in token_columns
        }

        def _pick_from_row(row: pd.Series, source_col: str, mapping: Dict[str, str]) -> float:
            value = row.get(source_col)
            if pd.isna(value):
                return 0.0
            col = mapping.get(str(value))
            if col is None or col not in row or pd.isna(row[col]):
                return 0.0
            return float(row[col])

        if "factual" in df.columns:
            df["base_p_factual"] = df.apply(
                lambda row: _pick_from_row(row, "factual", token_to_column),
                axis=1,
            )
        if "alternative" in df.columns:
            df["base_p_alternative"] = df.apply(
                lambda row: _pick_from_row(row, "alternative", token_to_column),
                axis=1,
            )

        class_to_column = {class_name: col for class_name, col in class_columns.items()}
        for source_col, output_col in (
            ("factual_id", "base_p_factual_class"),
            ("label_class", "base_p_factual_class"),
            ("factual_cls", "base_p_factual_class"),
            ("alternative_id", "base_p_alternative_class"),
            ("alternative_cls", "base_p_alternative_class"),
        ):
            if source_col in df.columns and output_col not in df.columns:
                df[output_col] = df.apply(
                    lambda row: _pick_from_row(row, source_col, class_to_column),
                    axis=1,
                )

        if "base_p_factual" not in df.columns and "base_p_factual_class" in df.columns:
            df["base_p_factual"] = df["base_p_factual_class"]
        if "base_p_alternative" not in df.columns and "base_p_alternative_class" in df.columns:
            df["base_p_alternative"] = df["base_p_alternative_class"]

        return df

    def _annotate_text_prediction_rows(
        self,
        df: pd.DataFrame,
        *,
        model: Any,
        tokenizer: Any,
        targets_by_class: Dict[str, List[str]],
        eval_batch_size: Optional[int] = None,
        unicode_string_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str], Dict[str, str]]:
        masked_col = getattr(getattr(self, "config", None), "masked_col", "masked")
        annotated_probs_df, target_token_map = annotate_text_probability_rows(
            model,
            tokenizer,
            df,
            targets=targets_by_class,
            key_text=masked_col,
            batch_size=eval_batch_size or self._default_from_training_args(None, "eval_batch_size", fallback=32),
            safe_token_map=self._make_safe_token_map(
                sorted({str(token) for values in targets_by_class.values() for token in values}),
                string_map=unicode_string_map,
            ),
            prefix="base_",
        )
        annotated_df = pd.concat([df.reset_index(drop=True), annotated_probs_df.reset_index(drop=True)], axis=1)
        token_columns = {safe: f"base_p_target_{safe}" for safe in target_token_map.keys()}
        class_columns = {str(cls): f"base_p_class_{cls}" for cls in targets_by_class.keys()}
        return annotated_df, target_token_map, token_columns, class_columns

    def _annotate_text_classification_rows(
        self,
        df: pd.DataFrame,
        *,
        model: Any,
        tokenizer: Any,
        targets_by_class: Dict[str, List[str]],
        eval_batch_size: Optional[int] = None,
        unicode_string_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str], Dict[str, str]]:
        text_col = "text" if "text" in df.columns else getattr(getattr(self, "config", None), "text_col", "text")
        if text_col not in df.columns:
            raise ValueError(
                f"annotate_data() expected a text column for classification-style annotation, got columns: {list(df.columns)}"
            )
        rows = df.reset_index(drop=True)
        texts = rows[text_col].astype(str).tolist()
        batch_size = eval_batch_size or self._default_from_training_args(None, "eval_batch_size", fallback=32)
        device = next(model.parameters()).device
        class_names = [str(name) for name in targets_by_class.keys()]
        safe_token_map = self._make_safe_token_map(class_names, string_map=unicode_string_map)
        token_columns = {safe: f"base_p_target_{safe}" for safe in safe_token_map.keys()}
        class_columns = {class_name: f"base_p_class_{class_name}" for class_name in class_names}

        prob_rows: List[Dict[str, float]] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                if not hasattr(out, "logits") or out.logits is None:
                    raise AttributeError("annotate_data() requires classification logits for classification-style annotation.")
                probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
                id2label = getattr(self, "_id2label", None) or {
                    idx: name for idx, name in enumerate(class_names)
                }
                label_to_idx = {str(label): idx for idx, label in id2label.items()}
                for row_probs in probs:
                    record: Dict[str, float] = {}
                    for class_name in class_names:
                        idx = label_to_idx.get(class_name)
                        prob = float(row_probs[idx]) if idx is not None and idx < len(row_probs) else 0.0
                        record[class_columns[class_name]] = prob
                    for safe, token in safe_token_map.items():
                        record[token_columns[safe]] = record.get(class_columns[token], 0.0)
                    prob_rows.append(record)

        annotated_df = pd.concat([rows, pd.DataFrame(prob_rows)], axis=1)
        return annotated_df, safe_token_map, token_columns, class_columns

    def annotate_data(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        split: Optional[str] = None,
        output_csv: Optional[str] = None,
        output_json: Optional[str] = None,
        max_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        unicode_string_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Annotate evaluation-like data with base-model target and class probabilities.

        Args:
            data: Optional DataFrame to annotate. When omitted, uses combined
                trainer data or decoder-evaluation data.
            split: Optional split filter applied when ``data`` has a split column.
            output_csv: Optional explicit CSV output path.
            output_json: Optional explicit summary JSON output path.
            max_size: Optional cap on rows to annotate.
            eval_batch_size: Optional batch size for probability evaluation.
            use_cache: If True, load cached annotation CSV/JSON when present.
            unicode_string_map: Optional mapping used to make class/token column
                names safe and reversible.

        Returns:
            Dict containing the annotated DataFrame and summary JSON payload.
        """
        use_cache = self._resolve_artifact_use_cache(use_cache, fallback=False)
        csv_path = resolve_annotated_data_csv_path(self.experiment_dir, output_csv)
        json_path = resolve_annotated_data_json_path(self.experiment_dir, output_json)

        if use_cache and csv_path and json_path and os.path.isfile(csv_path) and os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            return {
                "annotated_df": pd.read_csv(csv_path),
                "summary": summary,
            }

        annotation_model = self._load_base_annotation_model()
        tokenizer = getattr(annotation_model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("annotate_data() requires the annotation model to expose a tokenizer.")

        original_target_classes = list(self.target_classes) if self.target_classes is not None else None
        original_config_target_classes = (
            list(getattr(self.config, "target_classes"))
            if hasattr(self, "config") and getattr(self.config, "target_classes", None) is not None
            else None
        )
        try:
            if data is None:
                combined_df = getattr(self, "combined_data", None)
                if combined_df is not None:
                    training_like_df = combined_df.copy()
                    split_col = getattr(getattr(self, "config", None), "split_col", "split")
                    if split_col in training_like_df.columns:
                        if split is not None:
                            split_mask = training_like_df[split_col].astype(str).str.lower() == str(split).lower()
                            training_like_df = training_like_df.loc[split_mask].reset_index(drop=True)
                        else:
                            eval_mask = training_like_df[split_col].astype(str).str.lower().isin(("test", "validation"))
                            if bool(eval_mask.any()):
                                training_like_df = training_like_df.loc[eval_mask].reset_index(drop=True)
                    if max_size is not None and len(training_like_df) > max_size:
                        training_like_df = training_like_df.sample(n=max_size, random_state=42).reset_index(drop=True)
                    if "text_factual" in training_like_df.columns and "text" not in training_like_df.columns:
                        training_like_df["text"] = training_like_df["text_factual"]
                    if "factual_cls" in training_like_df.columns and "factual_id" not in training_like_df.columns:
                        training_like_df["factual_id"] = training_like_df["factual_cls"]
                    if "alternative_cls" in training_like_df.columns and "alternative_id" not in training_like_df.columns:
                        training_like_df["alternative_id"] = training_like_df["alternative_cls"]
                    if "label_class" not in training_like_df.columns and "factual_id" in training_like_df.columns:
                        training_like_df["label_class"] = training_like_df["factual_id"]
                else:
                    training_like_df, _ = self._get_decoder_eval_dataframe(
                        tokenizer,
                        max_size_training_like=max_size,
                        max_size_neutral=max_size,
                    )
            else:
                training_like_df = data.copy()
                if split is not None and "split" in training_like_df.columns:
                    training_like_df = training_like_df[
                        training_like_df["split"].astype(str).str.lower() == str(split).lower()
                    ].reset_index(drop=True)

            if training_like_df is None or len(training_like_df) == 0:
                raise ValueError("annotate_data() received no rows to annotate.")

            targets_by_class = self._get_decoder_eval_targets()
            if not targets_by_class:
                raise ValueError("annotate_data() requires non-empty decoder evaluation targets.")

            masked_col = getattr(getattr(self, "config", None), "masked_col", None)
            if masked_col and masked_col in training_like_df.columns:
                annotated_df, target_token_map, token_columns, class_columns = self._annotate_text_prediction_rows(
                    training_like_df,
                    model=getattr(annotation_model, "base_model", annotation_model),
                    tokenizer=tokenizer,
                    targets_by_class=targets_by_class,
                    eval_batch_size=eval_batch_size,
                    unicode_string_map=unicode_string_map,
                )
            else:
                annotated_df, target_token_map, token_columns, class_columns = self._annotate_text_classification_rows(
                    training_like_df,
                    model=getattr(annotation_model, "base_model", annotation_model),
                    tokenizer=tokenizer,
                    targets_by_class=targets_by_class,
                    eval_batch_size=eval_batch_size,
                    unicode_string_map=unicode_string_map,
                )

            annotated_df = self._add_expected_probability_columns(
                annotated_df,
                target_token_map=target_token_map,
                token_columns=token_columns,
                class_columns=class_columns,
            )

            summary = self._build_annotation_summary(
                annotated_df,
                target_token_map=target_token_map,
                targets_by_class=targets_by_class,
                token_columns=token_columns,
                class_columns=class_columns,
                unicode_string_map=unicode_string_map,
            )

            if csv_path:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                annotated_df.to_csv(csv_path, index=False)
            if json_path:
                _write_json_atomic(json_path, summary)

            return {
                "annotated_df": annotated_df,
                "summary": summary,
            }
        finally:
            if hasattr(self, "_target_classes"):
                self._target_classes = original_target_classes
            if hasattr(self, "config") and hasattr(self.config, "target_classes"):
                self.config.target_classes = original_config_target_classes
