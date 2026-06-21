"""
TextClassificationTrainer: GRADIEND trainer for sequence classification.

Uses AutoModelForSequenceClassification, unified (factual, alternative) data,
and same decoder-eval contract (probs_by_dataset, probs_factual).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch

from gradiend.visualizer.plot_delegation import see_implementation
from gradiend.trainer.text.common.loading import AutoModelForLM
from gradiend.trainer.text.common.lm_eval import compute_lms
from gradiend.trainer.trainer import Trainer
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.model import ModelWithGradiend
from gradiend.trainer.core.dataset import GradientTrainingDataset
from gradiend.util import normalize_split_name
from gradiend.util.encoder_splits import EncoderSplit, resolve_encoder_splits
from gradiend.trainer.text.classification.config import TextClassificationConfig
from gradiend.trainer.text.classification.data import (
    build_classification_training_df,
    build_classification_head_df_from_pairs,
    ClassificationTrainingDataset,
    TEXT_FACTUAL,
    TEXT_ALTERNATIVE,
    LABEL_FACTUAL,
    LABEL_ALTERNATIVE,
    FACTUAL_ID,
    ALTERNATIVE_ID,
    FACTUAL_CLS,
    ALTERNATIVE_CLS,
)
from gradiend.trainer.text.classification.model_with_gradiend import TextClassificationModelWithGradiend
from gradiend.trainer.text.classification.classification_head import train_classification_head
from gradiend.trainer.text.classification.evaluator import ClassificationEvaluator
from gradiend.evaluator import Evaluator
from gradiend.util.paths import resolve_classification_head_dir, resolve_encoder_analysis_path
from gradiend.util.logging import get_logger
from gradiend.util.encoding_rows import encode_dataset_to_rows
from gradiend.data.core import resolve_base_data

logger = get_logger(__name__)


def _resolve_dataframe(
    data: Optional[Union[pd.DataFrame, str, Path]],
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (str, Path)):
        path = str(data)
        suffix = Path(path).suffix.lower()
        # Treat known local table files first (csv / parquet / tsv, or existing path).
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif suffix in {".csv", ".tsv"} or Path(path).exists():
            df = pd.read_csv(path)
        else:
            # Fall back to modality-agnostic base loader (HF dataset id, DataFrame-like CSV, or list[str]).
            # This matches TextPredictionDataCreator so TextClassification supports the same neutral sources.
            texts = resolve_base_data(path)
            df = pd.DataFrame({"text": texts})
    else:
        return None
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


class TextClassificationTrainer(Trainer):
    """
    Trainer for sequence classification GRADIEND using AutoModelForSequenceClassification.

    Implements FeatureLearningDefinition: create_training_data, create_gradient_training_dataset,
    _get_decoder_eval_dataframe, _get_decoder_eval_targets, evaluate_base_model, _analyze_encoder.
    """

    def __init__(
        self,
        model: Union[str, Any],
        args: Optional[TrainingArguments] = None,
        config: Optional[TextClassificationConfig] = None,
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        target_classes: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        n_features: int = 1,
        **kwargs: Any,
    ):
        self.config = config or TextClassificationConfig()
        if data is not None:
            self.config.data = data
        if target_classes is not None:
            self.config.target_classes = target_classes
        if run_id is not None:
            self.config.run_id = run_id
        self.config.n_features = n_features or self.config.n_features
        self._combined_data: Optional[pd.DataFrame] = None
        self._label2id: Optional[Dict[Union[str, int], int]] = None
        self._id2label: Optional[Dict[int, str]] = None
        super().__init__(
            model=model,
            target_classes=target_classes or self.config.target_classes,
            args=args,
            run_id=run_id or self.config.run_id,
            n_features=self.config.n_features,
            **kwargs,
        )

    @property
    def default_model_with_gradiend_cls(self) -> Type[ModelWithGradiend]:
        return TextClassificationModelWithGradiend

    def resolve_custom_prediction_head_dir(self) -> Optional[str]:
        """Return classification_head dir when it exists so resolve_model_path uses it."""
        return resolve_classification_head_dir(self.experiment_dir)

    def get_target_feature_class_ids(self) -> Optional[List[Any]]:
        if self.target_classes is not None:
            return list(self.target_classes)
        return None

    @staticmethod
    def _validate_min_factual_classes(df: pd.DataFrame, *, context: str) -> None:
        """Require at least two distinct factual classes in the given data slice."""
        class_col = FACTUAL_CLS if FACTUAL_CLS in df.columns else FACTUAL_ID
        unique_factual = df[class_col].astype(str).dropna().unique().tolist()
        if len(unique_factual) < 2:
            found = ", ".join(repr(v) for v in unique_factual) if unique_factual else "none"
            raise ValueError(
                "TextClassificationTrainer requires at least two distinct factual classes "
                f"in {context} ({class_col}). Found: {found}. "
                "Adjust your data so both target classes are present."
            )

    def _ensure_data(self) -> None:
        if self._combined_data is not None:
            return
        assert self.training_args is not None, "TrainingArguments required (pass args=... to the trainer)."
        df = _resolve_dataframe(self.config.data)
        if df is None or len(df) == 0:
            raise ValueError("No data provided. Set config.data or pass data= to the trainer.")
        self._combined_data, self._label2id, self._id2label = build_classification_training_df(
            self.config,
            df,
            seed=self.training_args.seed,
        )
        if self.config.target_classes is None:
            self._target_classes = list(self._id2label.values()) if self._id2label else []
        else:
            self._target_classes = list(self.config.target_classes)
        self._validate_min_factual_classes(self._combined_data, context="the dataset overall")
        split_col = getattr(self.config, "split_col", "split")
        if split_col in self._combined_data.columns:
            for split_name in ("train", "validation", "test"):
                split_df = self._combined_data[
                    self._combined_data[split_col].astype(str).str.lower() == split_name
                ]
                if len(split_df) == 0:
                    continue
                self._validate_min_factual_classes(split_df, context=f"the {split_name} split")

    @property
    def combined_data(self) -> Optional[pd.DataFrame]:
        self._ensure_data()
        return self._combined_data

    @property
    def evaluator(self) -> Any:
        """Lazy-init evaluator; always use ClassificationEvaluator so encoder eval can branch on actual eval data."""
        if self._evaluator is None:
            if self._evaluator_class is None:
                self._evaluator_class = ClassificationEvaluator
            cls = self._evaluator_class if self._evaluator_class is not None else Evaluator
            self._evaluator = cls(self)
        return self._evaluator

    def create_training_data(
        self,
        model_or_tokenizer: Any,
        split: EncoderSplit = "train",
        batch_size: Optional[int] = None,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> ClassificationTrainingDataset:
        """Create tokenized classification training/evaluation data.

        Args:
            model_or_tokenizer: Model-with-GRADIEND or tokenizer used to encode
                classification inputs.
            split: Split name(s) to load.
            batch_size: Accepted for trainer API consistency; batching is handled
                by the dataset/loader.
            max_size: Optional cap on rows.
            **kwargs: Reserved for future classification data options.
        """
        self._ensure_data()
        tokenizer = getattr(model_or_tokenizer, "tokenizer", model_or_tokenizer)
        split_col = getattr(self.config, "split_col", "split")
        if split_col in self._combined_data.columns:
            resolved = resolve_encoder_splits(
                split,
                available=self._combined_data[split_col].dropna().astype(str).tolist(),
            )
            norm_col = self._combined_data[split_col].astype(str).map(normalize_split_name)
            split_data = self._combined_data[norm_col.isin(resolved)].copy()
        else:
            split_data = self._combined_data
        if len(split_data) == 0:
            raise ValueError(f"No data for split {split!r}")
        if max_size is not None and len(split_data) > max_size:
            split_data = split_data.sample(n=max_size, random_state=42).reset_index(drop=True)
        return ClassificationTrainingDataset(
            split_data,
            tokenizer,
            self._label2id,
            text_factual_col=TEXT_FACTUAL,
            text_alternative_col=TEXT_ALTERNATIVE,
            label_factual_col=LABEL_FACTUAL,
            label_alternative_col=LABEL_ALTERNATIVE,
            max_length=getattr(self.config, "max_length", 512) or 512,
            pair=self.pair,
            feature_classes=self.target_classes,
        )

    def create_gradient_training_dataset(
        self,
        raw_training_data: ClassificationTrainingDataset,
        model_with_gradiend: Any,
        *,
        cache_dir: Optional[str] = None,
        use_cached_gradients: bool = False,
        **kwargs: Any,
    ) -> GradientTrainingDataset:
        """Wrap classification data in a gradient-producing dataset.

        Args:
            raw_training_data: Classification dataset produced by
                ``create_training_data``.
            model_with_gradiend: Model used to create gradients.
            cache_dir: Optional gradient cache directory.
            use_cached_gradients: Whether existing cached gradients may be reused.
            **kwargs: Optional gradient dataset settings such as ``source`` and
                ``target``.
        """
        source = kwargs.pop("source", None) or getattr(self.training_args, "source", "factual")
        target = kwargs.pop("target", None) or getattr(self.training_args, "target", "diff")
        tokenizer = model_with_gradiend.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

        def get_padding_value(subkey: str) -> int:
            """Return the padding value for a tensor field named by ``subkey``."""
            return pad_token_id if "input_ids" in subkey else 0

        return GradientTrainingDataset(
            raw_training_data,
            model_with_gradiend,
            source=source,
            target=target,
            cache_dir=cache_dir,
            use_cached_gradients=use_cached_gradients,
            cache_key_fields=["input_text", "label"] if (cache_dir and use_cached_gradients) else None,
            dtype=getattr(model_with_gradiend.gradiend, "torch_dtype", torch.float32),
            device=getattr(model_with_gradiend.gradiend, "device_encoder", None),
            get_padding_value=get_padding_value,
            timing_steps=kwargs.pop("timing_steps", 0),
            timing_label=kwargs.pop("timing_label", "classification-gradient"),
        )

    def _get_decoder_eval_dataframe(
        self,
        tokenizer: Any,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        cached_training_like_df: Optional[pd.DataFrame] = None,
        cached_neutral_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if cached_training_like_df is not None and cached_neutral_df is not None:
            return cached_training_like_df, cached_neutral_df
        self._ensure_data()
        split_col = getattr(self.config, "split_col", "split")
        if split_col in self._combined_data.columns:
            eval_data = self._combined_data[
                self._combined_data[split_col].astype(str).str.lower().isin(("test", "validation"))
            ].copy()
        else:
            eval_data = self._combined_data.copy()
        if len(eval_data) == 0:
            eval_data = self._combined_data.copy()
        if max_size_training_like and len(eval_data) > max_size_training_like:
            eval_data = eval_data.sample(n=max_size_training_like, random_state=42).reset_index(drop=True)
        training_like_df = eval_data.rename(columns={
            TEXT_FACTUAL: "text",
            FACTUAL_ID: "label_class",
        }).copy()
        if LABEL_FACTUAL in training_like_df.columns:
            training_like_df["label_class"] = training_like_df[LABEL_FACTUAL].map(
                lambda v: (self._id2label or {}).get(int(v), str(v)) if pd.notna(v) else v
            )
        if "text" not in training_like_df.columns:
            training_like_df["text"] = eval_data[TEXT_FACTUAL].values
        neutral_df = _resolve_dataframe(
            getattr(self.config, "eval_neutral_data", None),
            max_rows=max_size_neutral or getattr(self.config, "eval_neutral_max_rows"),
        )
        if neutral_df is None or len(neutral_df) == 0:
            neutral_df = training_like_df.copy()
        elif max_size_neutral and len(neutral_df) > max_size_neutral:
            neutral_df = neutral_df.sample(n=max_size_neutral, random_state=42).reset_index(drop=True)
        if "text" not in neutral_df.columns and "text_factual" in neutral_df.columns:
            neutral_df = neutral_df.copy()
            neutral_df["text"] = neutral_df["text_factual"]
        return training_like_df.reset_index(drop=True), neutral_df.reset_index(drop=True)

    def _get_decoder_eval_targets(self) -> Dict[str, List[str]]:
        """Return class name -> [class name] for decoder eval (classification uses class probs)."""
        self._ensure_data()
        classes = self.target_classes or list(self._id2label.values()) if self._id2label else []
        return {c: [c] for c in classes}

    def _get_base_model_name_or_path(self, model: Any) -> str:
        """
        Get the *base* model name_or_path for LM loading.

        For classification we always want the original base checkpoint (e.g. 'distilbert-base-uncased'),
        not the classification_head directory. Trainer.base_model_path stores exactly that initial
        argument, so prefer it when available and fall back to model/config only as a last resort.
        """
        # Prefer the trainer's original base_model_path when present (e.g. 'distilbert-base-uncased').
        try:
            base_path = getattr(self, "base_model_path", None)
        except Exception:
            base_path = None
        if isinstance(base_path, str) and base_path:
            return base_path

        # Fallback: introspect the model / config (useful in edge cases or tests).
        name = getattr(model, "name_or_path", None)
        if name:
            return str(name)
        config = getattr(model, "config", None)
        if config is not None:
            name = getattr(config, "name_or_path", None) or getattr(config, "_name_or_path", None)
        if not name:
            raise ValueError(
                "Cannot determine base LM path. Tried trainer.base_model_path and model/config.name_or_path. "
                "Required for decoder_lms_mode='lm'."
            )
        return str(name)

    def _compute_lms_via_base_lm(
        self,
        *,
        model: Any,
        tokenizer: Any,
        texts: List[str],
        max_texts: Optional[int],
        eval_batch_size: Optional[int],
    ) -> Optional[float]:
        """
        Compute LMS using the *current* model's encoder (with optional LM head).

        When the model has forward_lm (dual-head wrapper), we call it so LMS reflects
        the same GRADIEND-modified encoder as classification. No separate LM load or
        state_dict sync. Otherwise we fall back to loading a separate LM and syncing
        the backbone (less efficient and can be wrong across deepcopy).
        """
        limited = texts[:(max_texts or len(texts))]

        if hasattr(model, "forward_lm") and callable(getattr(model, "forward_lm", None)):
            # Dual-head wrapper: model has one encoder + cls head + lm head; use it directly.
            class _LMAdapter:
                def __init__(self, m):
                    self._m = m

                def __call__(self, **kwargs):
                    return self._m.forward_lm(**kwargs)

                @property
                def device(self):
                    return next(self._m.parameters()).device

                def eval(self):
                    self._m.eval()
                    return self

            lms_result = compute_lms(
                _LMAdapter(model),
                tokenizer,
                limited,
                ignore=[],
                max_texts=len(limited),
                batch_size=eval_batch_size or 32,
            )
            if isinstance(lms_result, (int, float)):
                return float(lms_result)
            return float(lms_result.get("lms", lms_result.get("perplexity", 0.0)))

        # Fallback: load and cache a separate LM, sync backbone from current model each time.
        name_or_path = self._get_base_model_name_or_path(model)
        device = next(model.parameters()).device

        cached = getattr(self, "_decoder_lm_model", None)
        cached_name = getattr(self, "_decoder_lm_model_name", None)
        if cached is None or cached_name != name_or_path:
            try:
                lm_model = AutoModelForLM.from_pretrained(name_or_path, trust_remote_code=True)
            except Exception as e:
                raise ValueError(
                    f"Failed to load LM from base model name_or_path={name_or_path!r}. "
                    "Sequence classification base has no LM head; LMS must be computed with a separately "
                    "loaded LM (AutoModelForLM) or use a dual-head model (forward_lm). Error: " + str(e)
                ) from e
            lm_model = lm_model.to(device)
            lm_model.eval()
            self._decoder_lm_model = lm_model
            self._decoder_lm_model_name = name_or_path
        else:
            lm_model = cached

        base_attr = getattr(model, "base_model_prefix", None)
        if base_attr and hasattr(model, base_attr) and hasattr(lm_model, base_attr):
            getattr(lm_model, base_attr).load_state_dict(getattr(model, base_attr).state_dict())

        lms_result = compute_lms(
            lm_model,
            tokenizer,
            limited,
            ignore=[],
            max_texts=len(limited),
            batch_size=eval_batch_size or 32,
        )
        if isinstance(lms_result, (int, float)):
            return float(lms_result)
        return float(lms_result.get("lms", lms_result.get("perplexity", 0.0)))

    def _compute_decoder_lms(
        self,
        *,
        model: Any,
        tokenizer: Any,
        training_like_df: pd.DataFrame,
        neutral_df: Optional[pd.DataFrame],
        text_col: str,
        dataset_class_col: str,
        id2label: Dict[int, str],
        max_size_neutral: Optional[int],
        eval_batch_size: Optional[int],
        _run_logits: Callable[[pd.DataFrame], Tuple[np.ndarray, Any]],
        decoder_lms_mode_override: Optional[str] = None,
    ) -> float:
        """
        Compute decoder 'LMS' for sequence classification. Never guess: raise if no score can be computed.
        Modes: classification_accuracy (on non-target classes, >=4 classes), lm (MLM/CLM via base name_or_path), both (average).
        """
        mode = decoder_lms_mode_override if decoder_lms_mode_override is not None else getattr(self.config, "decoder_lms_mode", None)
        target_classes = set(self.target_classes or list(id2label.values()) if id2label else [])
        label2id = {v: k for k, v in (id2label or {}).items()}

        lms_lm: Optional[float] = None
        lms_cls: Optional[float] = None

        # LM path: load AutoModelForLM from base model name_or_path (classification model has no LM head).
        # Do not use the classification model for LM; raise if the base has no MLM/CLM params.
        if mode in (None, "lm", "both"):
            texts: List[str] = []
            if neutral_df is not None and len(neutral_df) > 0 and "text" in neutral_df.columns:
                texts = neutral_df["text"].dropna().astype(str).tolist()[: (max_size_neutral or 500)]
            if not texts and text_col in training_like_df.columns:
                texts = training_like_df[text_col].dropna().astype(str).tolist()[: (max_size_neutral or 500)]
            if texts:
                lms_lm = self._compute_lms_via_base_lm(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    max_texts=max_size_neutral or 500,
                    eval_batch_size=eval_batch_size,
                )

        # Classification-accuracy path: accuracy on non-target classes only (>=4 classes total).
        if mode in (None, "classification_accuracy", "both"):
            if dataset_class_col in training_like_df.columns and label2id:
                all_classes = training_like_df[dataset_class_col].dropna().astype(str).unique().tolist()
                non_target = [c for c in all_classes if c not in target_classes]
                if len(all_classes) >= 4 and len(non_target) >= 1:
                    subset = training_like_df[
                        training_like_df[dataset_class_col].astype(str).isin(non_target)
                    ].copy()
                    if len(subset) > 0:
                        logits, _ = _run_logits(subset)
                        pred = np.argmax(logits, axis=-1)
                        true = np.array([label2id.get(str(c), -1) for c in subset[dataset_class_col]])
                        valid = true >= 0
                        if np.any(valid):
                            lms_cls = float(np.mean(pred[valid] == true[valid]))

        # Resolve final score
        if mode == "lm":
            lms = lms_lm
        elif mode == "classification_accuracy":
            lms = lms_cls
        elif mode == "both":
            if lms_lm is not None and lms_cls is not None:
                lms = (float(lms_lm) + float(lms_cls)) / 2.0
            else:
                lms = lms_lm if lms_lm is not None else lms_cls
        else:
            lms = lms_lm if lms_lm is not None else lms_cls

        if lms is None:
            raise ValueError(
                "Decoder evaluation for sequence classification requires a computable 'LMS' score. "
                "None could be computed. Options: "
                "(1) Set config.decoder_lms_mode='classification_accuracy' and provide evaluation data with "
                ">=4 classes and at least one non-target class (samples not in target_classes). "
                "(2) Set config.decoder_lms_mode='lm' and use a model that supports MLM/CLM with neutral or "
                "training-like text. "
                "(3) Set config.decoder_lms_mode='both' to use both when available (averaged). "
                "Do not guess LMS values; configure one of the above or provide suitable data."
            )
        return float(lms)

    def evaluate_base_model(
        self,
        model: Any,
        tokenizer: Any,
        use_cache: Optional[bool] = None,
        training_like_df: Optional[pd.DataFrame] = None,
        neutral_df: Optional[pd.DataFrame] = None,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run classifier on eval data for decoder-grid scoring.

        Args:
            model: Classification model to evaluate. Rewritten decoder-grid
                candidates are passed here unchanged.
            tokenizer: Tokenizer used for evaluation text.
            use_cache: Accepted for evaluator API consistency.
            training_like_df: Optional training-like evaluation rows.
            neutral_df: Optional neutral rows for LMS computation.
            max_size_training_like: Optional cap for generated training-like rows.
            max_size_neutral: Optional cap for neutral rows.
            eval_batch_size: Optional LMS/classification eval batch size.
            **kwargs: Additional decoder-LMS options, including
                ``decoder_lms_mode``.
        """
        # Use the passed-in model as-is so decoder grid sees different probs per (lr, ff).
        # Do not strip to base_model: for grid we receive the rewritten model from rewrite_base_model.
        if training_like_df is None or neutral_df is None:
            training_like_df, neutral_df = self._get_decoder_eval_dataframe(
                tokenizer,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
            )
        device = next(model.parameters()).device
        id2label = self._id2label or {}
        num_labels = len(id2label) or getattr(model.config, "num_labels", 2)
        text_col = "text" if "text" in training_like_df.columns else TEXT_FACTUAL
        dataset_class_col = "label_class" if "label_class" in training_like_df.columns else FACTUAL_ID

        def _run_logits(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            texts = df[text_col].astype(str).tolist()
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            if hasattr(out, "logits") and out.logits is not None:
                logits = out.logits.cpu().float().numpy()
                return logits, enc.get("attention_mask", None)
            # Encoder-only output (BaseModelOutput): use classification head from disk when available
            head_dir = resolve_classification_head_dir(self.experiment_dir) if self.experiment_dir else None
            if head_dir and os.path.isdir(head_dir):
                try:
                    from transformers import AutoModelForSequenceClassification
                    head_model = AutoModelForSequenceClassification.from_pretrained(head_dir)
                    head_model = head_model.to(device)
                    head_model.eval()
                    with torch.no_grad():
                        head_out = head_model(**enc)
                    logits = head_out.logits.cpu().float().numpy()
                    return logits, enc.get("attention_mask", None)
                except Exception as e:
                    logger.debug("Could not use classification_head for logits: %s", e)
            raise AttributeError(
                "Model output has no 'logits'. Decoder evaluation needs a sequence classification model. "
                "Train a classification head with train_classification_head() so classification_head/ exists, "
                "or ensure the loaded base model is AutoModelForSequenceClassification."
            )

        logits, _ = _run_logits(training_like_df)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        probs_by_dataset: Dict[str, Dict[str, float]] = {}
        if dataset_class_col in training_like_df.columns:
            for cls_name in training_like_df[dataset_class_col].dropna().unique().tolist():
                mask = training_like_df[dataset_class_col] == cls_name
                if not mask.any():
                    continue
                p_cls = probs[mask]
                probs_by_dataset[str(cls_name)] = {}
                for j in range(probs.shape[1]):
                    name = id2label.get(j, str(j))
                    probs_by_dataset[str(cls_name)][name] = float(np.mean(p_cls[:, j]))
        else:
            probs_by_dataset["default"] = {id2label.get(j, str(j)): float(np.mean(probs[:, j])) for j in range(probs.shape[1])}

        probs_factual: Dict[str, float] = {}
        for cls_name, d in probs_by_dataset.items():
            if cls_name in d:
                probs_factual[cls_name] = d[cls_name]

        # Decoder "LMS" for sequence classification: must be computed, never guessed.
        # Supports: classification_accuracy (on non-target classes, >=4 classes), lm (MLM/CLM if model supports it), or both (average).
        decoder_lms_mode_override = kwargs.get("decoder_lms_mode") or getattr(self, "_decoder_lms_mode_override", None)
        lms = self._compute_decoder_lms(
            model=model,
            tokenizer=tokenizer,
            training_like_df=training_like_df,
            neutral_df=neutral_df,
            text_col=text_col,
            dataset_class_col=dataset_class_col,
            id2label=id2label,
            max_size_neutral=max_size_neutral,
            eval_batch_size=eval_batch_size,
            _run_logits=_run_logits,
            decoder_lms_mode_override=decoder_lms_mode_override,
        )

        return {
            "probs_by_dataset": probs_by_dataset,
            "probs_factual": probs_factual,
            "probs": probs_factual,
            "lms": {"lms": lms},
        }

    def _analyze_encoder(
        self,
        model_with_gradiend: Any,
        split: str = "test",
        neutral_data_df: Optional[pd.DataFrame] = None,
        max_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        plot: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        use_cache = use_cache or getattr(self.training_args, "use_cache", False)
        max_size = max_size or getattr(self.training_args, "encoder_eval_max_size", None)
        cache_path = resolve_encoder_analysis_path(
            self.experiment_dir, split=split, max_size=max_size, **kwargs
        ) if self.experiment_dir else None
        if use_cache and cache_path and os.path.isfile(cache_path):
            try:
                return pd.read_csv(cache_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                logger.warning(
                    "Cached encoder analysis at %s is unreadable (%s). Deleting and recomputing.",
                    cache_path,
                    exc,
                )
                try:
                    os.remove(cache_path)
                except OSError:
                    logger.warning("Failed to remove unreadable encoder cache at %s", cache_path, exc_info=True)

        eval_data = self.create_training_data(model_with_gradiend, split=split, batch_size=1, max_size=max_size)
        grad_dataset = self.create_gradient_training_dataset(
            eval_data,
            model_with_gradiend,
            cache_dir=None,
            use_cached_gradients=False,
        )
        rows = encode_dataset_to_rows(model_with_gradiend, grad_dataset)
        for r in rows:
            r["type"] = "training"
        df = pd.DataFrame(rows)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}"
            try:
                df.to_csv(tmp_cache_path, index=False)
                os.replace(tmp_cache_path, cache_path)
            finally:
                if os.path.exists(tmp_cache_path):
                    try:
                        os.remove(tmp_cache_path)
                    except OSError:
                        logger.warning("Failed to remove temporary encoder cache file %s", tmp_cache_path, exc_info=True)
        return df

    def _classification_head_data(
        self,
    ) -> Tuple[pd.DataFrame, Dict[Union[str, int], int], Dict[int, str], int, Optional[str]]:
        """
        Build (text, label, split) data and label maps for classification-head training and eval.
        When the data has only one label (e.g. case #3: different texts, same label), we use
        factual text + factual label and alternative text + other label so the head sees both
        classes. Head classes come from target_classes if user set 2+, else [that_label, "not-{label}"].
        """
        self._ensure_data()
        split_col = getattr(self.config, "split_col", "split")
        target_classes = self._target_classes or (list(self._id2label.values()) if self._id2label else [])
        has_split = split_col in self._combined_data.columns
        unique_factual = self._combined_data[LABEL_FACTUAL].astype(str).unique().tolist()

        if len(unique_factual) <= 1 and len(unique_factual) > 0:
            # Single label in data (case #3): build (text, label, split) from pairs
            # factual + (alternative, other_label). Use target_classes if user set 2+, else derive.
            head_classes = (
                target_classes
                if len(target_classes) >= 2
                else [unique_factual[0], f"not-{unique_factual[0]}"]
            )
            head_df = build_classification_head_df_from_pairs(
                self._combined_data,
                head_classes,
                id2label=self._id2label,
                split_col=split_col,
                text_col="text",
                label_col="label",
            )
            id2label = {i: str(c) for i, c in enumerate(head_classes)}
            label2id = {str(c): i for i, c in enumerate(head_classes)}
            num_labels = len(head_classes)
            return head_df, label2id, id2label, num_labels, split_col if has_split else None
        # Default: use factual text + factual label per row
        id2label = self._id2label
        label2id = self._label2id
        num_labels = len(self._id2label) if self._id2label else 1
        if has_split:
            head_df = self._combined_data.copy()
        else:
            head_df = self._combined_data.copy()
        head_df = head_df.copy()
        head_df["text"] = head_df[TEXT_FACTUAL].values
        head_df["label"] = head_df[LABEL_FACTUAL].map(
            lambda v: (self._id2label or {}).get(int(v), str(v)) if pd.notna(v) else v
        ).values
        return head_df, label2id, id2label, num_labels, split_col if has_split else None

    def train_classification_head(
        self,
        train_df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        """Pre-train and save the experimental sequence-classification head.

        Args:
            train_df: Optional explicit head-training DataFrame. If omitted,
                derived from trainer data.
            output_path: Optional output directory. Defaults to
                ``classification_head`` under ``experiment_dir``.
            use_cache: If True and an existing head is present, reuse it.
            **kwargs: Forwarded to ``train_classification_head``.
        """
        out = output_path or resolve_classification_head_dir(self.experiment_dir)
        if out is None:
            raise ValueError("experiment_dir or output_path required for train_classification_head")
        if use_cache is None and self.training_args is not None:
            use_cache = getattr(self.training_args, "use_cache", False)
        if train_df is None:
            head_df, label2id, id2label, num_labels, split_col = self._classification_head_data()
        else:
            head_df = train_df
            label2id = self._label2id
            id2label = self._id2label
            num_labels = len(self._id2label) if self._id2label else None
            split_col = getattr(self.config, "split_col", "split")
            split_col = split_col if split_col in head_df.columns else None
        return train_classification_head(
            self.base_model_path,
            head_df,
            out,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            split_col=split_col,
            use_cache=bool(use_cache),
            **kwargs,
        )

    def plot_probability_shifts(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        target_class: Optional[str] = None,
        increase_target_probabilities: bool = True,
        use_cache: Optional[bool] = None,
        *,
        output: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: Optional[str] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> str:
        """Plot classification decoder probability shifts.

        Args:
            decoder_results: Optional result from ``evaluate_decoder``.
            target_class: Optional single target class to plot.
            increase_target_probabilities: True for strengthen plots, False for
                weaken plots.
            use_cache: Whether cached decoder results may be used.
            output: Optional explicit output file path.
            show: Whether to display the Matplotlib figure.
            figsize: Optional Matplotlib figure size.
            img_format: Optional output image format. Defaults to trainer config.
            highlight_non_convergence: Override non-convergence markers.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        if img_format is None:
            img_format = getattr(self.config, "img_format", "png")
        return self.evaluator.plot_probability_shifts(
            decoder_results=decoder_results,
            target_class=target_class,
            increase_target_probabilities=increase_target_probabilities,
            use_cache=use_cache,
            output=output,
            show=show,
            figsize=figsize,
            img_format=img_format,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_probability_shifts.__doc__ = (
        plot_probability_shifts.__doc__
        + see_implementation("gradiend.visualizer.probability_shifts.plot_probability_shifts")
    )

    def analyze_decoder_for_plotting(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build plotting payload for classification decoder results.

        Classification grid entries already contain ``probs_by_dataset``.

        Args:
            decoder_results: Optional decoder result. If omitted, computes or
                loads decoder evaluation.
            **kwargs: Forwarded to ``evaluate_decoder`` when results are omitted.
        """
        if decoder_results is None:
            kwargs = {"use_cache": True, **kwargs}
            decoder_results = self.evaluate_decoder(**kwargs)
        grid = decoder_results.get("grid", {})
        _reserved = {"grid", "plot_path", "plot_paths"}
        summary = {k: v for k, v in decoder_results.items() if k not in _reserved}
        return {"plotting_data": grid, "summary": summary}
