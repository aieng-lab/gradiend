"""
Prediction objectives for text-prediction GRADIEND.

An objective owns the pair of semantics that must stay aligned:

- the training model/objective used to produce gradients;
- the scoring model/objective used by decoder probability-shift analysis.

For decoder-only auxiliary MLM heads, those are intentionally different: the
auxiliary head is a training surrogate, while scoring must use the original CLM
head to measure real-world behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from gradiend.model.utils import is_decoder_only_model, is_seq2seq_model
from gradiend.trainer.text.prediction.decoder_eval_utils import (
    compute_lms,
    evaluate_probability_shift_score,
)
from gradiend.util.paths import has_saved_decoder_mlm_head, resolve_decoder_mlm_head_dir


SUPPORTED_PREDICTION_OBJECTIVES = {
    "auto",
    "mlm_mask_token",
    "clm_next_token",
    "clm_mlm_head",
    "clm_sequence_cloze",
    "seq2seq_decoder",
    "seq2seq_decoder_sequence_cloze",
    "seq2seq_encoder_mlm",
}

SEQ2SEQ_PREDICTION_OBJECTIVES = frozenset(
    {
        "seq2seq_decoder",
        "seq2seq_decoder_sequence_cloze",
        "seq2seq_encoder_mlm",
    }
)
from gradiend.trainer.text.prediction.objective_hints import format_seq2seq_objective_hint
DECODER_ONLY_PREDICTION_OBJECTIVES = frozenset(
    {
        "clm_next_token",
        "clm_mlm_head",
        "clm_sequence_cloze",
    }
)
ENCODER_MLM_PREDICTION_OBJECTIVES = frozenset({"mlm_mask_token"})


def _model_label(model_or_tokenizer: Any) -> str:
    name = getattr(model_or_tokenizer, "name_or_path", None)
    if name:
        return str(name)
    return getattr(model_or_tokenizer.__class__, "__name__", type(model_or_tokenizer).__name__)


def validate_prediction_objective_for_model(objective: str, model_or_tokenizer: Any) -> None:
    """
    Raise when an explicit ``prediction_objective`` is incompatible with the base model type.

    ``auto`` is not validated here (resolved per model at runtime).
    """
    if objective == "auto" or model_or_tokenizer is None:
        return

    label = _model_label(model_or_tokenizer)
    is_seq2seq = is_seq2seq_model(model_or_tokenizer)
    is_decoder_only = is_decoder_only_model(model_or_tokenizer)

    if objective in SEQ2SEQ_PREDICTION_OBJECTIVES:
        if not is_seq2seq:
            hint = (
                "Use prediction_objective='clm_sequence_cloze' for decoder-only sequence cloze, "
                "or 'mlm_mask_token' for BERT-style encoder MLM."
            )
            if objective == "seq2seq_encoder_mlm":
                hint = "Use prediction_objective='mlm_mask_token' for BERT-style encoder MLM."
            raise ValueError(
                f"prediction_objective={objective!r} requires an encoder-decoder model "
                f"(e.g. T5, BART, mT5), but {label!r} is not seq2seq. {hint}"
            )
        return

    if objective in DECODER_ONLY_PREDICTION_OBJECTIVES:
        if is_seq2seq:
            raise ValueError(
                f"prediction_objective={objective!r} requires a decoder-only (causal LM) model, "
                f"but {label!r} is encoder-decoder. "
                f"{format_seq2seq_objective_hint()}"
            )
        if not is_decoder_only:
            alt = "clm_sequence_cloze" if objective == "clm_sequence_cloze" else "clm_next_token"
            raise ValueError(
                f"prediction_objective={objective!r} requires a decoder-only (causal LM) model, "
                f"but {label!r} is an encoder MLM model. "
                f"Use prediction_objective='mlm_mask_token' for encoder MLM, or {alt!r} only on GPT-style models."
            )
        return

    if objective in ENCODER_MLM_PREDICTION_OBJECTIVES:
        if is_seq2seq:
            raise ValueError(
                f"prediction_objective={objective!r} requires an encoder-only MLM model (e.g. BERT), "
                f"but {label!r} is encoder-decoder. "
                f"{format_seq2seq_objective_hint()}"
            )
        if is_decoder_only:
            raise ValueError(
                f"prediction_objective={objective!r} requires an encoder MLM model (e.g. BERT), "
                f"but {label!r} is decoder-only. "
                "Use prediction_objective='clm_next_token', 'clm_mlm_head', or 'clm_sequence_cloze' instead."
            )


@dataclass(frozen=True)
class PredictionObjective:
    """Base strategy for prediction training/scoring semantics."""

    name: str

    def should_use_custom_prediction_head(self, trainer: Any) -> bool:
        return False

    def ensure_training_resources(self, trainer: Any, model: Any) -> None:
        """Prepare mandatory resources for this objective before model loading."""

    def resolve_scoring_model(self, model: Any, tokenizer: Any, trainer: Any) -> Any:
        if hasattr(model, "gradiend") and hasattr(model, "base_model"):
            base = getattr(model, "base_model", None)
            if base is not None:
                model = base
        return model

    def score_probability_shift(
        self,
        model: Any,
        tokenizer: Any,
        targets: Dict[str, list],
        eval_data_df: pd.DataFrame,
        *,
        key_text: str,
        dataset_class_col: Optional[str],
        factual_dataset_class_col: Optional[str] = None,
        use_row_wise: bool,
        return_per_row_df: bool,
        trainer: Any,
    ) -> Any:
        scoring_model = self.resolve_scoring_model(model, tokenizer, trainer)
        return evaluate_probability_shift_score(
            scoring_model,
            tokenizer,
            targets=targets,
            eval_data_df=eval_data_df,
            key_text=key_text,
            dataset_class_col=dataset_class_col,
            factual_dataset_class_col=factual_dataset_class_col,
            use_row_wise=use_row_wise,
            return_per_row_df=return_per_row_df,
            objective=self.name,
            rhs_window=getattr(getattr(trainer, "_training_args", None), "decoder_sequence_cloze_rhs_window", -1),
        )

    def compute_lms(
        self,
        model: Any,
        tokenizer: Any,
        texts: list,
        *,
        trainer: Any,
        ignore: Optional[list] = None,
        max_texts: Optional[int] = None,
        batch_size: int = 32,
    ) -> Any:
        scoring_model = self.resolve_scoring_model(model, tokenizer, trainer)
        return compute_lms(scoring_model, tokenizer, texts, ignore=ignore, max_texts=max_texts, batch_size=batch_size)


@dataclass(frozen=True)
class DecoderMLMHeadObjective(PredictionObjective):
    """Train with auxiliary MLM head, score with the original decoder CLM head."""

    name: str = "clm_mlm_head"

    def should_use_custom_prediction_head(self, trainer: Any) -> bool:
        head_dir = resolve_decoder_mlm_head_dir(trainer.experiment_dir)
        return bool(head_dir and os.path.isdir(head_dir))

    def ensure_training_resources(self, trainer: Any, model: Any) -> None:
        args = getattr(trainer, "_training_args", None)
        if args is None:
            return
        head_dir = resolve_decoder_mlm_head_dir(trainer.experiment_dir)
        if not head_dir:
            raise ValueError(
                "prediction_objective='clm_mlm_head' requires TrainingArguments.experiment_dir "
                "or an explicit decoder MLM-head output path."
            )
        if has_saved_decoder_mlm_head(head_dir):
            return
        trainer.train_decoder_only_mlm_head(
            model,
            output=head_dir,
            batch_size=getattr(args, "decoder_mlm_head_batch_size", 4),
            epochs=getattr(args, "decoder_mlm_head_epochs", 5),
            lr=getattr(args, "decoder_mlm_head_lr", 1e-4),
            max_size=getattr(args, "decoder_mlm_head_max_size", None),
            use_cache=getattr(args, "use_cache", False),
            model_use_cache=getattr(args, "model_use_cache", False),
        )

    def resolve_scoring_model(self, model: Any, tokenizer: Any, trainer: Any) -> Any:
        model = super().resolve_scoring_model(model, tokenizer, trainer)
        to_original = getattr(model, "to_original_model", None)
        if callable(to_original):
            return to_original()
        return model


def _explicit_objective_name(trainer: Any) -> str:
    args = getattr(trainer, "_training_args", None)
    name = getattr(args, "prediction_objective", "auto") if args is not None else "auto"
    name = str(name or "auto").strip()
    if name not in SUPPORTED_PREDICTION_OBJECTIVES:
        raise ValueError(
            f"Unsupported prediction_objective={name!r}. "
            f"Supported values: {sorted(SUPPORTED_PREDICTION_OBJECTIVES)}"
        )
    return name


def resolve_prediction_objective(trainer: Any, model_or_tokenizer: Any = None) -> PredictionObjective:
    """
    Resolve the objective while preserving existing behavior.

    auto:

    - existing decoder MLM-head cache: train through that head, score original CLM;
    - decoder-only without head: current next-token CLM behavior;
    - otherwise: true masked-LM behavior.
    """
    explicit = _explicit_objective_name(trainer)
    if explicit == "clm_mlm_head":
        objective = DecoderMLMHeadObjective()
    elif explicit != "auto":
        objective = PredictionObjective(explicit)
    else:
        head_dir = resolve_decoder_mlm_head_dir(trainer.experiment_dir)
        if head_dir and has_saved_decoder_mlm_head(head_dir):
            objective = DecoderMLMHeadObjective()
        elif model_or_tokenizer is not None and is_seq2seq_model(model_or_tokenizer):
            objective = PredictionObjective("seq2seq_encoder_mlm")
        elif model_or_tokenizer is not None and is_decoder_only_model(model_or_tokenizer):
            objective = PredictionObjective("clm_next_token")
        else:
            objective = PredictionObjective("mlm_mask_token")

    if explicit != "auto" and model_or_tokenizer is not None:
        validate_prediction_objective_for_model(explicit, model_or_tokenizer)
    return objective


def should_use_decoder_mlm_head_for_auto(trainer: Any) -> bool:
    """Whether resolve_model_path should substitute the cached decoder MLM head."""
    return resolve_prediction_objective(trainer).should_use_custom_prediction_head(trainer)

