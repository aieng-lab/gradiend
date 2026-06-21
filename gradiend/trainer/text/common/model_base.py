"""
TextModelWithGradiend: Base text implementation of ModelWithGradiend.

Shared by prediction (MLM/CLM) and classification. Holds tokenizer,
loading/saving, and from_pretrained. Prediction-specific logic (create_gradients,
create_inputs, mask_and_encode) lives in TextPredictionModelWithGradiend.
"""

import re
from typing import Any, Dict, Optional

import torch
from gradiend.util import get_logger
from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.model import ParamMappedGradiendModel
from gradiend.trainer.text.common.loading import AutoModelForLM, AutoTokenizerForLM, InstructTokenizerWrapper
from gradiend.model.core.seq2seq_backbone import (
    DEFAULT_SEQ2SEQ_GRADIENT_MODE,
    SEQ2SEQ_ENCODER_MLM,
    configure_seq2seq_gradiend_backbone,
    resolve_seq2seq_mode_from_kwargs,
)
from gradiend.model.utils import is_decoder_only_model, is_seq2seq_model

logger = get_logger(__name__)


_MODEL_SIZE_B_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*b(?!\w)", re.IGNORECASE)


def _infer_model_size_b_from_name(name_or_path: Any) -> Optional[float]:
    if name_or_path is None:
        return None
    text = str(name_or_path).replace("\\", "/")
    candidates = _MODEL_SIZE_B_RE.findall(text)
    if not candidates:
        return None
    return max(float(value) for value in candidates)


def _auto_base_model_device_map(name_or_path: Any, device_map):
    if device_map is not None:
        return device_map
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 3:
        return None
    size_b = _infer_model_size_b_from_name(name_or_path)
    if size_b is None or size_b <= 4:
        return None
    logger.info(
        "Detected large model identifier %s (~%sB) with %s CUDA devices; using base_model_device_map='auto'. "
        "Set base_model_device_map=False to force single-device loading.",
        name_or_path,
        size_b,
        torch.cuda.device_count(),
    )
    return "auto"


def _auto_base_model_max_memory_for_device_map(device_map) -> Optional[Dict[int, str]]:
    if device_map != "auto":
        return None
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return None

    max_memory: Dict[int, str] = {0: "0GiB"}
    for i in range(1, torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            gib = max(1, int((props.total_memory / 1024**3) * 0.92))
        except Exception:
            gib = 1
        max_memory[i] = f"{gib}GiB"
    return max_memory


class TextModelWithGradiend(ModelWithGradiend):
    """
    Base text implementation of ModelWithGradiend for MLM/CLM and classification.

    Subclasses: TextPredictionModelWithGradiend (create_gradients, create_inputs,
    mask_and_encode) (planned: TextClassificationModelWithGradiend (CLS-based gradients)).
    """

    def __init__(self,
                 base_model,
                 gradiend,
                 tokenizer,
                 gradient_creator=None,
                 source='factual',
                 target='diff',
                 **kwargs
                 ):
        super().__init__(base_model, gradiend, gradient_creator=gradient_creator, source=source, target=target, **kwargs)
        self.tokenizer = tokenizer
        if self._gradient_creator is None:
            self._gradient_creator = self
        self.is_instruction_model = isinstance(self.tokenizer, InstructTokenizerWrapper)
        if self.is_decoder_only_model:
            if self.tokenizer.eos_token is None:
                raise ValueError('Tokenizer must have an eos_token for decoder-only models')
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def is_seq2seq_model(self) -> bool:
        """Whether the base model is encoder-decoder (e.g. T5, BART)."""
        if getattr(self, "base_model", None) is not None and is_seq2seq_model(self.base_model):
            return True
        return is_seq2seq_model(self.tokenizer)

    @property
    def is_decoder_only_model(self) -> bool:
        """Whether the base model is decoder-only (causal LM, no [MASK] token)."""
        if self.is_seq2seq_model:
            return False
        return is_decoder_only_model(self.tokenizer)

    @property
    def seq2seq_gradient_mode(self) -> str:
        """Seq2seq training mode (alias for ``prediction_objective`` seq2seq values)."""
        return getattr(self.base_model, "_gradiend_seq2seq_gradient_mode", DEFAULT_SEQ2SEQ_GRADIENT_MODE)

    @property
    def use_seq2seq_encoder_mlm(self) -> bool:
        return self.is_seq2seq_model and self.seq2seq_gradient_mode == SEQ2SEQ_ENCODER_MLM

    def _save_model(self, save_directory, **kwargs):
        kwargs['base_model'] = self.base_model.name_or_path
        kwargs['tokenizer'] = self.tokenizer.name_or_path

    @classmethod
    def _load_model(
        cls,
        load_directory,
        base_model_id: str = None,
        gradiend_kwargs: Dict[str, Any] = None,
        base_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        torch_dtype: torch.dtype = torch.float32,
        base_model_device=None,
        base_model_device_map=None,
        base_model_max_memory=None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> tuple:
        load_path = base_model_id if base_model_id is not None else (
            load_directory if isinstance(load_directory, str) else getattr(load_directory, "name_or_path")
        )
        base_model_device_map = _auto_base_model_device_map(load_path, base_model_device_map)

        load_kwargs = {"dtype": torch_dtype, "trust_remote_code": trust_remote_code}
        if base_model_device_map not in (None, False):
            load_kwargs["device_map"] = base_model_device_map
            load_kwargs["max_memory"] = base_model_max_memory or _auto_base_model_max_memory_for_device_map(
                base_model_device_map
            )
            if load_kwargs["max_memory"] is None:
                load_kwargs.pop("max_memory")

        if base_model_id is not None:
            base_model = AutoModelForLM.from_pretrained(load_path, **load_kwargs)
            if base_model_device is not None and base_model_device_map is None:
                base_model = base_model.to(base_model_device)
            if tokenizer is not None:
                return base_model, tokenizer
            tokenizer_id = None
            if gradiend_kwargs:
                tokenizer_id = gradiend_kwargs.get("tokenizer")
            tokenizer_id = tokenizer_id or base_model_id
            tokenizer = AutoTokenizerForLM.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
            return base_model, tokenizer

        if isinstance(load_directory, str):
            base_model = AutoModelForLM.from_pretrained(load_directory, **load_kwargs)
        else:
            base_model = load_directory

        if base_model_device is not None and base_model_device_map is None:
            base_model = base_model.to(base_model_device)
        tokenizer = AutoTokenizerForLM.from_pretrained(base_model.name_or_path, trust_remote_code=trust_remote_code)
        return base_model, tokenizer

    @classmethod
    def _create_gradiend(cls, base_model: Any, load_directory: str, **kwargs) -> ParamMappedGradiendModel:
        mode = resolve_seq2seq_mode_from_kwargs(kwargs)
        if mode is not None and is_seq2seq_model(base_model):
            configure_seq2seq_gradiend_backbone(base_model, mode=mode)
            objective = str(kwargs.get("prediction_objective", "auto") or "auto").strip()
            if objective == "auto":
                objective = mode
            base_model._gradiend_prediction_objective = objective  # type: ignore[attr-defined]
            rhs_window = kwargs.get("decoder_sequence_cloze_rhs_window", -1)
            base_model._gradiend_decoder_sequence_cloze_rhs_window = rhs_window  # type: ignore[attr-defined]
        return super()._create_gradiend(base_model, load_directory, **kwargs)
