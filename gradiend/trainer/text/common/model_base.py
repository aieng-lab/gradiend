"""
TextModelWithGradiend: Base text implementation of ModelWithGradiend.

Shared by prediction (MLM/CLM) and classification. Holds tokenizer,
loading/saving, and from_pretrained. Prediction-specific logic (create_gradients,
create_inputs, mask_and_encode) lives in TextPredictionModelWithGradiend.
"""

from typing import Any, Dict, Optional

import torch
from gradiend.util import get_logger
from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.model import ParamMappedGradiendModel
from gradiend.trainer.text.common.loading import AutoModelForLM, AutoTokenizerForLM, InstructTokenizerWrapper
from gradiend.model.utils import is_decoder_only_model

logger = get_logger(__name__)


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
    def is_decoder_only_model(self) -> bool:
        """Whether the base model is decoder-only (causal LM, no [MASK] token)."""
        return is_decoder_only_model(self.tokenizer)

    def _save_model(self, save_directory, **kwargs):
        kwargs['base_model'] = self.base_model.name_or_path
        kwargs['tokenizer'] = self.tokenizer.name_or_path

    @classmethod
    def _load_model(
        cls,
        load_directory,
        base_model_id: str = None,
        gradiend_kwargs: Dict[str, Any] = None,
        torch_dtype: torch.dtype = torch.float32,
        base_model_device=None,
        trust_remote_code: bool = False,
        device_encoder=None,
        device_decoder=None,
        **kwargs,
    ) -> tuple:
        load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": trust_remote_code}
        load_path = base_model_id if base_model_id is not None else (
            load_directory if isinstance(load_directory, str) else getattr(load_directory, "name_or_path")
        )

        if base_model_id is not None:
            base_model = AutoModelForLM.from_pretrained(base_model_id, **load_kwargs)
            if base_model_device is not None:
                base_model = base_model.to(base_model_device)
            tokenizer_id = None
            if gradiend_kwargs:
                tokenizer_id = gradiend_kwargs.get("tokenizer")
            tokenizer_id = tokenizer_id or base_model_id
            tokenizer = AutoTokenizerForLM.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
            return base_model, tokenizer

        if isinstance(load_directory, str):
            base_model = AutoModelForLM.from_pretrained(load_directory, **load_kwargs)
            if base_model_device is not None:
                base_model = base_model.to(base_model_device)
            tokenizer = AutoTokenizerForLM.from_pretrained(load_directory, trust_remote_code=trust_remote_code)
            return base_model, tokenizer

        base_model = load_directory
        if base_model_device is not None:
            base_model = base_model.to(base_model_device)
        tokenizer = AutoTokenizerForLM.from_pretrained(base_model.name_or_path, trust_remote_code=trust_remote_code)
        return base_model, tokenizer
