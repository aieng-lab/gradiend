"""
TextClassificationModelWithGradiend: Sequence classification implementation of ModelWithGradiend.

Uses AutoModelForSequenceClassification as base. create_gradients / forward take
classification inputs (input_ids, attention_mask, labels); single segment or pair (SEP).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gradiend.util.util import set_requires_grad_true
from gradiend.util.logging import get_logger
from gradiend.trainer.text.common.dual_head_model import try_build_dual_head_from_base_path
from gradiend.trainer.text.common.model_base import TextModelWithGradiend
from gradiend.trainer.text.classification.data import tokenize_for_classification

logger = get_logger(__name__)

CONFIG_CLASSIFICATION_HEAD = "config_classification_head.json"


def _base_path_from_config(model: Any, load_path: str) -> Optional[str]:
    """
    Try to get the original base model path from the model config (e.g. when loading from a classification_head dir).
    Returns None if not found or if the only path is the load_path (saved dir).
    """
    config = getattr(model, "config", None)
    if config is None:
        return None
    name = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if not name or name == load_path:
        return None
    # Prefer paths that look like HF ids (short, or no long absolute path)
    if "/" in name and len(name) > 80:
        return None
    return name


def _effective_max_length(tokenizer: Any, base_model: Any, default: int = 512) -> int:
    tokenizer_max = getattr(tokenizer, "model_max_length", default)
    if tokenizer_max is None or (isinstance(tokenizer_max, int) and tokenizer_max > 10**9):
        tokenizer_max = default
    config = getattr(base_model, "config", None)
    model_max = None
    if config is not None:
        model_max = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
    if model_max is not None and model_max > 0:
        return min(tokenizer_max, model_max)
    return tokenizer_max if tokenizer_max != default else default


class TextClassificationModelWithGradiend(TextModelWithGradiend):
    """
    Text model for sequence classification: base = AutoModelForSequenceClassification.

    create_inputs(text_or_pair, label_id) -> {input_ids, attention_mask, labels}.
    forward(inputs) / create_gradients(inputs or text, label) -> gradients for GRADIEND.
    """

    def __init__(
        self,
        base_model: Any,
        gradiend: Any,
        tokenizer: Any,
        gradient_creator: Any = None,
        source: str = "factual",
        target: str = "diff",
        **kwargs: Any,
    ):
        # TextModelWithGradiend expects base_model + gradiend + tokenizer; it uses AutoModelForLM in _load_model
        # We pass base_model (AutoModelForSequenceClassification) and tokenizer
        super().__init__(
            base_model=base_model,
            gradiend=gradiend,
            tokenizer=tokenizer,
            gradient_creator=gradient_creator or self,
            source=source,
            target=target,
            **kwargs,
        )

    @classmethod
    def _load_model(
        cls,
        load_directory: str,
        base_model_id: Optional[str] = None,
        gradiend_kwargs: Optional[Dict[str, Any]] = None,
        base_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        torch_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
        base_model_device: Optional[torch.device] = None,
        num_labels: Optional[int] = None,
        id2label: Optional[Dict[int, str]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Load AutoModelForSequenceClassification and tokenizer.

        When load_directory is a path with config_classification_head.json, load from that dir.
        Otherwise load from base model id/path (e.g. bert-base-cased). Optional num_labels/id2label
        for head resize when loading from base id.
        """
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels
        if id2label is not None:
            load_kwargs["id2label"] = id2label

        path = str(load_directory).strip()
        if base_model_id is not None:
            path = base_model_id
        if base_model is not None:
            if tokenizer is not None:
                set_requires_grad_true(base_model)
                if base_model_device is not None:
                    base_model = base_model.to(base_model_device)
                return base_model, tokenizer
            tokenizer_id = None
            if gradiend_kwargs:
                tokenizer_id = gradiend_kwargs.get("tokenizer")
            tokenizer_id = tokenizer_id or getattr(base_model, "name_or_path", None)
            if not tokenizer_id:
                raise ValueError("Shared base_model must provide name_or_path or gradiend_kwargs['tokenizer']")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
            set_requires_grad_true(base_model)
            if base_model_device is not None:
                base_model = base_model.to(base_model_device)
            return base_model, tokenizer
        if path and os.path.isdir(path):
            marker = os.path.join(path, CONFIG_CLASSIFICATION_HEAD)
            if os.path.isfile(marker):
                logger.info("Loading sequence classification model from %s", path)
                base_model = AutoModelForSequenceClassification.from_pretrained(path, **load_kwargs)
                if tokenizer is not None:
                    set_requires_grad_true(base_model)
                    if base_model_device is not None:
                        base_model = base_model.to(base_model_device)
                    return base_model, tokenizer
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
                set_requires_grad_true(base_model)
                if base_model_device is not None:
                    base_model = base_model.to(base_model_device)
                # Try to attach LM head from base if config stores base path (for decoder LMS).
                base_path = _base_path_from_config(base_model, path)
                if base_path:
                    base_model = try_build_dual_head_from_base_path(
                        base_model, base_path,
                        trust_remote_code=trust_remote_code,
                        device=base_model_device,
                    )
                return base_model, tokenizer

        base_model = AutoModelForSequenceClassification.from_pretrained(path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        set_requires_grad_true(base_model)
        if base_model_device is not None:
            base_model = base_model.to(base_model_device)
        # Attach LM head from same base path so decoder LMS can use forward_lm (shared backbone).
        base_model = try_build_dual_head_from_base_path(
            base_model, path,
            trust_remote_code=trust_remote_code,
            device=base_model_device,
        )
        return base_model, tokenizer

    def create_inputs(
        self,
        text_or_pair: Union[str, Tuple[str, str]],
        label_id: int,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build {input_ids, attention_mask, labels} for one (text or pair, label)."""
        with self.exclusive_base_gradient_access():
            if max_length is None:
                max_length = _effective_max_length(self.tokenizer, self.base_model)
            device = getattr(self.base_model, "device", None)
            out = tokenize_for_classification(
                self.tokenizer,
                text_or_pair,
                label_id,
                max_length=max_length,
                device=None,
            )
            if device is not None:
                out = {k: v.to(device) for k, v in out.items()}
            return out

    def create_gradients(
        self,
        text_or_pair: Union[str, Tuple[str, str]],
        label: Union[int, str],
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create gradients for one (text or pair, label). Label can be int or str (mapped via config)."""
        if isinstance(label, str) and hasattr(self, "label2id") and self.label2id is not None:
            label_id = self.label2id.get(label, self.label2id.get(str(label), 0))
        else:
            label_id = int(label)
        inputs = self.create_inputs(text_or_pair, label_id, **kwargs)
        return self.forward(inputs, return_dict=return_dict, **kwargs)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_dict: bool = False,
        target_device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run base model forward with labels, backward, return gradients. Used as gradient_creator by GradientTrainingDataset.
        """
        with self.exclusive_base_gradient_access():
            inputs = self._place_inputs_for_base_forward(inputs)
            # Ensure batch dim
            inputs = {
                k: v.unsqueeze(0) if v.ndim == 1 else (v.squeeze(0) if v.ndim == 3 and v.shape[1] == 1 else v)
                for k, v in inputs.items()
            }
            base_for_grad = self._get_base_forward_model()
            outputs = base_for_grad(**inputs)
            loss = outputs.loss
            target_dev = target_device or kwargs.pop("target_device", self.gradiend.device_encoder)

            self._zero_base_grad(set_to_none=True)
            if getattr(self, "base_model_is_sharded", False):
                return self.gradiend.extract_gradients_streaming(
                    base_for_grad,
                    lambda: loss.backward(),
                    return_dict=return_dict,
                    target_device=torch.device("cpu") if target_dev is not None else None,
                )

            loss.backward()
            grads = self.gradiend.extract_gradients(
                base_for_grad,
                return_dict=return_dict,
                target_device=target_dev,
            )
            self._zero_base_grad(set_to_none=True)
            return grads
