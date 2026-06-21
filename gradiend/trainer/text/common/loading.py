"""
Model and tokenizer loading utilities for GRADIEND text trainers.

Used by both prediction and classification. DecoderModelWithMLMHead is imported
lazily when loading a path with config_mlm_head.json to avoid circular imports.
"""

import os
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from gradiend.util.util import set_requires_grad_true
from gradiend.trainer.text.prediction.decoder_only_mlm import DecoderModelWithMLMHead

HF_TOKEN = os.getenv("HF_TOKEN")


def _is_unknown_transformers_architecture_error(error: Exception) -> bool:
    """Detect HF errors for checkpoints newer than the installed Transformers."""
    message = str(error)
    return (
        "does not recognize this architecture" in message
        or "not recognize this architecture" in message
        or "Unrecognized model" in message
    )


def _raise_unknown_transformers_architecture_error(name_or_path, error: Exception) -> None:
    version = getattr(transformers, "__version__", "unknown")
    raise ValueError(
        f"Could not load {name_or_path!r} because Transformers {version} does not "
        "recognize the checkpoint architecture. This usually means the model was "
        "released after the installed Transformers version. Upgrade Transformers "
        "in the runtime environment, or choose a model whose model_type is supported "
        "by that version."
    ) from error



class AutoModelForLM(nn.Module):
    """
    Utility class for loading language models (both masked and causal).

    Automatically detects and loads the appropriate model type based on the
    model name or path.
    """

    @classmethod
    def from_pretrained(cls, name_or_path, torch_dtype=torch.float32, trust_remote_code=False, **kwargs):
        """
        Load a language model from a pretrained checkpoint.

        Tries to load as Seq2SeqLM (encoder-decoder), MaskedLM, or CausalLM.
        Handles special cases like decoder-only models with MLM heads and Llama models.
        """
        hf_dtype = kwargs.get("dtype", kwargs.get("torch_dtype", torch_dtype))
        rest = {k: v for k, v in kwargs.items() if k not in ("torch_dtype", "dtype")}
        load_kwargs = {"trust_remote_code": trust_remote_code, "dtype": hf_dtype, **rest}
        if load_kwargs.get("device_map") is not None:
            load_kwargs.setdefault("low_cpu_mem_usage", True)
        try:
            config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        except Exception as config_error:
            if _is_unknown_transformers_architecture_error(config_error):
                _raise_unknown_transformers_architecture_error(name_or_path, config_error)
            raise
        if getattr(config, "is_encoder_decoder", False):
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, token=HF_TOKEN, **load_kwargs)
                set_requires_grad_true(model)
                return model
            except Exception as seq2seq_error:
                if _is_unknown_transformers_architecture_error(seq2seq_error):
                    _raise_unknown_transformers_architecture_error(name_or_path, seq2seq_error)
                raise
        try:
            model = AutoModelForMaskedLM.from_pretrained(name_or_path, **load_kwargs)
        except Exception:
            config_file = os.path.join(name_or_path, 'config_mlm_head.json')
            if os.path.exists(config_file):
                model = DecoderModelWithMLMHead.from_pretrained(name_or_path, **load_kwargs)
            else:
                try:
                    model = AutoModelForCausalLM.from_pretrained(name_or_path, token=HF_TOKEN, **load_kwargs)
                except Exception as causal_error:
                    if _is_unknown_transformers_architecture_error(causal_error):
                        _raise_unknown_transformers_architecture_error(name_or_path, causal_error)
                    raise
        set_requires_grad_true(model)
        return model


class InstructTokenizerWrapper:
    """
    Wrapper for instruction-tuned model tokenizers.
    Adds system prompts and proper formatting for instruction-tuned models like Llama-Instruct.
    """

    system_prompt_mlm = """
    You are a language model that fills in masked words. In the following sentence, all [MASK] tokens refer to the same word. 
    Your task is to predict the missing word and return only that word — no explanation, no formatting, nothing else.
    """

    system_prompt = """
    You are a language model that completes sentences. Predict the next word that naturally follows the given text. 
    Return only that word — no punctuation, no quotes, and no explanations.
    """


    def __init__(self, tokenizer, user_prompt_header="user", assistant_prompt_header="assistant"):
        self.tokenizer = tokenizer
        self.user_prompt_header = user_prompt_header
        self.assistant_prompt_header = assistant_prompt_header
        self.BEGIN = "<|begin_of_text|>"
        self.START = "<|start_header_id|>"
        self.END = "<|end_header_id|>"
        self.EOT = "<|eot_id|>"

    def _wrap_prompt(self, user_text):
        if isinstance(user_text, str):
            user_texts = [user_text]
        elif isinstance(user_text, list):
            user_texts = user_text
        else:
            raise TypeError("user_text must be a string or a list of strings")
        prompts = []
        for text in user_texts:
            parts = [self.BEGIN]
            if self.system_prompt:
                parts.append(f"{self.START}system{self.END}\n{self.system_prompt}\n{self.EOT}")
            parts.append(f"{self.START}{self.user_prompt_header}{self.END}\n{text}\n{self.EOT}")
            parts.append(f"{self.START}{self.assistant_prompt_header}{self.END}\n")
            prompts.append(''.join(parts))
        return prompts if len(prompts) > 1 else prompts[0]

    def __call__(self, text, **kwargs):
        if 'add_special_tokens' in kwargs and not kwargs['add_special_tokens']:
            wrapped = text
        else:
            wrapped = self._wrap_prompt(text)
        kwargs['add_special_tokens'] = False
        return self.tokenizer(wrapped, **kwargs)

    def tokenize(self, text, **kwargs):
        wrapped = self._wrap_prompt(text)
        return self.tokenizer.tokenize(wrapped, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __setattr__(self, key, value):
        if key in ['tokenizer', 'system_prompt']:
            super().__setattr__(key, value)
        else:
            setattr(self.tokenizer, key, value)


class AutoTokenizerForLM(AutoTokenizer):
    """Tokenizer loader that automatically wraps instruction-tuned model tokenizers."""

    @classmethod
    def from_pretrained(cls, name, *args, trust_remote_code=False, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            name, token=HF_TOKEN, trust_remote_code=trust_remote_code, *args, **kwargs
        )
        if "instruct" in name.lower():
            return InstructTokenizerWrapper(tokenizer)
        return tokenizer
