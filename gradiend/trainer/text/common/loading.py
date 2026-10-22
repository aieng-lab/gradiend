"""
Model and tokenizer loading utilities for GRADIEND text trainers.

Used by both prediction and classification. DecoderModelWithMLMHead is imported
lazily when loading a path with config_mlm_head.json to avoid circular imports.
"""

import os
from pathlib import Path

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

from gradiend.util import get_logger
from gradiend.util.util import set_requires_grad_true
from gradiend.trainer.text.prediction.decoder_only_mlm import DecoderModelWithMLMHead

try:
    from transformers import Gemma3ForCausalLM as _Gemma3ForCausalLM
except ImportError:
    _Gemma3ForCausalLM = None

HF_TOKEN = os.getenv("HF_TOKEN")
logger = get_logger(__name__)


def _is_gemma3_multimodal_text_checkpoint(config) -> bool:
    """True for Gemma 3 4B/12B/27B checkpoints that include a vision tower in config."""
    return (
        getattr(config, "model_type", None) == "gemma3"
        and getattr(config, "vision_config", None) is not None
    )


class _CatastrophicWeightLoad(Exception):
    """Raised when ``from_pretrained`` returned a model without actually loading its weights.

    ``AutoModelForMaskedLM.from_pretrained`` does not reliably RAISE for an architecture it
    cannot really load (confirmed on a real Gemma 3 4B/27B checkpoint, transformers 5.x: it
    returns a model with every backbone weight -- embeddings, every layer, ``lm_head`` --
    reported "newly initialized", plus a "this checkpoint seem corrupted" tied-weights
    warning, and no exception). The ``except Exception`` fallback to the causal-LM path
    (which already special-cases Gemma 3's multimodal wrapper, see
    ``_load_causal_lm_for_config``) therefore never ran, and every trainer-based method
    (GRADIEND/ACTIEND/AGIEND/CGA/CAGA) silently trained against a randomly initialized
    backbone. CAA/SAE were unaffected because the study loads them directly via
    ``AutoModelForCausalLM``, bypassing this class entirely.
    """


def _missing_weight_fraction(model: nn.Module, loading_info: dict) -> float:
    missing = set(loading_info.get("missing_keys") or ())
    total = sum(1 for _ in model.state_dict())
    return len(missing) / total if total else 0.0


def _catastrophic_load_message(name_or_path, model: nn.Module, loading_info: dict, *, threshold: float = 0.8):
    """The failure message when (almost) every weight came back "missing", else None.

    A legitimate head mismatch (e.g. a masked-LM head absent from a causal checkpoint)
    leaves a handful of keys missing, not the embeddings and every transformer block.
    """
    fraction = _missing_weight_fraction(model, loading_info)
    if fraction < threshold:
        return None
    missing = loading_info.get("missing_keys") or ()
    return (
        f"{name_or_path!r}: {len(missing)}/{sum(1 for _ in model.state_dict())} "
        f"({fraction:.0%}) of this model's weights came back 'missing' -- the checkpoint "
        "almost certainly does not match this architecture (e.g. a multimodal wrapper's "
        "weights live under a different key prefix). Treating the load as failed instead "
        "of silently training on a randomly initialized backbone. "
        f"Sample missing keys: {sorted(missing)[:5]}"
    )


def _raise_if_catastrophic_load(name_or_path, model: nn.Module, loading_info: dict, *, threshold: float = 0.8) -> None:
    """Fail loudly when (almost) every weight came back "missing" instead of loaded."""
    message = _catastrophic_load_message(name_or_path, model, loading_info, threshold=threshold)
    if message is not None:
        raise _CatastrophicWeightLoad(message)


def _release_accelerator_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _from_pretrained_checked(loader, name_or_path, **load_kwargs):
    """``loader.from_pretrained`` + the catastrophic-load check, FREEING a failed load first.

    Every fallback attempt used to keep the failed model alive (in the caller's local and in
    the exception traceback's frames) while the next attempt loaded. With a device map, the
    second and third copies of a 27B backbone then no longer fit an 80 GiB GPU and accelerate
    silently offloaded part of the real model to the CPU (gemma-3-27b on one A100: masked-LM
    attempt + Gemma3ForCausalLM attempt + the working AutoModelForCausalLM load). The check
    runs in a helper that does not raise, so no frame holding the failed model survives.
    """
    model, loading_info = loader.from_pretrained(name_or_path, output_loading_info=True, **load_kwargs)
    message = _catastrophic_load_message(name_or_path, model, loading_info)
    if message is None:
        return model
    del model, loading_info
    _release_accelerator_memory()
    raise _CatastrophicWeightLoad(message)


def _load_causal_lm_for_config(name_or_path, config, load_kwargs):
    """
    Load a causal LM, preferring text-only wrappers for multimodal HF auto mappings.

    Transformers maps ``gemma3`` to ``Gemma3ForConditionalGeneration`` in
    ``AutoModelForCausalLM``, which loads the vision tower even for text-only GRADIEND
    training and can yield unusable backbone gradients. ``Gemma3ForCausalLM`` is meant to
    omit vision weights and use the standard text forward path -- but on some installed
    Transformers builds its state-dict key layout does not match a real Gemma 3 4B/27B
    checkpoint's safetensors at all (confirmed on the cluster: ``Gemma3ForCausalLM.
    from_pretrained`` reports 100% of weights "missing" for both ``google/gemma-3-4b-pt``
    and ``google/gemma-3-27b-pt``, i.e. it is itself a silent no-op on those checkpoints,
    not merely "vision tower omitted"). Prefer it when it actually binds real weights;
    otherwise fall back to plain ``AutoModelForCausalLM`` (loads the full
    ``Gemma3ForConditionalGeneration`` wrapper, vision tower included) rather than hard
    failing -- that path is independently proven to work (the study's CAA/SAE stage loads
    these exact checkpoints this way, successfully, in production).
    """
    if _is_gemma3_multimodal_text_checkpoint(config):
        if _Gemma3ForCausalLM is None:
            logger.warning(
                "Gemma 3 multimodal checkpoint %r should use Gemma3ForCausalLM for text-only "
                "training, but the installed Transformers build does not provide it; falling "
                "back to AutoModelForCausalLM.",
                name_or_path,
            )
        else:
            try:
                model = _from_pretrained_checked(
                    _Gemma3ForCausalLM, name_or_path, token=HF_TOKEN, **load_kwargs
                )
            except _CatastrophicWeightLoad as exc:
                logger.warning(
                    "Gemma3ForCausalLM could not load %r (%s); its key layout does not match "
                    "this checkpoint on the installed Transformers build. Falling back to "
                    "AutoModelForCausalLM (loads the vision tower too, but is a correct text "
                    "load).",
                    name_or_path,
                    exc,
                )
            else:
                logger.info(
                    "Loading %r with Gemma3ForCausalLM (text-only; vision tower omitted).",
                    name_or_path,
                )
                return model
    return _from_pretrained_checked(AutoModelForCausalLM, name_or_path, token=HF_TOKEN, **load_kwargs)


def _is_unknown_transformers_architecture_error(error: Exception) -> bool:
    """Detect HF errors for checkpoints newer than the installed Transformers."""
    message = str(error)
    return (
        "does not recognize this architecture" in message
        or "not recognize this architecture" in message
        or "Unrecognized model" in message
    )


def _describe_local_checkpoint(path) -> str:
    """Describe what a local checkpoint directory actually contains.

    Hugging Face reports a config without a usable ``model_type`` as
    "Unrecognized model in <path>", which is indistinguishable from a genuinely
    too-new architecture. For a directory this repo wrote itself, the real cause
    is almost always an incomplete artifact -- e.g. a checkpoint promoted after
    zero seeds converged -- and blaming the Transformers version sends the reader
    somewhere unrelated.
    """
    import json
    import os

    directory = Path(path)
    config_path = directory / "config.json"
    if not config_path.exists():
        present = sorted(p.name for p in directory.iterdir())[:12]
        return (
            f"the directory has no config.json (contains: {present or 'nothing'}), "
            "so it is not a loadable model checkpoint"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        return f"config.json could not be parsed ({exc})"
    if not isinstance(config, dict) or not config:
        return "config.json is empty"
    if not config.get("model_type"):
        keys = sorted(config)[:10]
        return (
            "config.json has no 'model_type' key "
            f"(keys present: {keys}), which is what Transformers reports as an "
            "unrecognized architecture"
        )
    weights = [
        name
        for name in os.listdir(directory)
        if name.endswith((".safetensors", ".bin", ".pt"))
    ]
    if not weights:
        return (
            f"config.json declares model_type={config['model_type']!r} but the "
            "directory contains no weight files"
        )
    return (
        f"config.json declares model_type={config['model_type']!r}, which this "
        "Transformers version does not support"
    )


def _raise_unknown_transformers_architecture_error(name_or_path, error: Exception) -> None:
    version = getattr(transformers, "__version__", "unknown")
    if Path(str(name_or_path)).is_dir():
        detail = _describe_local_checkpoint(name_or_path)
        raise ValueError(
            f"Could not load the local checkpoint {str(name_or_path)!r}: {detail}. "
            "A local directory that fails to load is usually an incomplete or "
            "partially written artifact rather than a Transformers version "
            "problem -- check whether the run that produced it actually "
            "converged and finished saving. "
            f"(Installed Transformers: {version}.)"
        ) from error
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
                model = _from_pretrained_checked(
                    AutoModelForSeq2SeqLM, name_or_path, token=HF_TOKEN, **load_kwargs
                )
                set_requires_grad_true(model)
                return model
            except Exception as seq2seq_error:
                if _is_unknown_transformers_architecture_error(seq2seq_error):
                    _raise_unknown_transformers_architecture_error(name_or_path, seq2seq_error)
                raise
        try:
            # ``from_pretrained`` does not reliably raise for an architecture it cannot really
            # load (see ``_CatastrophicWeightLoad``'s docstring) -- verify the load actually
            # bound real weights before trusting this branch over the causal-LM fallback below.
            model = _from_pretrained_checked(AutoModelForMaskedLM, name_or_path, **load_kwargs)
        except Exception:
            config_file = os.path.join(name_or_path, 'config_mlm_head.json')
            if os.path.exists(config_file):
                model = DecoderModelWithMLMHead.from_pretrained(name_or_path, **load_kwargs)
            else:
                try:
                    model = _load_causal_lm_for_config(name_or_path, config, load_kwargs)
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
