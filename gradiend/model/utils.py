"""
Utility functions for GRADIEND models.
"""
import json
import logging
import os
from typing import Dict, Tuple, Optional, Any, Union, Set

import torch
import torch.nn as nn
from gradiend.util.device import cuda_unusable_runtime_error, validate_cuda_usable_if_visible

logger = logging.getLogger(__name__)


def get_hf_device_map(model: Any) -> Optional[Any]:
    """
    Return a Hugging Face/Accelerate device map from model or common wrappers.

    Accelerate stores dispatched placement on ``hf_device_map``. GRADIEND may
    receive either the dispatched HF module directly or a lightweight wrapper
    around it, so callers should use this helper instead of checking only the
    top-level object.
    """
    seen: Set[int] = set()

    def visit(obj: Any) -> Optional[Any]:
        if obj is None:
            return None
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        device_map = getattr(obj, "hf_device_map", None)
        if device_map is not None:
            return device_map

        for attr in ("base_model", "decoder", "model", "module"):
            child = getattr(obj, attr, None)
            child_map = visit(child)
            if child_map is not None:
                return child_map
        return None

    return visit(model)


# -----------------------------
# helpers for optional safetensors
# -----------------------------
def _save_tensor_dict(path: str, tensors: Dict[str, torch.Tensor], *, prefer_safetensors: bool):
    if prefer_safetensors:
        try:
            from safetensors.torch import save_file
            save_file(tensors, path)
            return
        except ImportError:
            pass
    # fallback
    torch.save(tensors, path)


def _load_tensor_dict(path: str, *, prefer_safetensors: bool) -> Dict[str, torch.Tensor]:
    if prefer_safetensors and path.endswith(".safetensors"):
        from safetensors.torch import load_file  # may raise ImportError, fine
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _tensor_file_name(stem: str, *, prefer_safetensors: bool) -> str:
    return f"{stem}.safetensors" if prefer_safetensors else f"{stem}.pth"


def save_model_weights(
    directory: str,
    state_dict: Dict[str, torch.Tensor],
    use_safetensors: Optional[bool] = None,
) -> None:
    """Save model state_dict to directory with optional safetensors support.

    Uses model.safetensors when safetensors is available (unless use_safetensors=False),
    otherwise pytorch_model.bin. Same convention as GRADIEND model and transformers.

    When the state_dict contains tied/shared tensors (e.g. GPT-2 lm_head and wte),
    safetensors.save_file() raises; we fall back to torch.save (pytorch_model.bin).
    PyTorch preserves tensor sharing on load, so memory is not duplicated in RAM.
    Alternative: if you have the nn.Module, use safetensors.torch.save_model(model, path)
    for tied weights; this API is state_dict-based so we use the .bin fallback.

    Args:
        directory: Target directory.
        state_dict: Model state dict to save.
        use_safetensors: If True, require safetensors. If False, force bin. If None, prefer safetensors.
    """
    os.makedirs(directory, exist_ok=True)
    bin_path = os.path.join(directory, "pytorch_model.bin")
    prefer = use_safetensors is not False
    if prefer:
        try:
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(directory, "model.safetensors"))
            return
        except ImportError as e:
            if use_safetensors is True:
                raise ImportError("safetensors not installed, cannot save as safetensors") from e
        except RuntimeError as e:
            # Tied weights: safetensors refuses; .bin is correct and preserves sharing on load
            if "share memory" in str(e).lower() or "duplicate memory" in str(e).lower():
                torch.save(state_dict, bin_path)
                return
            raise
    torch.save(state_dict, bin_path)


def load_model_weights(directory: str) -> Dict[str, torch.Tensor]:
    """Load model state_dict from directory.

    Tries model.safetensors first, then pytorch_model.bin. Handles nested model/ subdir.
    """
    st_path = os.path.join(directory, "model.safetensors")
    bin_path = os.path.join(directory, "pytorch_model.bin")
    if not os.path.exists(st_path) and not os.path.exists(bin_path):
        nested = os.path.join(directory, "model")
        st_path = os.path.join(nested, "model.safetensors")
        bin_path = os.path.join(nested, "pytorch_model.bin")
    if os.path.exists(st_path):
        try:
            from safetensors.torch import load_file
            return load_file(st_path)
        except ImportError:
            if os.path.exists(bin_path):
                return torch.load(bin_path, map_location="cpu", weights_only=True)
            raise
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model weights found in {directory}")


def _is_int32_safe(numel: int) -> bool:
    # indices need to address [0, numel)
    return numel <= (2**31 - 1)


def _bytes_per_index(numel: int) -> int:
    return 4 if _is_int32_safe(numel) else 8



def freeze_params_until_target(model, *target_param_names):
    """
    Freeze all params until reaching the first target param.

    Iterates through the model parameters starting from the input param
    and freezes them until it hits the first target param name.

    Args:
        model: The model whose params should be frozen
        *target_param_names: One or more target param names. Freezing stops
            when the first matching param is encountered.

    Raises:
        AssertionError: If no target param names are provided.
    """
    assert len(target_param_names) > 0, 'At least one target param name must be provided'

    # iterate through the parameters from the model, starting at the input param
    # we stop iterating as soon as we hit the first target param
    for name, param in model.named_parameters():
        if name in target_param_names:
            return

        param.requires_grad = False


def get_activation(activation: str, encoder=False):
    """
    Get an activation_encoder function by name.
    
    Args:
        activation: Name of the activation_encoder function. Supported values:
            - 'relu': ReLU activation_encoder
            - 'leakyrelu': LeakyReLU activation_encoder
            - 'tanh': Tanh activation_encoder
            - 'smht': Hardtanh activation_encoder
            - 'elu': ELU activation_encoder
            - 'gelu': GELU activation_encoder
            - 'sigmoid': Sigmoid activation_encoder
            - 'id': Identity (or LayerNorm for encoder)
            - 'silu': SiLU activation_encoder
        encoder: If True and activation_encoder is 'id', returns LayerNorm(1) instead of Identity.
    
    Returns:
        The activation_encoder function module.
    
    Raises:
        ValueError: If the activation_encoder function name is not supported.
    """
    if activation == 'relu':
        activation_fnc = nn.ReLU(inplace=True)
    elif activation == 'leakyrelu':
        activation_fnc = nn.LeakyReLU(inplace=True)
    elif activation == 'tanh':
        activation_fnc = nn.Tanh()
    elif activation == 'smht':
        activation_fnc = nn.Hardtanh()
    elif activation == 'elu':
        activation_fnc = nn.ELU()
    elif activation == 'gelu':
        activation_fnc = nn.GELU()
    elif activation == 'sigmoid':
        activation_fnc = nn.Sigmoid()
    elif activation == 'id':
        if encoder:
            activation_fnc = nn.LayerNorm(1)
        else:
            activation_fnc = nn.Identity()
    elif activation == 'silu':
        activation_fnc = nn.SiLU()
    else:
        raise ValueError('Unsupported activation_encoder function:', activation)
    return activation_fnc


def is_seq2seq_model(model_or_tokenizer):
    """
    Whether the model or tokenizer is encoder-decoder (e.g. T5, BART).

    Prefers ``config.is_encoder_decoder`` when available; falls back to class/name heuristics.
    """
    config = getattr(model_or_tokenizer, "config", None)
    if config is not None and getattr(config, "is_encoder_decoder", False):
        return True
    if hasattr(model_or_tokenizer, "is_encoder_decoder"):
        return bool(model_or_tokenizer.is_encoder_decoder)
    cls = model_or_tokenizer.__class__.__name__.lower()
    name = (getattr(model_or_tokenizer, "name_or_path", None) or "").lower()
    markers = ("t5", "bart", "mbart", "pegasus", "marian", "blenderbot", "mvp", "led")
    return any(m in cls or m in name for m in markers)


def prediction_eval_kind(model_or_tokenizer):
    """Return a ``prediction_objective``-aligned eval kind for decoder probability scoring."""
    if is_seq2seq_model(model_or_tokenizer):
        from gradiend.model.core.seq2seq_backbone import (
            DEFAULT_SEQ2SEQ_GRADIENT_MODE,
            resolve_seq2seq_gradient_mode,
        )
        return resolve_seq2seq_gradient_mode(model_or_tokenizer) or DEFAULT_SEQ2SEQ_GRADIENT_MODE
    if is_decoder_only_model(model_or_tokenizer):
        return "clm_next_token"
    return "mlm_mask_token"


def is_decoder_only_model(model_or_tokenizer):
    """
    Single place of truth: whether the model or tokenizer is decoder-only (causal LM, no [MASK]).

    Pass either a model or a tokenizer. Prefers tokenizer-style attributes when present.

    - **Tokenizer** (has ``mask_token`` or ``mask_token_id``): decoder-only if
      ``mask_token`` is None (no [MASK] token), unless the model is encoder-decoder.
    - **Model** (no mask_token): decoder-only if the class name does not indicate
      MaskedLM/MLM (e.g. GPT-2, LLaMA are decoder-only; BERT with MLM head is not).

    Args:
        model_or_tokenizer: A Hugging Face model or tokenizer.

    Returns:
        True if decoder-only (causal LM), False if masked LM or encoder-decoder.
    """
    if is_seq2seq_model(model_or_tokenizer):
        return False
    obj = model_or_tokenizer
    if hasattr(obj, "mask_token"):
        return getattr(obj, "mask_token", None) is None
    if hasattr(obj, "mask_token_id"):
        return getattr(obj, "mask_token_id", None) is None
    if hasattr(obj, "__class__") and hasattr(obj.__class__, "__name__"):
        name = obj.__class__.__name__
        if "Model" in name and "ForMaskedLM" not in name and "MLM" not in name:
            return True
        return "ForMaskedLM" not in name and "MLMHead" not in name and "WithMLM" not in name
    return False


def read_gradiend_context(load_directory: str) -> Tuple[str, str, Optional[Dict[str, int]]]:
    """
    Read source/target and optional feature_class_encoding_direction from gradiend_context.json.

    Returns:
        (source, target, feature_class_encoding_direction)
        feature_class_encoding_direction is None if not present in the file.
    """
    context_path = os.path.join(load_directory, "gradiend_context.json")
    if not os.path.isfile(context_path):
        raise FileNotFoundError(
            "GRADIEND checkpoint at {} must contain gradiend_context.json (source/target).".format(load_directory)
        )
    with open(context_path) as f:
        adapter_config = json.load(f)
    source = adapter_config["source"]
    target = adapter_config["target"]
    feature_class_encoding_direction = adapter_config.get("feature_class_encoding_direction")
    return source, target, feature_class_encoding_direction


def write_gradiend_context(
    save_directory: str,
    source: str,
    target: str,
    feature_class_encoding_direction: Optional[Dict[str, int]] = None,
) -> str:
    """
    Write source/target and optional feature_class_encoding_direction to gradiend_context.json.
    
    Args:
        save_directory: Directory to write the file to.
        source: Source for gradients (e.g., "factual", "alternative", "diff").
        target: Target for gradients (e.g., "factual", "alternative", "diff").
        feature_class_encoding_direction: Optional dict mapping class names to encoding directions (+1, -1, 0).
    
    Returns:
        Path to the written gradiend_context.json file.
    """
    os.makedirs(save_directory, exist_ok=True)
    context_path = os.path.join(save_directory, "gradiend_context.json")
    context_data = {"source": source, "target": target}
    if feature_class_encoding_direction is not None:
        context_data["feature_class_encoding_direction"] = feature_class_encoding_direction
    with open(context_path, "w") as f:
        json.dump(context_data, f, indent=2)
    return context_path


def resolve_device_config_for_model(
    device=None,
    device_encoder=None,
    device_decoder=None,
    device_base_model=None,
    encoder_decoder_same_device: bool = False,
):
    """
    Determine device configuration from GPU count.

    Placement rules:
    - **1 GPU**: encoder, decoder, base_model all on cuda:0
    - **2 GPUs**, encoder_decoder_same_device=False: encoder+base on cuda:0, decoder on cuda:1
    - **2 GPUs**, encoder_decoder_same_device=True: encoder+decoder on cuda:0, base on cuda:1
    - **>=3 GPUs**, encoder_decoder_same_device=False: encoder cuda:0, decoder cuda:1, base cuda:2
    - **>=3 GPUs**, encoder_decoder_same_device=True: encoder+decoder cuda:0, base on cuda:1,2,...
    - **0 GPUs**: all on CPU (unavoidable)
    - **CPU (explicit)**: pass device=\"cpu\" to force CPU when GPUs are available

    Override individual devices with device_encoder, device_decoder, device_base_model.
    Pass device=\"cpu\" or device=torch.device(\"cpu\") to load on CPU intentionally.

    Args:
        device: Single device for all components. Pass \"cpu\" to force CPU.
        device_encoder: Override encoder device.
        device_decoder: Override decoder device.
        device_base_model: Override base model device.
        encoder_decoder_same_device: If True, put encoder and decoder on same GPU (cuda:0),
            giving the base model the remaining GPUs.

    Returns:
        Tuple of (device_encoder, device_decoder, device_base_model, requires_multiple_gpus).
    """
    def _to_device(x):
        return torch.device(x) if isinstance(x, str) else x

    def _is_cpu_device(x):
        dev = _to_device(x)
        return dev is not None and dev.type == "cpu"

    def _is_cuda_device(x):
        dev = _to_device(x)
        return dev is not None and dev.type == "cuda"

    def _raise_unusable_cuda(cuda_count):
        raise cuda_unusable_runtime_error(cuda_count)

    explicit_device_cpu = device is not None and _is_cpu_device(device)
    explicit_device_cuda = device is not None and _is_cuda_device(device)
    individual_devices = (device_encoder, device_decoder, device_base_model)
    any_individual_cuda = any(_is_cuda_device(d) for d in individual_devices if d is not None)
    all_provided_individual_cpu = all(
        d is not None and _is_cpu_device(d) for d in individual_devices
    )
    allow_unusable_visible_cuda = explicit_device_cpu or all_provided_individual_cpu

    cuda_available, cuda_count = validate_cuda_usable_if_visible(
        allow_unusable_visible_cuda=allow_unusable_visible_cuda
    )

    if not cuda_available:
        if explicit_device_cuda or any_individual_cuda:
            _raise_unusable_cuda(cuda_count)

    # Explicit device override (e.g. device="cpu"): use for all components unless individually overridden
    if device is not None:
        dev = _to_device(device)
        device_encoder = _to_device(device_encoder) if device_encoder is not None else dev
        device_decoder = _to_device(device_decoder) if device_decoder is not None else dev
        device_base_model = _to_device(device_base_model) if device_base_model is not None else dev
        return device_encoder, device_decoder, device_base_model, device_encoder != device_decoder

    # All individual devices provided: use them
    if device_encoder is not None and device_decoder is not None and device_base_model is not None:
        device_encoder = _to_device(device_encoder)
        device_decoder = _to_device(device_decoder)
        device_base_model = _to_device(device_base_model)
        logger.debug(
            f"Device config: encoder={device_encoder}, decoder={device_decoder}, base_model={device_base_model}"
        )
        return device_encoder, device_decoder, device_base_model, device_encoder != device_decoder

    # Auto-place based on GPU count (fill only None slots)
    if cuda_count == 0:
        cpu = torch.device("cpu")
        device_encoder = _to_device(device_encoder) if device_encoder is not None else cpu
        device_decoder = _to_device(device_decoder) if device_decoder is not None else cpu
        device_base_model = _to_device(device_base_model) if device_base_model is not None else cpu
        return device_encoder, device_decoder, device_base_model, False

    if cuda_count == 1:
        # Single GPU: all on cuda:0 (no logging)
        dev = torch.device("cuda:0")
        device_encoder = _to_device(device_encoder) if device_encoder is not None else dev
        device_decoder = _to_device(device_decoder) if device_decoder is not None else dev
        device_base_model = _to_device(device_base_model) if device_base_model is not None else dev
        return device_encoder, device_decoder, device_base_model, False

    # 2+ GPUs: non-trivial placement — log
    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")
    cuda2 = torch.device("cuda:2")
    if cuda_count == 2:
        if encoder_decoder_same_device:
            device_encoder = _to_device(device_encoder) if device_encoder is not None else cuda0
            device_decoder = _to_device(device_decoder) if device_decoder is not None else cuda0
            device_base_model = _to_device(device_base_model) if device_base_model is not None else cuda1
            logger.debug(f"2 GPUs (encoder+decoder same): encoder+decoder on {device_encoder}, base on {device_base_model}")
        else:
            device_encoder = _to_device(device_encoder) if device_encoder is not None else cuda0
            device_decoder = _to_device(device_decoder) if device_decoder is not None else cuda1
            device_base_model = _to_device(device_base_model) if device_base_model is not None else device_encoder
            logger.debug(f"2 GPUs: encoder+base on {device_encoder}, decoder on {device_decoder}")
    else:
        # >=3 GPUs
        if encoder_decoder_same_device:
            device_encoder = _to_device(device_encoder) if device_encoder is not None else cuda0
            device_decoder = _to_device(device_decoder) if device_decoder is not None else cuda0
            device_base_model = _to_device(device_base_model) if device_base_model is not None else cuda1
            logger.debug(
                f"{cuda_count} GPUs (encoder+decoder same): encoder+decoder on {device_encoder}, "
                f"base on {device_base_model}"
            )
        else:
            device_encoder = _to_device(device_encoder) if device_encoder is not None else cuda0
            device_decoder = _to_device(device_decoder) if device_decoder is not None else cuda1
            device_base_model = _to_device(device_base_model) if device_base_model is not None else cuda2
            logger.debug(
                f"{cuda_count} GPUs: encoder={device_encoder}, decoder={device_decoder}, base_model={device_base_model}"
            )
    return device_encoder, device_decoder, device_base_model, True



def infer_base_model_device_map(
    name_or_path: str,
    *,
    device_encoder: Any,
    device_decoder: Any,
    device_base_model: Any = None,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    warn_if_unavailable: bool = True,
) -> Optional[Dict[str, Union[int, str]]]:
    """
    Infer a Hugging Face/Accelerate device map for placing the base model across GPUs.

    Only needed when multiple GPUs are available for the base (e.g. 3+ GPUs with split
    encoder/decoder, or 2+ GPUs with encoder_decoder_same_device). When only one GPU is
    available for the base, the caller should use .to(device) instead; accelerate is not required.

    Returns None if device_map cannot be computed. Caller should fall back to single-device loading.
    """
    try:
        from accelerate import infer_auto_device_map
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM
    except ImportError:
        if warn_if_unavailable:
            logger.warning(
                "device_map requested but accelerate is not installed. "
                "Install with: pip install accelerate. Falling back to single-device loading."
            )
        else:
            logger.debug("accelerate or transformers not available for device_map inference")
        return None

    # Build max_memory: 0 for encoder/decoder GPUs, full for the rest
    cuda_count = torch.cuda.device_count()
    if cuda_count == 0:
        return None

    def _gpu_index(dev: Any) -> Optional[int]:
        if dev.type != "cuda":
            return None
        return dev.index if dev.index is not None else 0

    reserved_indices = set()
    for d in (device_encoder, device_decoder):
        idx = _gpu_index(d)
        if idx is not None:
            reserved_indices.add(idx)

    base_gpu_count = sum(1 for i in range(cuda_count) if i not in reserved_indices)
    if base_gpu_count <= 1:
        # Only one GPU for base: use .to(device) instead; accelerate not needed
        return None

    base_idx = _gpu_index(device_base_model)
    if base_idx is not None and base_idx in reserved_indices:
        if warn_if_unavailable:
            logger.warning(
                "device_map requested but base model shares GPU with encoder/decoder. "
                "Using single-device loading."
            )
        return None

    max_memory: Dict[Union[int, str], Union[int, str]] = {}
    for i in range(cuda_count):
        if i in reserved_indices:
            max_memory[i] = "0MiB"
        else:
            try:
                total = torch.cuda.get_device_properties(i).total_memory
                max_memory[i] = f"{int(total * 0.95 / 1e9)}GiB"
            except Exception:
                logger.warning(f"Could not get total memory for cuda:{i}; Falling back to single-device loading.")
                return None

    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if hasattr(config, "init_device"):
        config.init_device = "meta"
    model_type = getattr(config, "model_type", "").lower()

    # Create minimal model for infer_auto_device_map (meta device to avoid loading weights)
    try:
        if "llama" in model_type or "mistral" in model_type or "qwen" in model_type or "gpt" in model_type:
            model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)
        else:
            try:
                model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)
            except Exception:
                model = AutoModelForMaskedLM.from_config(config, dtype=torch_dtype)
    except Exception as e:
        if warn_if_unavailable:
            logger.warning("device_map requested but could not create model from config: %s. Using single-device loading.", e)
        else:
            logger.debug("Could not create model from config for device_map: %s", e)
        return None

    try:
        device_map = infer_auto_device_map(model, max_memory=max_memory)
    except Exception as e:
        if warn_if_unavailable:
            logger.warning("device_map requested but infer_auto_device_map failed: %s. Using single-device loading.", e)
        else:
            logger.debug("infer_auto_device_map failed: %s", e)
        return None

    return device_map


def _normalize_param_name(name: str) -> str:
    # Remove common wrapper prefixes repeatedly
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                changed = True
    return name
