"""
Utility functions for GRADIEND models.
"""
import json
import logging
import os
import re
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig

logger = logging.getLogger(__name__)


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


def is_decoder_only_model(model_or_tokenizer):
    """
    Single place of truth: whether the model or tokenizer is decoder-only (causal LM, no [MASK]).

    Pass either a model or a tokenizer. Prefers tokenizer-style attributes when present.

    - **Tokenizer** (has ``mask_token`` or ``mask_token_id``): decoder-only if
      ``mask_token`` is None (no [MASK] token).
    - **Model** (no mask_token): decoder-only if the class name does not indicate
      MaskedLM/MLM (e.g. GPT-2, LLaMA are decoder-only; BERT with MLM head is not).

    Args:
        model_or_tokenizer: A Hugging Face model or tokenizer.

    Returns:
        True if decoder-only (causal LM), False if masked LM.
    """
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


def estimate_model_size(model_name_or_path, trust_remote_code=False):
    """
    Estimate model size in billions of parameters from training_args.
    
    Always uses training_args to estimate model size. Falls back to name extraction
    only if training_args cannot be loaded.
    
    Args:
        model_name_or_path: Model name or path
        trust_remote_code: If True, pass to Hugging Face from_pretrained (e.g. for EuroBERT)
    
    Returns:
        Estimated model size in billions (float), or None if cannot determine
    """
    # Try to load config and estimate from hidden_size and num_layers
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if hasattr(config, "hidden_size") and hasattr(config, "num_hidden_layers"):
            hidden_size = config.hidden_size
            num_layers = config.num_hidden_layers
            vocab_size = getattr(config, "vocab_size", 32000)

            # Rough parameter count estimation
            # Embedding: vocab_size * hidden_size
            # Layers: num_layers * (hidden_size^2 * 12) for attention + MLP
            # This is a simplified estimate but works reasonably well
            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * (hidden_size ** 2 * 12)
            total_params = embedding_params + layer_params

            return total_params / 1e9
    except Exception:
        pass

    # Try to resolve base model from GRADIEND checkpoint metadata
    try:
        from gradiend.model import ParamMappedGradiendModel

        gradiend = ParamMappedGradiendModel.from_pretrained(model_name_or_path)
        base_model_id = gradiend.base_model_id

        if base_model_id:
            config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
            if hasattr(config, "hidden_size") and hasattr(config, "num_hidden_layers"):
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = getattr(config, "vocab_size", 32000)

                embedding_params = vocab_size * hidden_size
                layer_params = num_layers * (hidden_size ** 2 * 12)
                total_params = embedding_params + layer_params

                return total_params / 1e9
    except Exception:
        pass
    
    # Fallback: Try to extract from model name (e.g., "Llama-3.2-3B" -> 3.0)
    size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', str(model_name_or_path))
    if size_match:
        return float(size_match.group(1))
    
    return None


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
    model_name_or_path,
    torch_dtype=torch.float32,
    is_training=False,
    device=None,
    device_encoder=None,
    device_decoder=None,
    device_base_model=None,
    trust_remote_code=False,
):
    """
    Determine device configuration for models based on size, dtype, and training mode.
    
    Memory estimation considers:
    - Model size (parameters)
    - Data type (bfloat16 uses ~half memory of float32)
    - Training vs inference (training needs ~2-3x more memory)
    
    Thresholds (based on A100 80GB):
    - 1B bfloat16: ~40GB (single GPU OK for inference, may need 2 for training)
    - 3B bfloat16: 2 GPUs for inference, 3 GPUs for training
    
    Args:
        model_name_or_path: Model name or path
        torch_dtype: Data type (torch.float32, torch.bfloat16, etc.)
        is_training: If True, assumes training mode (needs more GPUs)
        device: Single device to use for all components (if provided, overrides auto-detection)
        device_encoder: Override encoder device (if None, auto-detect or use device)
        device_decoder: Override decoder device (if None, auto-detect or use device)
        device_base_model: Override base model device (if None, auto-detect or use device)
    
    Returns:
        Tuple of (device_encoder, device_decoder, device_base_model, requires_multiple_gpus).

        - device_encoder: torch.device used for GRADIEND encoder operations.
        - device_decoder: torch.device used for GRADIEND decoder operations.
        - device_base_model: torch.device used for the base model forward/backward.
        - requires_multiple_gpus: True when the heuristic suggests multiple GPUs,
          False when a single device should be sufficient.
    """
    # If a single device is provided, use it for all components unless individually overridden
    if device is not None:
        device = torch.device(device) if isinstance(device, str) else device
        if device_encoder is None:
            device_encoder = device
        if device_decoder is None:
            device_decoder = device
        if device_base_model is None:
            device_base_model = device
        logger.info(f"Using provided device for all components: {device}")
        return device_encoder, device_decoder, device_base_model, device_encoder != device_decoder
    
    # If all individual devices are provided, use them directly
    if device_encoder is not None and device_decoder is not None and device_base_model is not None:
        logger.info(f"Using provided device configuration: encoder={device_encoder}, decoder={device_decoder}, base_model={device_base_model}")
        return device_encoder, device_decoder, device_base_model, device_encoder != device_decoder


    # Check if any device is provided - if so, we should respect it and only auto-detect the rest
    has_any_device_override = device_encoder is not None or device_decoder is not None or device_base_model is not None
    
    model_size = estimate_model_size(model_name_or_path, trust_remote_code=trust_remote_code)
    cuda_count = torch.cuda.device_count()
    
    if model_size is None:
        # If we can't determine size, assume small model
        default_device = torch.device("cuda:0" if cuda_count > 0 else "cpu")
        if has_any_device_override:
            # Respect provided devices, use default for None
            if device_encoder is None:
                device_encoder = default_device
            if device_decoder is None:
                device_decoder = device_encoder
            if device_base_model is None:
                device_base_model = device_encoder
            logger.info(f"Could not determine model size. Using provided devices with defaults: encoder={device_encoder}, decoder={device_decoder}, base_model={device_base_model}")
        else:
            logger.warning(f"Could not determine model size for {model_name_or_path}. Using single device: {default_device}")
            device_encoder = device_decoder = device_base_model = default_device
        return device_encoder, device_decoder, device_base_model, device_encoder != device_decoder
    
    # Determine bytes per parameter based on dtype
    if torch_dtype in (torch.bfloat16, torch.float16):
        bytes_per_param = 2
    elif torch_dtype == torch.float32:
        bytes_per_param = 4
    else:
        bytes_per_param = 4  # Default to float32
    
    # Estimate memory needed per GPU
    # model_size is in billions, so convert to actual parameters, then to GB
    # Base model: model_size (billions) * 1e9 * bytes_per_param / (1024^3) = GB
    # Training needs ~2-3x more (gradients, optimizer states, activations)
    memory_multiplier = 3.0 if is_training else 1.5  # Conservative estimate
    estimated_memory_gb = model_size * 1e9 * bytes_per_param * memory_multiplier / (1024 ** 3)
    
    # If no GPUs available, use CPU (unless devices are explicitly provided)
    if cuda_count == 0:
        cpu_device = torch.device("cpu")
        if has_any_device_override:
            # Respect provided devices, use CPU for None
            if device_encoder is None:
                device_encoder = cpu_device
            if device_decoder is None:
                device_decoder = device_encoder
            if device_base_model is None:
                device_base_model = device_encoder
            logger.info(f"No GPUs available. Using provided devices with CPU defaults: encoder={device_encoder}, decoder={device_decoder}, base_model={device_base_model}")
        else:
            logger.info(f"No GPUs available. Using CPU for all components (estimated {estimated_memory_gb:.1f}GB)")
            device_encoder = device_decoder = device_base_model = cpu_device
        return device_encoder, device_decoder, device_base_model, device_encoder != device_decoder
    
    # Get available GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    
    # Thresholds for multi-GPU setup
    # For inference: if estimated memory > 60% of single GPU, use multiple GPUs
    # For training: if estimated memory > 40% of single GPU, use multiple GPUs
    single_gpu_threshold = 0.6 if not is_training else 0.4
    needs_multiple_gpus = estimated_memory_gb > (gpu_memory_gb * single_gpu_threshold)
    
    # If multiple GPUs needed but only one available, use single GPU with warning
    # (unless devices are explicitly provided)
    if needs_multiple_gpus and cuda_count < 2:
        single_gpu_device = torch.device("cuda:0")
        if has_any_device_override:
            # Respect provided devices, use single GPU for None
            if device_encoder is None:
                device_encoder = single_gpu_device
            if device_decoder is None:
                device_decoder = device_encoder
            if device_base_model is None:
                device_base_model = device_encoder
            logger.warning(
                f"Model ({model_size:.2f}B parameters, estimated {estimated_memory_gb:.1f}GB) "
                f"may benefit from multiple GPUs, but only 1 GPU available. "
                f"Using provided devices with single GPU defaults: encoder={device_encoder}, decoder={device_decoder}, base_model={device_base_model}"
            )
        else:
            logger.warning(
                f"Model ({model_size:.2f}B parameters, estimated {estimated_memory_gb:.1f}GB) "
                f"may benefit from multiple GPUs, but only 1 GPU available. Using single GPU: {single_gpu_device}. "
                f"Consider using CPU mode if GPU memory is insufficient."
            )
            device_encoder = device_decoder = device_base_model = single_gpu_device
        return device_encoder, device_decoder, device_base_model, False
    
    if needs_multiple_gpus:
        # Use multiple GPUs - only auto-detect devices that are None
        if device_encoder is None:
            device_encoder = torch.device("cuda:0")
        if device_decoder is None:
            device_decoder = torch.device("cuda:1")
        
        # Decide where to put base model (only if not provided)
        if device_base_model is None:
            # Check if we can fit base model on encoder GPU
            if estimated_memory_gb * 0.5 < gpu_memory_gb * 0.7:  # Base model + encoder can fit on one GPU
                device_base_model = device_encoder
                logger.info(
                    f"Using GPU configuration: encoder={device_encoder}, decoder={device_decoder}, "
                    f"base_model={device_encoder} (estimated {estimated_memory_gb:.1f}GB, GPU has {gpu_memory_gb}GB)"
                )
            elif cuda_count >= 3:
                device_base_model = torch.device("cuda:2")
                logger.info(
                    f"Using GPU configuration: encoder={device_encoder}, decoder={device_decoder}, "
                    f"base_model={device_base_model} (estimated {estimated_memory_gb:.1f}GB)"
                )
            else:
                # Only 2 GPUs available, put base model on encoder GPU (may cause OOM)
                device_base_model = device_encoder
                logger.warning(
                    f"Only 2 GPUs available. Base model will share GPU with encoder "
                    f"(estimated {estimated_memory_gb:.1f}GB on {gpu_memory_gb}GB GPU). "
                    f"This may cause OOM errors. Consider using a third GPU or reducing model size."
                )
        else:
            logger.info(
                f"Using GPU configuration: encoder={device_encoder}, decoder={device_decoder}, "
                f"base_model={device_base_model} (estimated {estimated_memory_gb:.1f}GB)"
            )
        
        return device_encoder, device_decoder, device_base_model, True
    else:
        # Single GPU is sufficient - only auto-detect devices that are None
        if device_encoder is None:
            device_encoder = torch.device("cuda:0" if cuda_count > 0 else "cpu")
        if device_decoder is None:
            device_decoder = device_encoder
        if device_base_model is None:
            device_base_model = device_encoder
        
        # Boundary warning if close to threshold
        if estimated_memory_gb > gpu_memory_gb * 0.5:
            logger.warning(
                f"Model estimated memory ({estimated_memory_gb:.1f}GB) is close to GPU capacity ({gpu_memory_gb}GB). "
                f"May experience OOM errors."
            )
        else:
            logger.debug(
                f"Model ({model_size:.2f}B parameters, {torch_dtype}, {'training' if is_training else 'inference'}) "
                f"fits on single device: {device_encoder} (estimated {estimated_memory_gb:.1f}GB)"
            )
        
        return device_encoder, device_decoder, device_base_model, False
