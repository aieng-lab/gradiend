"""
Factory functions for creating ModelWithGradiend instances for training.
"""

from typing import Any, Optional, List, Union, Type

import torch

from gradiend.util.logging import get_logger
from gradiend.model.model_with_gradiend import ModelWithGradiend

logger = get_logger(__name__)


def create_model_with_gradiend(
    model: Union[str, ModelWithGradiend],
    model_class: Type[ModelWithGradiend],
    param_map: Optional[List[str]] = None,
    activation_encoder: str = 'tanh',
    activation_decoder: str = 'id',
    bias_decoder: bool = True,
    torch_dtype: torch.dtype = torch.float32,
    latent_dim: int = 1,
    **kwargs: Any,
) -> ModelWithGradiend:
    """
    Create a ModelWithGradiend instance.
    
    This is a generic factory function that can create any type of ModelWithGradiend.
    The model_class parameter must be specified explicitly.
    
    Args:
        model: Base model name/path or ModelWithGradiend instance
        param_map: List of param names to use (None = all core model params (e.g., excluding prediction layers)
        activation_encoder: Activation function for encoder
        activation_decoder: Activation function for decoder
        bias_decoder: Whether decoder has bias
        torch_dtype: Data type for model
        latent_dim: Latent dimension (number of features)
        model_class: ModelWithGradiend subclass to use (required).
                     For text models, use TextModelWithGradiend.
        **kwargs: Additional arguments passed to model_class.from_pretrained
    
    Returns:
        ModelWithGradiend instance (of the specified type)
    
    Examples:
        # Create a text model
        from gradiend.model import TextModelWithGradiend
        model = create_model_with_gradiend("model-base-cased", model_class=TextModelWithGradiend)
    """
    if param_map is not None and not isinstance(param_map, list):
        raise TypeError(f"param_map must be a list or None, got {type(param_map).__name__}")
    if not isinstance(activation_encoder, str):
        raise TypeError(f"activation_encoder must be str, got {type(activation_encoder).__name__}")
    if not isinstance(activation_decoder, str):
        raise TypeError(f"activation_decoder must be str, got {type(activation_decoder).__name__}")
    if not isinstance(bias_decoder, bool):
        raise TypeError(f"bias_decoder must be bool, got {type(bias_decoder).__name__}")
    if not isinstance(latent_dim, int):
        raise TypeError(f"latent_dim must be int, got {type(latent_dim).__name__}")

    # If model is already a ModelWithGradiend instance, return it
    if isinstance(model, ModelWithGradiend):
        return model
    
    # Create model using the specified class
    model_with_gradiend = model_class.from_pretrained(
        model,
        param_map=param_map,
        activation_encoder=activation_encoder,
        activation_decoder=activation_decoder,
        bias_decoder=bias_decoder,
        torch_dtype=torch_dtype,
        latent_dim=latent_dim,
        is_training=True,  # Training mode for GPU allocation
        **kwargs,
    )
    
    return model_with_gradiend
