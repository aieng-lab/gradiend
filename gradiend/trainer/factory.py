"""
Factory functions for creating ModelWithGradiend instances for training.
"""

from typing import Optional, List, Union, Type

import torch

from gradiend.util.logging import get_logger
from gradiend.model.model_with_gradiend import ModelWithGradiend

logger = get_logger(__name__)


def create_model_with_gradiend(
    model: Union[str, ModelWithGradiend],
    model_class: Type[ModelWithGradiend],
    param_map: Optional[List[str]] = None,
    activation: str = 'tanh',
    activation_decoder: str = 'id',
    bias_decoder: bool = True,
    torch_dtype: torch.dtype = torch.float32,
    latent_dim: int = 1,
    **kwargs
    ) -> ModelWithGradiend:
    """
    Create a ModelWithGradiend instance.
    
    This is a generic factory function that can create any type of ModelWithGradiend.
    The model_class parameter must be specified explicitly.
    
    Args:
        model: Base model name/path or ModelWithGradiend instance
        param_map: List of param names to use (None = all core model params (e.g., excluding prediction layers)
        activation: Activation function for encoder
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
    # If model is already a ModelWithGradiend instance, return it
    if isinstance(model, ModelWithGradiend):
        return model
    
    # Create model using the specified class
    model_with_gradiend = model_class.from_pretrained(
        model,
        param_map=param_map,
        activation=activation,
        activation_decoder=activation_decoder,
        bias_decoder=bias_decoder,
        torch_dtype=torch_dtype,
        latent_dim=latent_dim,
        is_training=True,  # Training mode for GPU allocation
        **kwargs,
    )
    
    return model_with_gradiend
