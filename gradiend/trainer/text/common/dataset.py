from typing import List, Any, Optional

import torch

from gradiend.trainer import GradientTrainingDataset


class TextGradientTrainingDataset(GradientTrainingDataset):
    """
    Text modality wrapper for GradientTrainingDataset.

    Supplies padding from tokenizer (pad_token_id for 'input_ids') and cache_key_fields
    ['input_text', 'label'] when caching is used. Both keys must be present in the batch.
    """

    # Batch keys required for gradient cache hash when caching is enabled.
    CACHE_KEY_FIELDS: List[str] = ['input_text', 'label']

    def __init__(
        self,
        training_data: Any,
        tokenizer: Any,
        gradient_creator: Any,
        *,
        source: str = 'factual',
        target: str = 'diff',
        cache_dir: Optional[str] = None,
        use_cached_gradients: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        return_metadata: bool = False,
    ):
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0) if tokenizer is not None else 0

        def get_padding_value(subkey: str) -> int:
            return pad_token_id if 'input_ids' in subkey else 0

        super().__init__(
            training_data,
            gradient_creator,
            source=source,
            target=target,
            cache_dir=cache_dir,
            use_cached_gradients=use_cached_gradients,
            cache_key_fields=self.CACHE_KEY_FIELDS if (cache_dir and use_cached_gradients) else None,
            dtype=dtype,
            device=device,
            return_metadata=return_metadata,
            get_padding_value=get_padding_value,
        )
        self.tokenizer = tokenizer
