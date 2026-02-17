"""
Gradient training datasets: modality-agnostic GradientTrainingDataset and text wrapper.

GradientTrainingDataset wraps any training dataset that yields batches with 'factual' and
'alternative' inputs, runs a gradient_creator on them, and returns source/target tensors.
Modality-specific: padding value when batching variable-length tensors (get_padding_value),
and when caching is used, cache_key_fields (list of batch keys to include in the hash).
"""

import os
from typing import Any, Callable, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from gradiend.util import hash_it
from gradiend.trainer.core.config import (
    factual_computation_required_keywords,
    alternative_computation_required_keywords,
    source_target_keywords,
)


class GradientTrainingDataset:
    """
    Modality-agnostic dataset for GRADIEND gradient creation and batching.

    Wraps a training dataset that provides items with 'factual' and 'alternative' inputs.
    Batches items, optionally pads variable-length tensors (via get_padding_value),
    runs gradient_creator on factual/alternative, and returns source/target tensors
    (factual, alternative, or diff). When cache_dir and use_cached_gradients are set, cache_key_fields must be provided:
    list of batch dict keys whose values are included in the cache hash (e.g. ['input_text', 'label']).
    All listed keys must be present in every batch when caching is used.

    Args:
        training_data: Dataset with __len__ and __getitem__ returning dicts containing
            at least 'factual' and 'alternative' (modality-specific, e.g. tokenizer outputs).
        gradient_creator: Callable(inputs) -> gradients tensor (e.g. model.forward_pass_create_gradients).
        source: 'factual' | 'alternative' | 'diff' | None. When None (e.g. supervised_decoder), source gradients are not computed.
        target: Same options or None. When None (e.g. supervised_encoder), target gradients are not computed.
        cache_dir: Optional directory for caching gradients.
        use_cached_gradients: If True and cache_dir set, load/save gradients.
        cache_key_fields: When caching is used, list of batch keys to include in cache hash.
            Required when cache_dir is set and use_cached_gradients is True. All keys must exist in batch.
        dtype: Tensor dtype.
        device: Device for tensors.
        return_metadata: If True, pass through batch 'metadata'.
        get_padding_value: Callable(subkey: str) -> int used when batching variable-length
            tensors (pad_sequence). Default 0. Text uses tokenizer.pad_token_id for 'input_ids'.
    """

    def __init__(
        self,
        training_data: Any,
        gradient_creator: Any,
        *,
        source: str = 'factual',
        target: str = 'diff',
        cache_dir: Optional[str] = None,
        use_cached_gradients: bool = True,
        cache_key_fields: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        return_metadata: bool = False,
        get_padding_value: Optional[Callable[[str], int]] = None,
    ):
        assert source in source_target_keywords, f'Invalid source {source}, must be one of {source_target_keywords}'
        assert target in source_target_keywords, f'Invalid target {target}, must be one of {source_target_keywords}'

        if cache_dir is not None and use_cached_gradients and (not cache_key_fields or len(cache_key_fields) == 0):
            raise ValueError(
                "When cache_dir is set and use_cached_gradients is True, cache_key_fields must be provided "
                "(list of batch keys to include in cache hash, e.g. ['input_text', 'label'])."
            )

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.training_data = training_data
        self.batch_size = getattr(training_data, 'batch_size', None) or 1
        self.gradient_creator = gradient_creator
        self.source = source
        self.target = target
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.use_cached_gradients = use_cached_gradients
        self.cache_key_fields = cache_key_fields or []
        self.dtype = dtype
        self.device = device
        self.return_metadata = return_metadata
        self._get_padding_value = get_padding_value if callable(get_padding_value) else (lambda _: 0)

    def __len__(self) -> int:
        return len(self.training_data) // self.batch_size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _merge_batch(self, indices: list) -> dict:
        """Collect items at indices and merge into one batch; pad variable-length tensors when needed."""
        if len(indices) == 1:
            return self.training_data[indices[0]]

        batch = {}
        for idx in indices:
            data = self.training_data[idx]
            for key in data:
                if key not in batch:
                    batch[key] = []
                batch[key].append(data[key])

        for key in batch:
            if not (isinstance(batch[key], list) and all(isinstance(d, dict) for d in batch[key])):
                continue
            first = batch[key][0]
            first_key = next(iter(first))
            if not hasattr(first[first_key], 'shape'):
                raise NotImplementedError(
                    'Nested dictionary structure in batch without tensor shapes detected. '
                    'This is an unexpected edge case. Please report this issue.'
                )
            needs_padding = any(
                any(d[subkey].shape != first[subkey].shape for d in batch[key])
                for subkey in first
            )
            if needs_padding:
                padded = {}
                for subkey in first:
                    tensors = [d[subkey] for d in batch[key]]
                    padding_value = self._get_padding_value(subkey)
                    padded[subkey] = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
                batch[key] = padded
            else:
                batch[key] = {subkey: torch.stack([d[subkey] for d in batch[key]]) for subkey in first}
        return batch

    def __getitem__(self, index: int) -> dict:
        indices = list(range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.training_data))))
        batch = self._merge_batch(indices)

        cache_file_factual = ''
        cache_file_alternative = ''
        if self.use_cached_gradients and self.cache_dir is not None and self.cache_key_fields:
            missing = [k for k in self.cache_key_fields if k not in batch]
            if missing:
                raise KeyError(
                    f"Cache key requires batch keys {self.cache_key_fields}; missing in batch: {missing}. "
                    "Ensure training_data yields these keys when caching is used."
                )
            h = hash_it([batch[k] for k in self.cache_key_fields] + [self.dtype])
            cache_file_factual = os.path.join(self.cache_dir, f'factual_{h}.pt')
            cache_file_alternative = os.path.join(self.cache_dir, f'alternative_{h}.pt')

        factual_gradients = None
        alternative_gradients = None

        if self.use_cached_gradients and self.cache_dir is not None and cache_file_factual:
            if os.path.exists(cache_file_factual):
                factual_gradients = torch.load(cache_file_factual, weights_only=True)
            if os.path.exists(cache_file_alternative):
                alternative_gradients = torch.load(cache_file_alternative, weights_only=True)

        requires_factual = self.source in factual_computation_required_keywords or self.target in factual_computation_required_keywords
        if factual_gradients is None and requires_factual:
            factual_inputs = batch["factual"]
            factual_gradients = self.gradient_creator(factual_inputs)
            del factual_inputs
            factual_gradients = factual_gradients.to(dtype=self.dtype, device=self.device)
            if self.use_cached_gradients and self.cache_dir is not None and cache_file_factual:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(factual_gradients, cache_file_factual)

        requires_alternative = self.source in alternative_computation_required_keywords or self.target in alternative_computation_required_keywords
        if alternative_gradients is None and requires_alternative:
            alternative_inputs = batch['alternative']
            alternative_gradients = self.gradient_creator(alternative_inputs)
            del alternative_inputs
            alternative_gradients = alternative_gradients.to(dtype=self.dtype, device=self.device)
            if self.use_cached_gradients and self.cache_dir is not None and cache_file_alternative:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(alternative_gradients, cache_file_alternative)

        if self.source == 'factual':
            source_tensor = factual_gradients
        elif self.source == 'alternative':
            source_tensor = alternative_gradients
        elif self.source == 'diff':
            source_tensor = factual_gradients - alternative_gradients
        elif self.source is None:
            source_tensor = None  # e.g. supervised_decoder: only target needed
        else:
            raise ValueError(f'Unknown source: {self.source}')

        if self.target == 'factual':
            target_tensor = factual_gradients
        elif self.target == 'alternative':
            target_tensor = alternative_gradients
        elif self.target == 'diff':
            target_tensor = source_tensor.clone() if self.source == 'diff' else (factual_gradients - alternative_gradients)
        elif self.target is None:
            target_tensor = None
        else:
            raise ValueError(f'Unknown target: {self.target}')

        del factual_gradients
        del alternative_gradients

        output = {'source': source_tensor, 'target': target_tensor}
        for key in batch:
            if key not in output and key != 'metadata':
                output[key] = batch[key]
        if self.return_metadata and 'metadata' in batch:
            output['metadata'] = batch['metadata']
        return output
