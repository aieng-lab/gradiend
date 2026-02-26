"""
Prediction datasets: TextBatchedDataset, TextTrainingDataset, create_masked_pair_from_text.
"""

import random
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch

from gradiend.util.logging import suppress_tokenizer_length_warning
from gradiend.trainer.text.common.dataset_base import TextBatchedDatasetBase
from gradiend.trainer.text.prediction.unified_data import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_FACTUAL,
    UNIFIED_MASKED,
)


def create_masked_pair_from_text(
    text: str,
    tokenizer: Any,
    is_decoder_only_model: bool,
    excluded_tokens: Optional[List[str]] = None,
    mask_token: Optional[str] = None,
    min_prefix_tokens: int = 5,
) -> Optional[Tuple[str, str]]:
    """Create a single (masked_text, target_token) pair from raw text.

    For MLM models: masks one random non-special token. For decoder-only models:
    uses prefix up to a random position, inserts [MASK], and uses the next token
    as target.

    Args:
        text: Raw input text to create a masked example from.
        tokenizer: Tokenizer for encoding and decoding.
        is_decoder_only_model: If True, uses prefix+[MASK] mode; else uses MLM mask mode.
        excluded_tokens: Tokens to avoid masking (e.g., target class tokens).
        mask_token: Mask token string for MLM (e.g., "[MASK]").
        min_prefix_tokens: Minimum prefix length for decoder-only mode.

    Returns:
        Tuple of (masked_text, target_token), or None if no valid pair could be created.
    """
    excluded_tokens = excluded_tokens or []
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return None
    if not is_decoder_only_model and mask_token:
        valid_indices = [i for i, token in enumerate(tokens) if not (token.startswith("[") and token.endswith("]") or (excluded_tokens and any(excl.lower() in token.lower() for excl in excluded_tokens)))]
        if not valid_indices:
            return None
        mask_idx = random.choice(valid_indices)
        target_token = tokens[mask_idx]
        tokens[mask_idx] = mask_token
        masked_text = tokenizer.convert_tokens_to_string(tokens)
        return (masked_text, target_token)
    if len(tokens) < 2:
        return None
    valid_k = [k for k in range(min_prefix_tokens, len(tokens)) if not (tokens[k].startswith("[") and tokens[k].endswith("]") or (excluded_tokens and any(excl.lower() in tokens[k].lower() for excl in excluded_tokens)))]
    if not valid_k:
        return None
    split_at = random.choice(valid_k)
    prefix_tokens = tokens[:split_at]
    true_next_token = tokens[split_at]
    prefix_str = tokenizer.convert_tokens_to_string(prefix_tokens)
    next_str = tokenizer.convert_tokens_to_string([true_next_token])
    masked_text = prefix_str + (" [MASK]" if next_str and (next_str[0] in " \t" or next_str[0] == "\u2581") else "[MASK]")
    return (masked_text, true_next_token)


class TextBatchedDataset(TextBatchedDatasetBase):
    """Prediction batched dataset with mask token support for MLM/decoder-only models.

    Adds mask_token and mask_token_id from the tokenizer and implements _create_item
    to produce input_ids, attention_mask, and labels for training.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Any,
        batch_size: int,
        batch_criterion: Any,
        max_size: Optional[int] = None,
        seed: int = 42,
        shuffle_batches: Optional[bool] = None,
        max_length: int = 256,
        balance_column: Optional[str] = None,
        shuffle_within: Optional[bool] = None,
    ):
        """Initialize the batched dataset.

        Args:
            data: DataFrame with text/label data.
            tokenizer: Tokenizer for encoding.
            batch_size: Batch size.
            batch_criterion: Criterion for batching (e.g., target key).
            max_size: Optional maximum number of samples.
            seed: Random seed.
            shuffle_batches: Whether to shuffle batches.
            max_length: Maximum sequence length.
            balance_column: Column for balancing batches.
            shuffle_within: Whether to shuffle within batches.
        """
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            batch_criterion=batch_criterion,
            max_size=max_size,
            seed=seed,
            shuffle_batches=shuffle_batches,
            max_length=max_length,
            balance_column=balance_column,
            shuffle_within=shuffle_within,
        )
        self.mask_token = getattr(tokenizer, "mask_token", None)
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None)

    def _create_item(self, text: str, target: str):
        """Create a single training item: input_ids, attention_mask, labels.

        For decoder-only: labels at last non-pad position. For MLM: labels at mask positions.
        """
        is_decoder_only_model = getattr(self, "is_decoder_only_model", False)
        if not is_decoder_only_model and self.mask_token and self.mask_token not in text:
            raise ValueError("Input text must contain at least one [MASK] token placeholder.")
        mask_count = 0 if is_decoder_only_model else (text.count(self.mask_token) if self.mask_token else 0)
        target_tokens = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        num_target_tokens = len(target_tokens)
        if is_decoder_only_model:
            expanded_text = text.split("[MASK]")[0] if "[MASK]" in text else text
        else:
            expanded_text = text.replace(self.mask_token, " ".join([self.mask_token] * num_target_tokens), 1) if mask_count == 1 and self.mask_token else text
        with suppress_tokenizer_length_warning():
            encoded = self.tokenizer(expanded_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        if is_decoder_only_model:
            last_idxs = attention_mask.sum(dim=1)
            if hasattr(self.tokenizer, "vocab") and target in self.tokenizer.vocab:
                target_idx = self.tokenizer.vocab[target]
            else:
                target_idx = self.tokenizer(target, add_special_tokens=False)["input_ids"][0]
            for i, last_idx in enumerate(last_idxs):
                if last_idx < input_ids.size(1):
                    labels[i, last_idx - 1] = target_idx
        else:
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)
            for i, idx in enumerate(mask_positions):
                b, pos = idx.tolist()
                labels[b, pos] = target_tokens[min(i, len(target_tokens) - 1)]
        return {"input_ids": input_ids.squeeze(0), "attention_mask": attention_mask.squeeze(0), "labels": labels.squeeze(0)}


class TextTrainingDataset(TextBatchedDataset):
    """Text training dataset with dict interface for TextGradientTrainingDataset.

    Produces batches with factual and alternative items for gradient-based training,
    including template, input_text, label, and token ids for both options.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Any,
        batch_size: int,
        is_decoder_only_model: bool = False,
        max_size: Optional[int] = None,
        target_key: str = "label",
        balance_column: str = "feature_class_id",
        max_length: int = 256,
        seed: Optional[int] = None,
    ):
        """Initialize the training dataset.

        Args:
            data: DataFrame with masked, factual, alternative columns (unified schema).
            tokenizer: Tokenizer for encoding.
            batch_size: Batch size.
            is_decoder_only_model: If True, uses decoder-only ([MASK] after prefix) format.
            max_size: Optional maximum number of samples.
            target_key: Key for label/target in data.
            balance_column: Column for balancing (e.g., feature_class_id).
            max_length: Maximum sequence length.
            seed: Random seed for batch ordering (default 42). Use training_args.seed for reproducibility.
        """
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            batch_criterion=target_key,
            max_size=max_size,
            balance_column=balance_column,
            max_length=max_length,
            seed=seed if seed is not None else 42,
        )
        self.target_key = target_key
        self.is_decoder_only_model = is_decoder_only_model
        if is_decoder_only_model and getattr(self.tokenizer, "pad_token", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token

    def __getitem__(self, idx: int):
        """Return a single item with factual/alternative pairs and gradient-ready structure."""
        entry = super().__getitem__(idx)
        template = entry[UNIFIED_MASKED]
        input_text = template.split("[MASK]")[0] if self.is_decoder_only_model else template.replace("[MASK]", self.mask_token)
        text = template.replace("[MASK]", entry[UNIFIED_FACTUAL])
        label = entry[self.target_key]
        try:
            import numpy as _np
            if isinstance(label, _np.generic):
                label = int(label)
        except Exception:
            pass
        item_factual = self._create_item(input_text, entry[UNIFIED_FACTUAL])
        item_alternative = self._create_item(input_text, entry[UNIFIED_ALTERNATIVE])
        out = {
            "factual": item_factual,
            "alternative": item_alternative,
            "text": text,
            "template": template,
            "input_text": input_text,
            "label": label,
            "factual_token": entry[UNIFIED_FACTUAL],
            "factual_id": entry["factual_id"],
            "alternative_token": entry[UNIFIED_ALTERNATIVE],
            "alternative_id": entry["alternative_id"],
        }
        if "feature_class_id" in entry:
            out["feature_class_id"] = entry["feature_class_id"]
        return out