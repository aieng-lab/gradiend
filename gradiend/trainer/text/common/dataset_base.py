"""
TextBatchedDatasetBase: shared batching/balance logic for text datasets.

Used by prediction (TextBatchedDataset, TextTrainingDataset) and classification
(ClassificationBatchedDataset). Subclasses implement __getitem__ for their schema.
"""

import random
from abc import ABC
from typing import Any, Optional

import pandas as pd
from torch.utils.data import Dataset

from gradiend.util.logging import get_logger

logger = get_logger(__name__)


class TextBatchedDatasetBase(Dataset, ABC):
    """
    Base class for text batched datasets with shared __init__ and batching logic.

    Handles: DataFrame, tokenizer, batch_size, batch_criterion, max_size, seed,
    balance_column, shuffle_within; builds group_batches and total_samples.
    Subclasses add modality-specific attributes (e.g. mask_token for prediction)
    and implement __getitem__ to return the item format expected by training.
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
        assert batch_size > 0, "batch_size must be set and > 0"
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.batch_criterion = batch_criterion
        self.seed = seed
        self.shuffle_batches = shuffle_batches or batch_size > 1
        self.max_length = max_length
        self.balance_column = balance_column
        self.shuffle_within = shuffle_within or batch_size > 1
        rng = random.Random(seed)

        if max_size is not None:
            data = data.sample(n=max_size, random_state=seed).reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)
        self.data = data

        if balance_column is not None:
            grouped_by_balance = dict(tuple(data.groupby(balance_column)))
        else:
            grouped_by_balance = {"__all__": data}

        self.group_batches = {}
        total_batches = 0

        for gname, gdata in grouped_by_balance.items():
            bc = batch_criterion
            if bc == "source_target":
                bc = ["source", "target"]
                group_keys = gdata[bc].apply(lambda row: tuple(row), axis=1)
            elif isinstance(bc, str):
                group_keys = gdata[bc]
            elif callable(bc):
                group_keys = gdata.apply(bc, axis=1)
            elif isinstance(bc, list):
                group_keys = gdata[bc].apply(lambda row: tuple(row), axis=1)
            else:
                raise TypeError("batch_criterion must be str, list, or callable")

            grouped = gdata.groupby(group_keys)
            batches = []
            for _, group in grouped:
                if len(group) < batch_size:
                    continue
                if self.shuffle_within:
                    group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
                n_full = len(group) // batch_size
                for i in range(n_full):
                    batches.append(group.iloc[i * batch_size : (i + 1) * batch_size])

            if self.shuffle_within:
                rng.shuffle(batches)
            self.group_batches[gname] = batches
            total_batches += len(batches)

            if not batches:
                counts = grouped.size()
                min_count = int(counts.min()) if len(counts) else 0
                raise ValueError(
                    f"No batches created for balance group '{gname}'. "
                    f"Each subgroup (by batch_criterion={batch_criterion!r}) must have at least "
                    f"batch_size={batch_size} samples; the smallest subgroup in this group has {min_count} "
                    f"sample(s). Use more training data (or a larger split for training) or reduce "
                    f"train_batch_size to {min_count} or less."
                )

        # ensure that each group_batches has at least one batch to avoid index errors in __getitem__
        if not self.group_batches:
            raise ValueError(
                "No batches created. Each subgroup (by batch_criterion) must have at least batch_size "
                "samples. Use more training data or reduce train_batch_size."
            )

        self.balance_keys = list(self.group_batches.keys())
        self.total_batches = total_batches
        self.total_samples = total_batches * batch_size

        logger.debug(
            f"BalancedBatchedTrainingDataset created with {self.total_batches} batches "
            f"({self.total_samples} samples) across {len(self.balance_keys)} balance groups."
        )

    def reshuffle(self):
        """Reshuffle batches inside each balance group (call at each epoch)."""
        rng = random.Random(self.seed)
        for _, batches in self.group_batches.items():
            rng.shuffle(batches)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        """Return a single row from the batched data. Subclasses may override to return a dict."""
        batch_idx = idx // self.batch_size
        in_batch_offset = idx % self.batch_size
        if self.balance_column is None:
            gname = self.balance_keys[0]
        else:
            gname = self.balance_keys[batch_idx % len(self.balance_keys)]
        group_batches = self.group_batches[gname]
        local_batch_idx = batch_idx // len(self.balance_keys)
        if local_batch_idx >= len(group_batches):
            local_batch_idx = local_batch_idx % len(group_batches)
        batch = group_batches[local_batch_idx]
        row = batch.iloc[in_batch_offset]
        return row
