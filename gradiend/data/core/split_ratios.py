"""Train/validation/test ratio helpers."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

SplitRatiosInput = Union[Tuple[float, float, float], Sequence[float], Mapping[str, float]]


def normalize_split_ratios(
    ratios: Optional[SplitRatiosInput] = None,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> Tuple[float, float, float]:
    """Resolve train/validation/test fractions from explicit values or a container.

    Args:
        ratios: Optional split ratio container. A sequence must have order
            ``(train, validation, test)``. A mapping must contain ``"train"``,
            ``"test"``, and either ``"validation"`` or ``"val"``.
        train: Train split fraction used when ``ratios`` is ``None``.
        val: Validation split fraction used when ``ratios`` is ``None``.
        test: Test split fraction used when ``ratios`` is ``None``.

    Returns:
        A ``(train, validation, test)`` tuple of floats.

    Raises:
        ValueError: If a mapping/sequence is malformed or the resolved ratios
            do not sum to ``1.0``.
    """
    if ratios is not None:
        if isinstance(ratios, Mapping):
            if "train" not in ratios or "test" not in ratios:
                raise ValueError("split_ratios mapping must include 'train' and 'test' keys")
            val_key = "validation" if "validation" in ratios else "val"
            if val_key not in ratios:
                raise ValueError("split_ratios mapping must include 'validation' or 'val'")
            train = float(ratios["train"])
            val = float(ratios[val_key])
            test = float(ratios["test"])
        else:
            seq = tuple(ratios)
            if len(seq) != 3:
                raise ValueError(
                    "split_ratios must be a 3-tuple/sequence (train, validation, test) "
                    f"or a mapping with train/validation/test keys; got length {len(seq)}"
                )
            train, val, test = (float(seq[0]), float(seq[1]), float(seq[2]))
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("train, validation, and test ratios must sum to 1.0")
    return train, val, test


def min_vocabulary_keys_for_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> int:
    """Return the minimum distinct keys required for all non-empty split buckets.

    Args:
        train_ratio: Train split fraction.
        val_ratio: Validation split fraction.
        test_ratio: Test split fraction.

    Returns:
        The number of requested buckets whose ratio is greater than zero.
    """
    return sum(1 for ratio in (train_ratio, val_ratio, test_ratio) if ratio > 0)


__all__ = ["SplitRatiosInput", "normalize_split_ratios", "min_vocabulary_keys_for_split_ratios"]
