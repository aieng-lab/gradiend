"""
Cheap sanity checks on the training signal.

A mis-built training item (e.g. the supervised label placed on a padding token) does not raise:
the loss stays finite and every input of a class becomes bit-identical, so the encoder "converges"
by memorising one vector per class. These checks turn that failure into an immediate error.

* :class:`SignalDiversityCallback` reads the within-class spread of the encoded values that every

  evaluation already computes (``std_by_class``). It adds no forward or backward pass and no memory,
  and judges the first evaluation, before training has changed anything: with a diverse signal the
  freshly initialised encoder already spreads the examples of a class, with identical inputs it
  cannot.

* :func:`assert_signal_diverse` is the same test on an explicit matrix of inputs, for callers that

  already hold stacked training examples.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from gradiend.trainer.core.callbacks import TrainingCallback
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SignalNotDiverseError", "assert_signal_diverse", "SignalDiversityCallback"]

_ADVICE = (
    "The inputs carry no within-class information; check where the supervised label/target is "
    "placed (padding side) before training or tuning the learning rate."
)


class SignalNotDiverseError(RuntimeError):
    """Raised when the training inputs carry no within-class variation (an item-construction bug)."""


def assert_signal_diverse(
    inputs: np.ndarray,
    labels: Sequence[float],
    *,
    where: str = "training inputs",
    rel_tol: float = 1e-12,
) -> None:
    """Fail if every row of a class is (numerically) identical.

    Args:
        inputs: ``(n, d)`` stacked encoder inputs, one row per training example.
        labels: one class label per row.
    """
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("inputs must be (n, d) with one label per row")
    for label in np.unique(y):
        rows = x[y == label]
        if len(rows) < 2:
            continue
        spread = float(np.abs(rows - rows[0]).max())
        scale = float(np.abs(rows).max()) or 1.0
        if spread <= rel_tol * scale:
            raise SignalNotDiverseError(
                f"{where}: all {len(rows)} rows of class {label:g} are identical (max |x - x0| = 0). {_ADVICE}"
            )


class SignalDiversityCallback(TrainingCallback):
    """Fail fast when the first evaluation shows no within-class spread of the encoded values.

    Only classes with a non-zero label and at least ``min_examples`` evaluated examples are judged;
    the error is raised when *every* such class has a standard deviation of (numerically) zero.
    Neutral rows (label 0) are ignored. The callback judges one evaluation (the first that carries
    ``std_by_class``) and is then inert.

    Args:
        min_examples: smallest class size that counts as evidence.
        abs_tol: standard deviations at or below this are "zero".
    """

    def __init__(self, min_examples: int = 4, abs_tol: float = 1e-9) -> None:
        self.min_examples = int(min_examples)
        self.abs_tol = float(abs_tol)
        self.checked = False

    def on_step_end(self, step, loss, model, config, **kwargs):
        if self.checked:
            return None
        eval_result = kwargs.get("eval_result")
        if not isinstance(eval_result, Mapping):
            return None
        std_by_class: Optional[Mapping[Any, float]] = eval_result.get("std_by_class")
        if not std_by_class:
            return None
        self.checked = True
        n_by_class: Mapping[Any, int] = eval_result.get("n_by_class") or {}
        judged: Dict[float, float] = {}
        for key, std in std_by_class.items():
            try:
                label = float(key)
            except (TypeError, ValueError):
                continue
            if label == 0.0 or int(n_by_class.get(key, self.min_examples)) < self.min_examples:
                continue
            judged[label] = float(std)
        if judged and all(math.isfinite(s) and s <= self.abs_tol for s in judged.values()):
            raise SignalNotDiverseError(
                f"step {step}: the encoded values of every class are identical "
                f"(std by class: {judged}). {_ADVICE}"
            )
        return None
