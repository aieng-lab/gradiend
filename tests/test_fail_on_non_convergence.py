"""Tests for fail_on_non_convergence behavior."""

import pytest

from gradiend import TrainingArguments, TextPredictionTrainer


def test_maybe_fail_on_non_convergence_raises():
    args = TrainingArguments(fail_on_non_convergence=True, min_convergent_seeds=2)
    with pytest.raises(RuntimeError, match="fail_on_non_convergence=True"):
        TextPredictionTrainer._maybe_fail_on_non_convergence(
            args,
            convergent_count=1,
            min_convergent=2,
        )


def test_maybe_fail_on_non_convergence_disabled():
    args = TrainingArguments(fail_on_non_convergence=False, min_convergent_seeds=2)
    TextPredictionTrainer._maybe_fail_on_non_convergence(
        args,
        convergent_count=0,
        min_convergent=2,
    )


def test_maybe_fail_on_non_convergence_passes_when_enough_seeds():
    args = TrainingArguments(fail_on_non_convergence=True, min_convergent_seeds=2)
    TextPredictionTrainer._maybe_fail_on_non_convergence(
        args,
        convergent_count=2,
        min_convergent=2,
    )
