"""
Shared verification helpers for test bench verified runs.

- Assert required files exist under model path.
- Load training stats (training.json) and assert correlation/score >= threshold.
"""

import os
from typing import Optional

from gradiend import load_training_stats


# More iterations for test bench (vs examples) to ensure proper conversion.
# Keys must match TrainingArguments field names (for train(**BENCH_TRAIN_CONFIG) overrides).
BENCH_TRAIN_CONFIG = dict(
    train_batch_size=8,
    encoder_eval_max_size=200,
    eval_steps=250,
    num_train_epochs=1,
    max_steps=1000,
    source="alternative",
    target="diff",
    eval_batch_size=8,
    learning_rate=1e-5,
)

# Config with pre/post pruning (for efficiency); use as much as possible per user request.
try:
    from gradiend.trainer import PrePruneConfig, PostPruneConfig

    BENCH_TRAIN_CONFIG_WITH_PRUNING = {
        **BENCH_TRAIN_CONFIG,
        "pre_prune_config": PrePruneConfig(n_samples=16, topk=0.01, source="diff"),
        "post_prune_config": PostPruneConfig(topk=0.05, part="decoder-weight"),
    }
except ImportError:
    BENCH_TRAIN_CONFIG_WITH_PRUNING = BENCH_TRAIN_CONFIG

# Config for decoder-only MLM head training
BENCH_MLM_HEAD_CONFIG = dict(
    epochs=3,
    batch_size=4,
    max_size=1000,
    use_cache=True,
)

# Minimum score (Pearson correlation) for verified runs; allow some tolerance
DEFAULT_MIN_CORRELATION = 0.5
DEFAULT_TOLERANCE = 0.15


def assert_model_files_exist(model_path: str) -> None:
    """Assert training_args.json, pytorch_model.bin (or model.safetensors), and training.json exist."""
    assert model_path, "model_path must be non-empty"
    assert os.path.isdir(model_path), f"Model directory should exist: {model_path}"
    config_path = os.path.join(model_path, "training_args.json")
    assert os.path.isfile(config_path), f"training_args.json should exist under {model_path}"
    has_weights = (
        os.path.isfile(os.path.join(model_path, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(model_path, "model.safetensors"))
    )
    assert has_weights, f"pytorch_model.bin or model.safetensors should exist under {model_path}"
    training_path = os.path.join(model_path, "training.json")
    assert os.path.isfile(training_path), f"training.json should exist under {model_path}"


def get_score_from_training_stats(model_path: str) -> Optional[float]:
    """Load training.json and return the final correlation (Pearson)."""
    data = load_training_stats(model_path)
    if not data:
        return None
    training_stats = data.get("training_stats") or {}
    corr = training_stats.get("correlation", training_stats.get("score"))
    if corr is not None:
        return float(corr)
    scores = training_stats.get("scores")
    if not scores:
        return None
    if isinstance(scores, dict):
        return float(list(scores.values())[-1]) if scores else None
    if isinstance(scores, list):
        return float(scores[-1]) if scores else None
    return None


def assert_correlation_threshold(
    model_path: str,
    min_correlation: float = DEFAULT_MIN_CORRELATION,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """
    Load training stats and assert score >= min_correlation - tolerance.
    Returns the score for logging.
    """
    score = get_score_from_training_stats(model_path)
    assert score is not None, (
        f"Could not read score from {model_path}/training.json"
    )
    # Allow NaN to fail the test
    assert score == score, f"Score is NaN in {model_path}"
    threshold = min_correlation - tolerance
    assert score >= threshold, (
        f"Score {score:.4f} should be >= {threshold:.4f} "
        f"(min={min_correlation}, tolerance={tolerance})"
    )
    return score
