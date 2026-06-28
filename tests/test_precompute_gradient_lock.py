"""Tests for precompute_gradient_batches thread safety and default resolution."""

import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.util.encoding_rows import encode_dataset_to_rows
from gradiend.trainer.core.callbacks import EvaluationCallback
from gradiend.trainer.core.dataset import (
    GradientTrainingDataset,
    PreComputedTrainingDataset,
)
from gradiend.model import ParamMappedGradiendModel
from tests.conftest import SimpleMockModel, MockTokenizer
from tests.test_trainer_model import MockModelWithGradiendForTest, _make_param_map_spec
from tests.test_workflow_encoder_decoder import _make_mock_load_model

_GRADIENT_DIM = 64 * 64
_FORWARD_SLEEP_SECONDS = 0.005


class _RowDataset:
    """Minimal training rows for GradientTrainingDataset."""

    def __init__(self, n_rows: int = 24):
        self.n_rows = n_rows

    def __len__(self):
        return self.n_rows

    def __getitem__(self, index: int):
        value = float(index % 7)
        return {
            "factual": {"input_ids": torch.tensor([1, 2, 3])},
            "alternative": {"input_ids": torch.tensor([4, 5, 6])},
            "label": value,
        }


class _AccessStats:
    active = 0
    max_active = 0
    call_count = 0
    counter_lock = threading.Lock()

    @classmethod
    def reset(cls):
        with cls.counter_lock:
            cls.active = 0
            cls.max_active = 0
            cls.call_count = 0

    @classmethod
    def enter(cls):
        with cls.counter_lock:
            cls.call_count += 1
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)

    @classmethod
    def leave(cls):
        with cls.counter_lock:
            cls.active -= 1


class TrackingModelWithGradiend(MockModelWithGradiendForTest):
    """Model that records concurrent forward calls."""

    def __init__(self, *args, use_gradient_lock: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gradient_lock = use_gradient_lock

    def forward(self, inputs, return_dict=False, **kwargs):
        def _body():
            _AccessStats.enter()
            try:
                time.sleep(_FORWARD_SLEEP_SECONDS)
                return torch.randn(_GRADIENT_DIM)
            finally:
                _AccessStats.leave()

        if self.use_gradient_lock:
            with self.exclusive_base_gradient_access():
                return _body()
        return _body()


def _make_tracking_model(*, use_gradient_lock: bool = True) -> TrackingModelWithGradiend:
    base = SimpleMockModel()
    gradiend = ParamMappedGradiendModel(
        input_dim=_GRADIENT_DIM,
        latent_dim=1,
        param_map=_make_param_map_spec(),
    )
    return TrackingModelWithGradiend(base, gradiend, use_gradient_lock=use_gradient_lock)


def _gradient_dataset(model: TrackingModelWithGradiend, n_rows: int = 24) -> GradientTrainingDataset:
    return GradientTrainingDataset(
        _RowDataset(n_rows=n_rows),
        model.forward,
        source="factual",
        target="diff",
        use_cached_gradients=False,
    )


def _run_concurrent_precompute_and_eval(*, use_gradient_lock: bool, repeats: int = 3) -> int:
    """Return max observed concurrent base-gradient calls across repeated races."""
    observed_max = 0
    for _ in range(repeats):
        _AccessStats.reset()
        model = _make_tracking_model(use_gradient_lock=use_gradient_lock)
        train_ds = _gradient_dataset(model, n_rows=12)
        eval_ds = _gradient_dataset(model, n_rows=6)
        precomputed = PreComputedTrainingDataset(train_ds, buffer_size=2)

        errors = []

        def _consume_precompute():
            try:
                for _row in precomputed:
                    del _row
            except Exception as exc:
                errors.append(exc)

        def _run_eval():
            try:
                encode_dataset_to_rows(model, eval_ds)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=_consume_precompute, name="precompute-worker")
        worker.start()
        time.sleep(_FORWARD_SLEEP_SECONDS)
        _run_eval()
        worker.join(timeout=30.0)
        assert not worker.is_alive(), "precompute worker did not finish"
        if errors:
            raise errors[0]
        observed_max = max(observed_max, _AccessStats.max_active)
    return observed_max


def _training_dataframe() -> pd.DataFrame:
    rows = []
    examples = [
        ("train", "3SG", "he", "3PL", "they"),
        ("train", "3SG", "she", "3PL", "they"),
        ("train", "3PL", "they", "3SG", "he"),
        ("train", "3PL", "they", "3SG", "she"),
        ("validation", "3SG", "he", "3PL", "they"),
        ("validation", "3PL", "they", "3SG", "he"),
        ("test", "3SG", "it", "3PL", "they"),
        ("test", "3PL", "they", "3SG", "it"),
    ]
    for split, label_class, label, alternative_class, alternative in examples:
        rows.append(
            {
                "masked": "The person [MASK] went home.",
                "split": split,
                "label_class": label_class,
                "label": label,
                "alternative_class": alternative_class,
                "alternative": alternative,
            }
        )
    return pd.DataFrame(rows)


@contextmanager
def _mock_trainer(tmpdir: str, **training_kw):
    defaults = dict(
        train_batch_size=1,
        eval_steps=1,
        num_train_epochs=1,
        max_steps=2,
        learning_rate=1e-4,
        experiment_dir=os.path.join(tmpdir, "precompute_lock_test"),
        use_cache=False,
        do_eval=True,
        encoder_eval_train_max_size=2,
        precompute_gradient_batches=True,
        precompute_gradient_buffer_size=1,
        convergent_score_threshold=None,
        max_seeds=1,
        save_steps=0,
        eval_strategy="steps",
    )
    defaults.update(training_kw)
    args = TrainingArguments(**defaults)
    mock_model = SimpleMockModel(name_or_path="mock-model", dtype=torch.float32)
    mock_tokenizer = MockTokenizer(vocab_size=1000)
    with patch(
        "gradiend.trainer.text.common.model_base.TextModelWithGradiend._load_model",
        _make_mock_load_model(mock_model, mock_tokenizer),
    ):
        yield TextPredictionTrainer(
            model="bert-base-uncased",
            data=_training_dataframe(),
            max_counterfactuals_per_sentence=1,
            args=args,
        )


def _find_training_json(experiment_dir: str) -> str:
    candidates = [
        os.path.join(experiment_dir, "model", "training.json"),
        os.path.join(experiment_dir, "seeds", "seed_0", "training.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    for dirpath, _, filenames in os.walk(experiment_dir):
        if "training.json" in filenames:
            return os.path.join(dirpath, "training.json")
    raise AssertionError(f"training.json not found under {experiment_dir}")


def _load_training_json(experiment_dir: str) -> dict:
    with open(_find_training_json(experiment_dir), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_concurrent_unsafe_forward(*, use_gradient_lock: bool, repeats: int = 4) -> int:
    """Race raw forward() calls (bypasses GradientTrainingDataset row lock)."""
    observed_max = 0
    inputs = {"input_ids": torch.tensor([1, 2, 3])}
    for _ in range(repeats):
        _AccessStats.reset()
        model = _make_tracking_model(use_gradient_lock=use_gradient_lock)
        errors = []

        def _worker():
            try:
                for _ in range(6):
                    model.forward(inputs)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, name=f"forward-worker-{i}") for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
            assert not thread.is_alive(), "forward worker did not finish"
        if errors:
            raise errors[0]
        observed_max = max(observed_max, _AccessStats.max_active)
    return observed_max


def test_precompute_with_in_training_eval_records_step_scores():
    """End-to-end: precompute + in-training eval completes and records per-step scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with _mock_trainer(tmpdir) as trainer:
            trainer.train()

        stats = _load_training_json(trainer.experiment_dir)
        scores = stats["training_stats"]["scores"]
        score_steps = {int(step) for step in scores}
        assert {0, 1}.issubset(score_steps)
        assert stats["training_stats"]["global_step"] >= 2


def test_trainer_wraps_gradient_dataset_with_precomputed_wrapper():
    """Explicit precompute=True must install PreComputedTrainingDataset (not silently disabled)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "gradiend.trainer.trainer.PreComputedTrainingDataset",
            wraps=PreComputedTrainingDataset,
        ) as wrapped_ctor:
            with _mock_trainer(tmpdir) as trainer:
                trainer.train()
        wrapped_ctor.assert_called_once()
        assert wrapped_ctor.call_args.kwargs.get("buffer_size") == 1


def test_explicit_precompute_not_disabled(caplog):
    """Explicit True must not be overridden when do_eval and eval_steps are set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with caplog.at_level(logging.INFO, logger="gradiend.trainer.trainer"):
            with _mock_trainer(tmpdir) as trainer:
                trainer.train()

    messages = caplog.text
    assert "precompute_gradient_batches=True: explicitly enabled." in messages
    assert "precompute_gradient_batches disabled" not in messages
    assert "Asynchronously precomputing gradient rows" in messages


def test_precompute_auto_enable_only_for_sharded_multi_gpu():
    """None resolves to True only for sharded base models on multi-GPU CUDA."""

    class _ModelStub:
        def __init__(self, sharded: bool):
            self.base_model_is_sharded = sharded

    def _resolve(config_value, model):
        enabled = config_value
        if enabled is None:
            enabled = (
                bool(getattr(model, "base_model_is_sharded", False))
                and torch.cuda.is_available()
                and torch.cuda.device_count() > 1
            )
        return enabled

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=2
    ):
        assert _resolve(None, _ModelStub(True)) is True
        assert _resolve(None, _ModelStub(False)) is False
        assert _resolve(True, _ModelStub(False)) is True
        assert _resolve(False, _ModelStub(True)) is False


def test_model_with_gradiend_has_exclusive_base_gradient_access():
    """Built-in ModelWithGradiend exposes a re-entrant base gradient lock."""
    model = _make_tracking_model()
    with model.exclusive_base_gradient_access():
        with model.exclusive_base_gradient_access():
            pass


def test_lock_serializes_precompute_worker_and_eval():
    """Concurrent precompute + encoder eval must never overlap base-gradient work."""
    max_active = _run_concurrent_precompute_and_eval(use_gradient_lock=True)
    assert max_active == 1


def test_unsafe_forward_without_lock_can_overlap():
    """Without model-level locking, concurrent forward() calls can overlap."""
    max_active = _run_concurrent_unsafe_forward(use_gradient_lock=False)
    assert max_active >= 2


def test_safe_forward_with_lock_never_overlaps():
    """Model-level lock must serialize raw concurrent forward() calls."""
    max_active = _run_concurrent_unsafe_forward(use_gradient_lock=True)
    assert max_active == 1


def test_evaluation_callback_runs_eval_under_model_lock():
    """EvaluationCallback must hold exclusive_base_gradient_access for the whole eval."""
    model = _make_tracking_model()
    eval_ds = _gradient_dataset(model, n_rows=4)
    lock_checks = []

    def _evaluate_fn(**kwargs):
        lock_checks.append(model._base_gradient_lock._is_owned())
        rows = encode_dataset_to_rows(model, eval_ds)
        assert len(rows) == len(eval_ds)
        return {"correlation": 0.5, "mean_by_class": {"a": 0.1}}

    callback = EvaluationCallback(evaluate_fn=_evaluate_fn, n_evaluation=1, do_eval=True)
    result = callback.on_step_end(
        step=1,
        loss=0.0,
        model=model,
        config={},
        training_stats={},
    )
    assert result is not None
    assert lock_checks == [True]
    assert result["correlation"] == 0.5
