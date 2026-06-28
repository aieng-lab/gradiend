"""Tests for pre-prune disk cache helpers."""

import json
import os

import pytest
import torch

from gradiend.trainer.core.pruning import (
    PrePruneConfig,
    build_pre_prune_cache_meta,
    load_pre_prune_cache,
    pre_prune_with_cache,
    save_pre_prune_cache,
    _validate_pre_prune_cache_meta,
)
from gradiend.util.paths import (
    has_saved_pre_prune_cache,
    remove_dir_if_empty,
    remove_pre_prune_cache,
    resolve_experiment_cache_dir,
    resolve_pre_prune_cache_dir,
)
from gradiend.visualizer.labels import format_label_with_convergence, non_convergence_marker_for_matplotlib
from gradiend.trainer.text.prediction.decoder_only_mlm import _normalize_pooling_lengths


class _DummyGradiend:
    input_dim = 10
    param_map_hash = "abc"


class _DummyModel:
    gradiend = _DummyGradiend()
    name_or_path = "mock-base"


def test_resolve_and_has_saved_pre_prune_cache(temp_dir):
    cache_dir = resolve_pre_prune_cache_dir(temp_dir)
    assert cache_dir is not None
    assert not has_saved_pre_prune_cache(cache_dir)

    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    keep_idx = torch.tensor([0, 2, 4], dtype=torch.long)
    save_pre_prune_cache(cache_dir, keep_idx, build_pre_prune_cache_meta(_DummyModel(), cfg))
    assert has_saved_pre_prune_cache(cache_dir)

    meta, loaded = load_pre_prune_cache(cache_dir)
    assert loaded.tolist() == [0, 2, 4]
    _validate_pre_prune_cache_meta(meta, _DummyModel(), cfg)


def test_remove_pre_prune_cache(temp_dir):
    cache_dir = resolve_pre_prune_cache_dir(temp_dir)
    parent_cache_dir = resolve_experiment_cache_dir(temp_dir)
    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    save_pre_prune_cache(cache_dir, torch.tensor([1]), build_pre_prune_cache_meta(_DummyModel(), cfg))
    assert os.path.isdir(parent_cache_dir)
    assert remove_pre_prune_cache(temp_dir) is True
    assert not has_saved_pre_prune_cache(cache_dir)
    assert not os.path.exists(resolve_experiment_cache_dir(temp_dir))
    assert not os.path.exists(parent_cache_dir)
    assert remove_pre_prune_cache(temp_dir) is False


def test_remove_pre_prune_cache_keeps_parent_when_other_cache_entries_remain(temp_dir):
    cache_dir = resolve_pre_prune_cache_dir(temp_dir)
    parent_cache_dir = resolve_experiment_cache_dir(temp_dir)
    gradients_dir = os.path.join(parent_cache_dir, "gradients")
    os.makedirs(gradients_dir, exist_ok=True)
    with open(os.path.join(gradients_dir, "shard.pt"), "wb") as handle:
        handle.write(b"x")

    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    save_pre_prune_cache(cache_dir, torch.tensor([1]), build_pre_prune_cache_meta(_DummyModel(), cfg))
    assert remove_pre_prune_cache(temp_dir) is True
    assert not os.path.isdir(cache_dir)
    assert os.path.isdir(parent_cache_dir)
    assert os.path.isdir(gradients_dir)


def test_remove_dir_if_empty(temp_dir):
    empty_dir = os.path.join(temp_dir, "cache")
    os.makedirs(empty_dir, exist_ok=True)
    assert remove_dir_if_empty(empty_dir) is True
    assert not os.path.exists(empty_dir)
    assert remove_dir_if_empty(empty_dir) is False

    nonempty_dir = os.path.join(temp_dir, "cache2")
    os.makedirs(nonempty_dir, exist_ok=True)
    with open(os.path.join(nonempty_dir, "keep.txt"), "w", encoding="utf-8") as handle:
        handle.write("x")
    assert remove_dir_if_empty(nonempty_dir) is False
    assert os.path.isdir(nonempty_dir)


def test_train_removes_pre_prune_cache_with_run_id(temp_dir):
    from unittest.mock import patch

    from gradiend.trainer.core.arguments import TrainingArguments
    from tests.test_trainer_model import MockTrainerForTest

    run_root = os.path.join(temp_dir, "experiment")
    run_id = "cell_a"
    resolved = os.path.join(run_root, run_id)
    os.makedirs(resolved, exist_ok=True)

    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    cache_dir = resolve_pre_prune_cache_dir(resolved)
    save_pre_prune_cache(cache_dir, torch.tensor([1]), build_pre_prune_cache_meta(_DummyModel(), cfg))
    assert has_saved_pre_prune_cache(cache_dir)

    args = TrainingArguments(
        experiment_dir=run_root,
        reuse_pre_prune=True,
        max_seeds=1,
        use_cache=False,
    )
    trainer = MockTrainerForTest(model="mock-base", run_id=run_id, args=args)
    out_path = os.path.join(resolved, "model")

    with patch.object(MockTrainerForTest, "_train", return_value=out_path):
        trainer.train(output_dir=out_path)

    assert not has_saved_pre_prune_cache(cache_dir)
    assert not os.path.exists(resolve_experiment_cache_dir(resolved))
    assert not has_saved_pre_prune_cache(resolve_pre_prune_cache_dir(run_root))


def test_train_removes_pre_prune_cache_after_finish(temp_dir):
    from unittest.mock import patch

    from gradiend.trainer.core.arguments import TrainingArguments
    from tests.test_trainer_model import MockTrainerForTest

    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    cache_dir = resolve_pre_prune_cache_dir(temp_dir)
    save_pre_prune_cache(cache_dir, torch.tensor([1]), build_pre_prune_cache_meta(_DummyModel(), cfg))
    assert has_saved_pre_prune_cache(cache_dir)

    args = TrainingArguments(
        experiment_dir=temp_dir,
        reuse_pre_prune=True,
        max_seeds=1,
        use_cache=False,
    )
    trainer = MockTrainerForTest(model="mock-base", args=args)
    out_path = os.path.join(temp_dir, "model")

    with patch.object(MockTrainerForTest, "_train", return_value=out_path):
        trainer.train(output_dir=out_path)

    assert not has_saved_pre_prune_cache(cache_dir)
    assert not os.path.exists(resolve_experiment_cache_dir(temp_dir))


def test_train_removes_pre_prune_cache_on_failure(temp_dir):
    from unittest.mock import patch

    from gradiend.trainer.core.arguments import TrainingArguments
    from tests.test_trainer_model import MockTrainerForTest

    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    cache_dir = resolve_pre_prune_cache_dir(temp_dir)
    save_pre_prune_cache(cache_dir, torch.tensor([1]), build_pre_prune_cache_meta(_DummyModel(), cfg))

    args = TrainingArguments(
        experiment_dir=temp_dir,
        reuse_pre_prune=True,
        max_seeds=1,
        use_cache=False,
    )
    trainer = MockTrainerForTest(model="mock-base", args=args)
    out_path = os.path.join(temp_dir, "model")

    with patch.object(MockTrainerForTest, "_train", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            trainer.train(output_dir=out_path)

    assert not has_saved_pre_prune_cache(cache_dir)
    assert not os.path.exists(resolve_experiment_cache_dir(temp_dir))


def test_validate_pre_prune_cache_meta_rejects_mismatch(temp_dir):
    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    meta = build_pre_prune_cache_meta(_DummyModel(), cfg)
    meta["input_dim"] = 999
    with pytest.raises(ValueError, match="input_dim"):
        _validate_pre_prune_cache_meta(meta, _DummyModel(), cfg)


def test_pre_prune_with_cache_skips_noop_topk_one(temp_dir):
    cache_dir = resolve_pre_prune_cache_dir(temp_dir)
    cfg = PrePruneConfig(n_samples=2, topk=1.0)
    model = _DummyModel()

    out = pre_prune_with_cache(
        model,
        dataset=[],
        config=cfg,
        cache_dir=cache_dir,
        reuse_cache=True,
    )
    assert out is model
    assert not has_saved_pre_prune_cache(cache_dir)


def test_pre_prune_with_cache_recomputes_on_stale_meta(temp_dir, monkeypatch):
    cache_dir = resolve_pre_prune_cache_dir(temp_dir)
    cached_cfg = PrePruneConfig(n_samples=2, topk=0.5, seed=1)
    current_cfg = PrePruneConfig(n_samples=2, topk=0.5, seed=42)
    save_pre_prune_cache(
        cache_dir,
        torch.tensor([1]),
        build_pre_prune_cache_meta(_DummyModel(), cached_cfg),
    )

    recomputed = {"called": False}

    def fake_pre_prune(model_with_gradiend, dataset, config, **kwargs):
        recomputed["called"] = True
        return model_with_gradiend, torch.tensor([2])

    monkeypatch.setattr("gradiend.trainer.core.pruning.pre_prune", fake_pre_prune)

    out = pre_prune_with_cache(
        _DummyModel(),
        dataset=[],
        config=current_cfg,
        cache_dir=cache_dir,
        reuse_cache=True,
    )
    assert recomputed["called"] is True
    assert out is not None


def test_format_label_with_convergence_marker():
    marker = non_convergence_marker_for_matplotlib()
    assert format_label_with_convergence("run_a", converged=False) == f"run_a {marker}"
    assert format_label_with_convergence("run_a", converged=False, highlight_non_convergence=False) == "run_a"
    assert format_label_with_convergence("run_a", converged=True) == "run_a"


def test_normalize_pooling_lengths():
    assert _normalize_pooling_lengths(3) == [3]
    assert _normalize_pooling_lengths(range(1, 4)) == [1, 2, 3]
    with pytest.raises(ValueError):
        _normalize_pooling_lengths([])
