"""Tests for pre-prune disk cache helpers."""

import json
import os

import pytest
import torch

from gradiend.trainer.core.pruning import (
    PrePruneConfig,
    build_pre_prune_cache_meta,
    load_pre_prune_cache,
    save_pre_prune_cache,
    _validate_pre_prune_cache_meta,
)
from gradiend.util.paths import has_saved_pre_prune_cache, remove_pre_prune_cache, resolve_pre_prune_cache_dir
from gradiend.visualizer.labels import NON_CONVERGENCE_MARKER, format_label_with_convergence
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
    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    save_pre_prune_cache(cache_dir, torch.tensor([1]), build_pre_prune_cache_meta(_DummyModel(), cfg))
    remove_pre_prune_cache(temp_dir)
    assert not has_saved_pre_prune_cache(cache_dir)


def test_validate_pre_prune_cache_meta_rejects_mismatch(temp_dir):
    cfg = PrePruneConfig(n_samples=2, topk=0.5)
    meta = build_pre_prune_cache_meta(_DummyModel(), cfg)
    meta["input_dim"] = 999
    with pytest.raises(ValueError, match="input_dim"):
        _validate_pre_prune_cache_meta(meta, _DummyModel(), cfg)


def test_format_label_with_convergence_marker():
    assert format_label_with_convergence("run_a", converged=False) == f"run_a {NON_CONVERGENCE_MARKER}"
    assert format_label_with_convergence("run_a", converged=False, highlight_non_convergence=False) == "run_a"
    assert format_label_with_convergence("run_a", converged=True) == "run_a"


def test_normalize_pooling_lengths():
    assert _normalize_pooling_lengths(3) == [3]
    assert _normalize_pooling_lengths(range(1, 4)) == [1, 2, 3]
    with pytest.raises(ValueError):
        _normalize_pooling_lengths([])
