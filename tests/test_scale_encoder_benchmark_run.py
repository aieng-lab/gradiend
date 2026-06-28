"""Tests for scale encoder benchmark run helpers."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def bench_run():
    from experiments.scale_encoder_benchmark import models, run

    return models, run


def _spec(bench_run, model_id: str = "gpt2"):
    models, _ = bench_run
    return models.ScaleModelSpec(
        model_id,
        "gpt2",
        "decoder_only",
        "124M",
        total_params=124_000_000,
        trust_remote_code=False,
    )


def test_shuffle_specs_reproducible_with_seed(bench_run):
    _, run = bench_run
    specs = [_spec(bench_run, f"gpt2-{i}") for i in range(5)]
    a = [s.model_id for s in run._shuffle_specs(specs, seed=42)]
    b = [s.model_id for s in run._shuffle_specs(specs, seed=42)]
    assert a == b
    assert sorted(a) == sorted(s.model_id for s in specs)
    assert a != [s.model_id for s in specs]


def test_filter_recently_active_specs_skips_touched_dirs(bench_run):
    _, run = bench_run
    root = Path(tempfile.mkdtemp())
    try:
        active = _spec(bench_run, "active/model")
        idle = _spec(bench_run, "idle/model")
        active_dir = root / "active_model" / run.RUN_ID
        active_dir.mkdir(parents=True)
        (active_dir / "touch.txt").write_text("running", encoding="utf-8")

        kept, skipped = run._filter_recently_active_specs(
            [active, idle],
            output_dir=str(root),
            experiment_suffix=None,
            within_seconds=3600,
        )
        assert [s.model_id for s in kept] == ["idle/model"]
        assert skipped == ["active/model"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_filter_recently_active_specs_allows_stale_dirs(bench_run):
    _, run = bench_run
    root = Path(tempfile.mkdtemp())
    try:
        spec = _spec(bench_run, "stale/model")
        exp_dir = root / "stale_model" / run.RUN_ID
        exp_dir.mkdir(parents=True)
        stale_file = exp_dir / "old.txt"
        stale_file.write_text("done", encoding="utf-8")
        old = time.time() - 7200
        os.utime(stale_file, (old, old))

        kept, skipped = run._filter_recently_active_specs(
            [spec],
            output_dir=str(root),
            experiment_suffix=None,
            within_seconds=3600,
        )
        assert kept == [spec]
        assert skipped == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_model_queue_shuffles_after_filtering(bench_run):
    _, run = bench_run
    root = Path(tempfile.mkdtemp())
    try:
        specs = [_spec(bench_run, f"m{i}") for i in range(4)]
        queue, skipped = run._prepare_model_queue(
            specs,
            output_dir=str(root),
            experiment_suffix=None,
            shuffle=True,
            shuffle_seed=7,
            skip_recently_active=False,
            recent_activity_seconds=3600,
        )
        assert skipped == []
        assert sorted(s.model_id for s in queue) == sorted(s.model_id for s in specs)
        assert [s.model_id for s in queue] != [s.model_id for s in specs]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_existing_seeds_tried_from_report(bench_run):
    _, run = bench_run
    exp = Path(tempfile.mkdtemp())
    try:
        seeds = exp / "seeds"
        seeds.mkdir(parents=True)
        (seeds / "seed_report.json").write_text(
            json.dumps({"seeds_tried": [0, 1, 2], "max_seeds": 3}),
            encoding="utf-8",
        )
        assert run._existing_seeds_tried(str(exp)) == 3
    finally:
        shutil.rmtree(exp, ignore_errors=True)


def test_parse_use_cache_arg_accepts_policy_modes(bench_run):
    _, run = bench_run
    assert run._parse_use_cache_arg("always") == "always"
    assert run._parse_use_cache_arg("only_convergent") == "only_convergent"
    assert run._parse_use_cache_arg("true") is True
    assert run._parse_use_cache_arg("false") is False


def test_unblock_moves_model_dir_when_more_seeds_requested(bench_run):
    _, run = bench_run
    exp = Path(tempfile.mkdtemp())
    try:
        model = exp / "model"
        seeds = exp / "seeds"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        seeds.mkdir(parents=True)
        (seeds / "seed_report.json").write_text(
            json.dumps({"seeds_tried": [0, 1, 2]}),
            encoding="utf-8",
        )
        for i in range(3):
            (seeds / f"seed_{i}").mkdir()

        backup = run._unblock_multi_seed_extension(str(exp), max_seeds=10, use_cache=True)
        assert backup is not None
        assert not model.is_dir()
        assert (exp / "model.__pre_seed_extension").is_dir()
        assert run._unblock_multi_seed_extension(str(exp), max_seeds=10, use_cache=True) is None
    finally:
        shutil.rmtree(exp, ignore_errors=True)


def test_unblock_noop_when_all_seeds_present(bench_run):
    _, run = bench_run
    exp = Path(tempfile.mkdtemp())
    try:
        model = exp / "model"
        seeds = exp / "seeds"
        model.mkdir(parents=True)
        seeds.mkdir(parents=True)
        (seeds / "seed_report.json").write_text(
            json.dumps({"seeds_tried": list(range(10))}),
            encoding="utf-8",
        )
        assert run._unblock_multi_seed_extension(str(exp), max_seeds=10, use_cache=True) is None
        assert model.is_dir()
    finally:
        shutil.rmtree(exp, ignore_errors=True)
