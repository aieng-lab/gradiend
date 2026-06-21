"""Tests for LR pilot output layout (per-model artifacts, no cross-model overwrite)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.modules.pop("run", None)


def _load_experiment_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_lr_pilot = _load_experiment_module(
    "scale_encoder_lr_pilot_under_test",
    "experiments/scale_encoder_benchmark/lr_pilot.py",
)

_load_trials_by_model = _lr_pilot._load_trials_by_model
_model_pilot_dir = _lr_pilot._model_pilot_dir
_model_results_path = _lr_pilot._model_results_path
_write_jsonl = _lr_pilot._write_jsonl
plot_lr_pilot_results = _lr_pilot.plot_lr_pilot_results


def _trial_row(model_id: str, lr: float, corr: float) -> dict:
    return {
        "model_id": model_id,
        "learning_rate": lr,
        "encoder_corr_test_mean": corr,
        "encoder_corr_test_std": 0.01,
        "convergent_count": 3,
        "max_seeds": 3,
        "status": "ok",
        "architecture": "encoder_only",
        "family": "bert",
    }


def test_per_model_results_paths(tmp_path):
    out = str(tmp_path)
    bert_dir = _model_pilot_dir(out, "bert-base-cased")
    _write_jsonl(_model_results_path(out, "bert-base-cased"), [_trial_row("bert-base-cased", 1e-4, 0.8)])
    _write_jsonl(_model_results_path(out, "gpt2"), [_trial_row("gpt2", 1e-5, 0.5)])

    by_model = _load_trials_by_model(out)
    assert set(by_model) == {"bert-base-cased", "gpt2"}
    assert len(by_model["bert-base-cased"]) == 1
    assert by_model["bert-base-cased"][0]["encoder_corr_test_mean"] == 0.8


def test_plot_only_writes_per_model_and_combined(tmp_path):
    pytest.importorskip("matplotlib")
    out = str(tmp_path)
    _write_jsonl(_model_results_path(out, "bert-base-cased"), [_trial_row("bert-base-cased", 1e-4, 0.8)])
    _write_jsonl(_model_results_path(out, "gpt2"), [_trial_row("gpt2", 1e-5, 0.5)])

    summary = plot_lr_pilot_results(out)

    bert_plot = os.path.join(_model_pilot_dir(out, "bert-base-cased"), "lr_vs_correlation.pdf")
    gpt_plot = os.path.join(_model_pilot_dir(out, "gpt2"), "lr_vs_correlation.pdf")
    combined = os.path.join(out, "lr_vs_correlation.pdf")

    assert os.path.isfile(bert_plot)
    assert os.path.isfile(gpt_plot)
    assert os.path.isfile(combined)
    assert summary["per_model_plots"]["bert-base-cased"] == bert_plot
    assert summary["plot_path"] == combined

    with open(os.path.join(out, "summary.json"), encoding="utf-8") as handle:
        root_summary = json.load(handle)
    assert root_summary["n_pilot_models"] == 2


def test_plot_only_single_model_does_not_require_other_models(tmp_path):
    pytest.importorskip("matplotlib")
    out = str(tmp_path)
    _write_jsonl(_model_results_path(out, "t5-base"), [_trial_row("t5-base", 1e-4, 0.6)])

    plot_lr_pilot_results(out, models=["t5-base"])

    assert os.path.isfile(os.path.join(_model_pilot_dir(out, "t5-base"), "lr_vs_correlation.pdf"))
    assert os.path.isfile(os.path.join(out, "lr_vs_correlation.pdf"))
