"""Smoke tests for scale encoder benchmark plot helpers."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def plot_architecture_strip():
    from experiments.scale_encoder_benchmark.aggregate import _plot_architecture_strip

    return _plot_architecture_strip


def _sample_rows():
    return [
        {
            "model_id": "distilbert-base-uncased",
            "family": "bert",
            "architecture": "encoder_only",
            "size_label": "66M",
            "total_params": 66_000_000,
            "status": "partial",
            "encoder_corr_best_abs": 0.99,
        },
        {
            "model_id": "gpt2",
            "family": "gpt2",
            "architecture": "decoder_only",
            "size_label": "124M",
            "total_params": 124_000_000,
            "status": "partial",
            "encoder_corr_best_abs": 0.95,
        },
        {
            "model_id": "t5-base",
            "family": "t5",
            "architecture": "encoder_decoder",
            "size_label": "220M",
            "total_params": 220_000_000,
            "status": "partial",
            "encoder_corr_best_abs": 0.2,
        },
        {
            "model_id": "roberta-base",
            "family": "roberta",
            "architecture": "encoder_only",
            "size_label": "125M",
            "total_params": 125_000_000,
            "status": "failed",
            "encoder_corr_best_abs": 0.4,
        },
    ]


def test_plot_architecture_strip_writes_pdf(plot_architecture_strip):
    out_dir = Path(tempfile.mkdtemp())
    try:
        path = out_dir / "strip.pdf"
        result = plot_architecture_strip(_sample_rows(), str(path))
        assert result == str(path)
        assert path.is_file()
        assert path.stat().st_size > 500
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
