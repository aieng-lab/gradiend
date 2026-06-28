"""Tests for scale encoder benchmark metrics helpers."""

import pytest


@pytest.fixture(scope="module")
def metrics():
    from experiments.scale_encoder_benchmark import metrics as mod

    return mod


def test_summarize_seed_correlations_best_abs(metrics):
    seed_report = {
        "max_seeds": 3,
        "convergent_count": 2,
        "runs": [
            {"seed": 0, "eval_correlation": 0.9, "converged": True},
            {"seed": 1, "eval_correlation": -0.5, "converged": False},
            {"seed": 2, "eval_correlation": 0.95, "converged": True},
        ],
    }
    summary = metrics.summarize_seed_correlations(seed_report)
    assert summary["encoder_corr_best_abs"] == 0.95
    assert summary["encoder_corr_test_mean_abs"] == pytest.approx(0.7833333333333333)


def test_enrich_result_row_backfills_from_seed_report(metrics):
    row = {
        "model_id": "gpt2",
        "status": "partial",
        "encoder_corr_best_abs": None,
        "seed_report": {
            "max_seeds": 3,
            "convergent_count": 1,
            "runs": [{"seed": 0, "eval_correlation": 0.88, "converged": True}],
        },
    }
    enriched = metrics.enrich_result_row(row)
    assert enriched["encoder_corr_best_abs"] == 0.88
    assert enriched["convergent_fraction"] == 1 / 3


def test_merge_result_rows_prefers_ok_over_failed(metrics):
    rows = [
        {
            "model_id": "meta-llama/Llama-3.1-70B",
            "status": "failed",
            "encoder_corr_best_abs": None,
            "benchmark_tag": "local",
        },
        {
            "model_id": "meta-llama/Llama-3.1-70B",
            "status": "partial",
            "encoder_corr_best_abs": 0.82,
            "benchmark_tag": "cluster",
        },
    ]
    merged = metrics.merge_result_rows(rows)
    assert len(merged) == 1
    assert merged[0]["status"] == "partial"
    assert merged[0]["encoder_corr_best_abs"] == 0.82


def test_enrich_row_from_registry_backfills_plot_params(metrics):
    row = {"model_id": "meta-llama/Llama-3.1-70B", "encoder_corr_best_abs": 0.5}
    enriched = metrics.enrich_row_from_registry(row)
    assert enriched["plot_params"] == 70_000_000_000
    assert enriched["family"] == "llama"
    assert metrics.categorize_failure({"status": "failed", "convergent_count": 0}) == "no_convergence"
    assert metrics.categorize_failure({"status": "failed", "error": "gated repository"}) == "gated_hf"
    assert metrics.categorize_failure({"status": "partial", "convergent_count": 2}) == "partial_convergence"


def test_compute_plot_value_falls_back_to_wall_time_total():
    from experiments.scale_encoder_benchmark.runtime_stats import compute_plot_value

    row = {
        "wall_time_total_s": 3600.0,
        "runtime": {"gpu_hours_mean": None, "wall_time_train_mean_s": None},
        "base_model_device_map": {"": 0},
    }
    assert compute_plot_value(row, "wall_time_train_mean_s") == 3600.0
    assert compute_plot_value(row, "gpu_hours_mean") == 1.0
    assert compute_plot_value(row, "peak_vram_total_mean_gb") is None
