from experiments.scale_encoder_benchmark.metrics import dedupe_latest_rows, summarize_seed_correlations


def test_summarize_seed_correlations_uses_abs_for_best():
    seed_report = {
        "max_seeds": 3,
        "convergent_count": 1,
        "runs": [
            {"seed": 0, "converged": True, "eval_correlation": -0.95, "selection_score": -0.95},
            {"seed": 1, "converged": False, "eval_correlation": 0.2, "selection_score": 0.2},
            {"seed": 2, "converged": False, "eval_correlation": 0.1, "selection_score": 0.1},
        ],
    }
    summary = summarize_seed_correlations(seed_report)
    assert summary["encoder_corr_best_abs"] == 0.95
    assert summary["best_seed"] == 0
    assert summary["encoder_corr_test_mean_abs"] == (0.95 + 0.2 + 0.1) / 3


def test_dedupe_latest_rows_keeps_last():
    rows = [
        {"model_id": "a", "benchmark_tag": "t1", "v": 1},
        {"model_id": "a", "benchmark_tag": "t1", "v": 2},
        {"model_id": "a", "benchmark_tag": "t2", "v": 3},
    ]
    out = dedupe_latest_rows(rows)
    assert len(out) == 2
    by_tag = {row["benchmark_tag"]: row["v"] for row in out}
    assert by_tag["t1"] == 2
    assert by_tag["t2"] == 3
