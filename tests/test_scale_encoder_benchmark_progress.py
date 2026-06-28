"""Tests for scale encoder benchmark progress plotting helpers."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def progress_mod():
    from experiments.scale_encoder_benchmark import progress as mod

    return mod


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_inventory_marks_in_progress_convergent_runs_as_partial(progress_mod):
    out_dir = Path(tempfile.mkdtemp())
    try:
        model_id = "allenai/OLMo-2-1124-7B"
        slug = model_id.replace("/", "_")
        exp_dir = out_dir / slug / progress_mod.RUN_ID
        seeds_dir = exp_dir / "seeds"
        seeds_dir.mkdir(parents=True)
        seed_report = {
            "max_seeds": 10,
            "convergent_count": 2,
            "seeds_tried": [0, 1, 2, 3],
            "runs": [
                {"seed": 0, "eval_correlation": 0.61, "converged": True},
                {"seed": 1, "eval_correlation": 0.1, "converged": False},
                {"seed": 2, "eval_correlation": 0.72, "converged": True},
                {"seed": 3, "eval_correlation": 0.05, "converged": False},
            ],
        }
        (seeds_dir / "seed_report.json").write_text(json.dumps(seed_report), encoding="utf-8")

        inventory = progress_mod.build_inventory(output_dir=str(out_dir))
        entry = next(e for e in inventory["entries"] if e["model_id"] == model_id)
        assert entry["state"] == "partial"
        assert model_id in inventory["partial_models"]
        assert model_id not in inventory["started_models"]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_rows_for_plot_prefers_seed_report_over_stale_failed_jsonl(progress_mod):
    out_dir = Path(tempfile.mkdtemp())
    try:
        model_id = "allenai/OLMo-2-1124-7B"
        slug = model_id.replace("/", "_")
        exp_dir = out_dir / slug / progress_mod.RUN_ID
        seeds_dir = exp_dir / "seeds"
        seeds_dir.mkdir(parents=True)
        seed_report = {
            "max_seeds": 10,
            "convergent_count": 1,
            "seeds_tried": [0, 1],
            "runs": [
                {"seed": 0, "eval_correlation": 0.66, "converged": True},
                {"seed": 1, "eval_correlation": 0.1, "converged": False},
            ],
        }
        (seeds_dir / "seed_report.json").write_text(json.dumps(seed_report), encoding="utf-8")

        local_results = out_dir / "results.jsonl"
        _write_jsonl(
            local_results,
            [
                {
                    "model_id": model_id,
                    "status": "failed",
                    "convergent_count": 0,
                    "encoder_corr_best_abs": None,
                }
            ],
        )

        inventory = progress_mod.build_inventory(output_dir=str(out_dir), results_path=str(local_results))
        by_model = {model_id: progress_mod.enrich_result_row(json.loads(local_results.read_text().strip()))}
        plot_rows = progress_mod._rows_for_plot(
            inventory,
            output_dir=str(out_dir),
            all_rows_by_model=by_model,
        )
        by_id = {row["model_id"]: row for row in plot_rows}
        assert model_id in by_id
        assert by_id[model_id]["encoder_corr_best_abs"] == 0.66
        assert by_id[model_id]["status"] == "partial"
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_rows_for_plot_includes_supplement_jsonl_without_local_dir(progress_mod):
    out_dir = Path(tempfile.mkdtemp())
    try:
        local_results = out_dir / "results.jsonl"
        remote_results = out_dir / "remote_results.jsonl"
        _write_jsonl(
            local_results,
            [
                {
                    "model_id": "gpt2",
                    "status": "partial",
                    "encoder_corr_best_abs": 0.9,
                    "convergent_count": 2,
                    "plot_params": 124_000_000,
                    "family": "gpt2",
                    "architecture": "decoder_only",
                }
            ],
        )
        _write_jsonl(
            remote_results,
            [
                {
                    "model_id": "meta-llama/Llama-3.1-70B",
                    "status": "partial",
                    "encoder_corr_best_abs": 0.75,
                    "convergent_count": 3,
                    "plot_params": 70_000_000_000,
                    "family": "llama",
                    "architecture": "decoder_only",
                }
            ],
        )

        inventory = progress_mod.build_inventory(
            output_dir=str(out_dir),
            results_path=str(local_results),
            supplement_results=[str(remote_results)],
        )
        merged = progress_mod.load_merged_results(str(local_results), str(remote_results))
        by_model = {str(row["model_id"]): row for row in merged}
        plot_rows = progress_mod._rows_for_plot(
            inventory,
            output_dir=str(out_dir),
            all_rows_by_model=by_model,
        )
        model_ids = {row["model_id"] for row in plot_rows}
        assert "meta-llama/Llama-3.1-70B" in model_ids
        assert "gpt2" in model_ids
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
