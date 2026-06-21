"""Tests for trainer.multi_seed() and MultiSeedTrainerView."""

import json
import os
import shutil
from unittest.mock import MagicMock, patch

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.multi_seed import (
    MultiSeedTrainerView,
    aggregate_eval_results,
    load_seed_model_group,
    resolve_default_seed_selection,
    resolve_seed_run_entries,
)
from tests.test_trainer_model import MockTrainerForTest


def _local_temp(name: str) -> str:
    root = os.path.join(os.getcwd(), "test_artifacts")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def _write_seed_report(base_dir: str, runs: list) -> None:
    seeds_dir = os.path.join(base_dir, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)
    report = {
        "best_seed": runs[0]["seed"] if runs else 0,
        "convergent_count": sum(1 for r in runs if r.get("converged")),
        "runs": runs,
    }
    with open(os.path.join(seeds_dir, "seed_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle)


class TestAggregateEvalResults:
    def test_mean_and_std(self):
        results = [
            {"correlation": 0.8, "n_samples": 100},
            {"correlation": 0.9, "n_samples": 100},
            {"correlation": 0.7, "n_samples": 100},
        ]
        merged = aggregate_eval_results(
            results,
            [10, 11, 12],
            aggregate="mean",
            dispersion="std",
        )
        assert merged["correlation"] == pytest.approx(0.8)
        assert merged["n_samples"] == 100
        assert merged["seeds"]["n"] == 3
        assert merged["seeds"]["values"] == [10, 11, 12]
        assert merged["seeds"]["stats"]["correlation"]["std"] == pytest.approx(0.0816496580927725)

    def test_per_seed_optional(self):
        results = [{"correlation": 0.5}, {"correlation": 0.7}]
        merged = aggregate_eval_results(
            results,
            [1, 2],
            aggregate="mean",
            dispersion="none",
            return_per_seed=True,
        )
        assert merged["seeds"]["per_seed"][1]["correlation"] == 0.5
        assert merged["seeds"]["per_seed"][2]["correlation"] == 0.7


class TestResolveSeedRunEntries:
    def test_all_convergent_filters(self):
        temp_dir = _local_temp("resolve_seed_entries")
        try:
            seed_a = os.path.join(temp_dir, "seed_10")
            seed_b = os.path.join(temp_dir, "seed_11")
            os.makedirs(seed_a)
            os.makedirs(seed_b)
            _write_seed_report(
                temp_dir,
                [
                    {"seed": 10, "output_dir": seed_a, "converged": True},
                    {"seed": 11, "output_dir": seed_b, "converged": False},
                ],
            )
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            entries = resolve_seed_run_entries(trainer, "all_convergent")
            assert entries == [(10, seed_a)]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestMultiSeedTrainerView:
    def test_evaluate_encoder_aggregates(self):
        temp_dir = _local_temp("multi_seed_eval_agg")
        try:
            seed_a = os.path.join(temp_dir, "seed_10")
            seed_b = os.path.join(temp_dir, "seed_11")
            os.makedirs(seed_a)
            os.makedirs(seed_b)
            _write_seed_report(
                temp_dir,
                [
                    {"seed": 10, "output_dir": seed_a, "converged": True},
                    {"seed": 11, "output_dir": seed_b, "converged": True},
                ],
            )
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            correlations = iter([0.6, 0.8])
            load_calls = []

            def _fake_evaluate_encoder(**kwargs):
                return {"correlation": next(correlations), "n_samples": 50}

            def _fake_load_model(load_directory, **kwargs):
                load_calls.append(load_directory)
                return MagicMock(base_model=MagicMock(), tokenizer=MagicMock())

            with patch.object(trainer, "evaluate_encoder", side_effect=_fake_evaluate_encoder):
                with patch.object(trainer, "load_model", side_effect=_fake_load_model):
                    view = trainer.multi_seed(dispersion="std")
                    result = view.evaluate_encoder(split="test")

            assert result["correlation"] == pytest.approx(0.7)
            assert result["seeds"]["n"] == 2
            assert set(result["seeds"]["values"]) == {10, 11}
            assert result["seeds"]["stats"]["correlation"]["std"] == pytest.approx(0.1)
            assert len(load_calls) == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_plot_runs_per_seed(self):
        temp_dir = _local_temp("multi_seed_plot")
        try:
            seed_a = os.path.join(temp_dir, "seed_10")
            seed_b = os.path.join(temp_dir, "seed_11")
            os.makedirs(seed_a)
            os.makedirs(seed_b)
            _write_seed_report(
                temp_dir,
                [
                    {"seed": 10, "output_dir": seed_a, "converged": True},
                    {"seed": 11, "output_dir": seed_b, "converged": True},
                ],
            )
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            paths = iter(["/plots/a.png", "/plots/b.png"])

            def _fake_plot(**kwargs):
                return next(paths)

            with patch.object(trainer, "plot_encoder_distributions", side_effect=_fake_plot):
                with patch.object(
                    trainer,
                    "load_model",
                    return_value=MagicMock(base_model=MagicMock(), tokenizer=MagicMock()),
                ):
                    view = trainer.multi_seed()
                    result = view.plot_encoder_distributions(show=False)

            assert result["paths"] == ["/plots/a.png", "/plots/b.png"]
            assert result["path"] == "/plots/a.png"
            assert result["seeds"]["n"] == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_single_seed_trainer_eval_unchanged(self):
        temp_dir = _local_temp("single_seed_unchanged")
        try:
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            calls = []

            def _fake_evaluate_encoder(**kwargs):
                calls.append(kwargs)
                return {"correlation": 0.95}

            with patch.object(trainer, "evaluate_encoder", side_effect=_fake_evaluate_encoder):
                result = trainer.evaluate_encoder(split="test")

            assert result["correlation"] == 0.95
            assert len(calls) == 1
            assert "seeds" not in result
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_default_dispersion_std_when_analyze_seed_stability(self):
        temp_dir = _local_temp("dispersion_std_default")
        try:
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
            )
            view = trainer.multi_seed()
            assert view.dispersion == "std"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_default_dispersion_none_otherwise(self):
        temp_dir = _local_temp("dispersion_none_default")
        try:
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            view = trainer.multi_seed()
            assert view.dispersion == "none"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_evaluate_combined(self):
        temp_dir = _local_temp("multi_seed_evaluate")
        try:
            seed_a = os.path.join(temp_dir, "seed_10")
            seed_b = os.path.join(temp_dir, "seed_11")
            os.makedirs(seed_a)
            os.makedirs(seed_b)
            _write_seed_report(
                temp_dir,
                [
                    {"seed": 10, "output_dir": seed_a, "converged": True},
                    {"seed": 11, "output_dir": seed_b, "converged": True},
                ],
            )
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            correlations = iter([0.4, 0.6])

            def _fake_evaluate(**kwargs):
                return {"encoder": {"correlation": next(correlations)}, "decoder": {"grid": {}}}

            with patch.object(trainer, "evaluate", side_effect=_fake_evaluate):
                with patch.object(
                    trainer,
                    "load_model",
                    return_value=MagicMock(base_model=MagicMock(), tokenizer=MagicMock()),
                ):
                    result = trainer.multi_seed().evaluate()

            assert result["encoder"]["correlation"] == pytest.approx(0.5)
            assert "seeds" in result["encoder"]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestAnalyzeSeedStabilityTrainingArguments:
    def test_rejects_best_only_with_stability(self):
        with pytest.raises(ValueError, match="best_only"):
            TrainingArguments(analyze_seed_stability=True, saved_seed_runs="best_only")

    def test_allows_all_convergent(self):
        args = TrainingArguments(analyze_seed_stability=True, saved_seed_runs="all_convergent")
        assert args.analyze_seed_stability is True


class TestAnalyzeSeedStabilityTrainingArguments:
    def test_train_raises_when_not_enough_convergent(self):
        temp_dir = _local_temp("multi_seed_stability_fail")
        try:
            args = TrainingArguments(
                experiment_dir=temp_dir,
                max_seeds=2,
                min_convergent_seeds=2,
                analyze_seed_stability=True,
                convergent_score_threshold=0.99,
                convergent_mean_by_class_threshold=0.1,
                seed=10,
            )
            trainer = MockTrainerForTest(model="mock-base", args=args)
            output_dir = os.path.join(temp_dir, "selected_model")

            def _make_stats(correlation: float) -> dict:
                return {
                    "training_stats": {"correlation": correlation, "mean_by_class": {1: {-1: -0.4, 1: 0.4}}},
                    "best_score_checkpoint": {"correlation": correlation, "global_step": 1},
                    "abs_mean_by_type": {"training": 0.7},
                }

            stats_by_seed_path = {}

            def _fake_train(self, output_dir=None, args=None, model=None, model_with_gradiend_cls=None, callbacks=None, runtime_monitor=None, **kwargs):
                os.makedirs(output_dir, exist_ok=True)
                stats_by_seed_path[output_dir] = _make_stats(0.5)
                return output_dir

            with patch.object(MockTrainerForTest, "_train", new=_fake_train):
                with patch.object(trainer, "get_training_stats", side_effect=lambda p: stats_by_seed_path.get(p)):
                    with patch.object(trainer, "evaluate_encoder", return_value={"correlation": 0.5}):
                        with patch.object(MockTrainerForTest, "plot_training_convergence", return_value=None):
                            with pytest.raises(RuntimeError, match="analyze_seed_stability=True"):
                                trainer.train(output_dir=output_dir, use_cache=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestResolveDefaultSeedSelection:
    def test_stability_mode_defaults_to_all_convergent(self):
        temp_dir = _local_temp("default_seed_selection_stability")
        try:
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
            )
            assert resolve_default_seed_selection(trainer, None) == "all_convergent"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_normal_mode_defaults_to_best(self):
        temp_dir = _local_temp("default_seed_selection_best")
        try:
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            assert resolve_default_seed_selection(trainer, None) == "best"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEncoderEvalCacheDirs:
    def test_iter_encoder_eval_cache_dirs_includes_seed_path(self):
        temp_dir = _local_temp("encoder_cache_dirs")
        try:
            seed_path = os.path.join(temp_dir, "seed_10")
            os.makedirs(seed_path, exist_ok=True)
            best_dir = os.path.join(temp_dir, "model")
            os.makedirs(best_dir, exist_ok=True)
            _write_seed_report(
                temp_dir,
                [{"seed": 10, "output_dir": seed_path, "converged": True, "selection_score": 0.9}],
            )
            report_path = os.path.join(temp_dir, "seeds", "seed_report.json")
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            report["best_seed"] = 10
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle)

            trainer = MockTrainerForTest(
                model=best_dir,
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            dirs = trainer.iter_encoder_eval_cache_dirs()
            assert os.path.normpath(temp_dir) in dirs
            assert os.path.normpath(seed_path) in dirs
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestLoadSeedModelGroup:
    def test_loads_all_convergent_paths(self):
        temp_dir = _local_temp("load_seed_model_group")
        try:
            seed_a = os.path.join(temp_dir, "seed_10")
            seed_b = os.path.join(temp_dir, "seed_11")
            os.makedirs(seed_a)
            os.makedirs(seed_b)
            _write_seed_report(
                temp_dir,
                [
                    {"seed": 10, "output_dir": seed_a, "converged": True},
                    {"seed": 11, "output_dir": seed_b, "converged": True},
                ],
            )
            trainer = MockTrainerForTest(
                model=os.path.join(temp_dir, "model"),
                args=TrainingArguments(experiment_dir=temp_dir),
            )
            with patch.object(
                trainer,
                "load_model",
                side_effect=lambda path, **kw: MagicMock(base_model=MagicMock(), tokenizer=MagicMock(), path=path),
            ) as load_mock:
                models, shared_base, _ = load_seed_model_group(trainer, selection="all_convergent")
            assert len(models) == 2
            assert shared_base is not None
            assert load_mock.call_count == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
