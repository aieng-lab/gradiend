"""Integration tests for multi-seed analysis mode and comparison policy."""

from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import gradiend.comparison.feature_cross_encoding as feature_cross_encoding_module
from gradiend.comparison.anchor_aligned import compute_anchor_aligned_encoding_matrix
from gradiend.comparison.feature_cross_encoding import (
    build_cross_task_encoder_summary,
    compute_gradiend_transition_cross_encoding_matrix,
)
from gradiend.comparison.seed_policy import (
    analysis_seed_entries,
    comparison_seed_metadata,
    enter_analysis_mode,
    enter_analysis_mode_for_trainers,
    evaluate_encoder_for_comparison,
    models_for_comparison,
)
from gradiend.comparison.similarity import compute_similarity_matrix
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.multi_seed import MultiSeedTrainerView, is_multi_seed_view
from gradiend.trainer.core.seed_models import SeedModelGroup
from tests.test_gradiend_feature_cross_encoding import _Trainer, _unified_row
from tests.test_multi_seed_view import _local_temp, _write_seed_report
from tests.test_trainer_model import MockTrainerForTest


class _TopKModel:
    def __init__(self, indices):
        self.indices = list(indices)

    def get_topk_weights(self, part="decoder-weight", topk=100):
        return self.indices[: int(topk)]


class _StabilityTrainer(_Trainer):
    """Minimal trainer with on-disk seed report for cross-task multi-seed tests."""

    def __init__(self, experiment_dir: str, trainer_id: str, target_classes, rows):
        super().__init__(trainer_id, target_classes, rows, experiment_dir=experiment_dir)
        self.training_args = TrainingArguments(
            experiment_dir=experiment_dir,
            analyze_seed_stability=True,
            saved_seed_runs="all_convergent",
        )

    def get_seed_report(self):
        path = os.path.join(self.experiment_dir, "seeds", "seed_report.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def get_best_seed_run_path(self):
        report = self.get_seed_report() or {}
        best = report.get("best_seed")
        for run in report.get("runs", []):
            if run.get("seed") == best:
                return run.get("output_dir")
        runs = report.get("runs") or []
        return runs[0].get("output_dir") if runs else self.experiment_dir

    @property
    def model_path(self):
        return self.get_best_seed_run_path() or self.experiment_dir

    def multi_seed(
        self,
        *,
        selection: str = "all_convergent",
        aggregate: str = "mean",
        dispersion: str | None = None,
        return_per_seed: bool = False,
    ) -> MultiSeedTrainerView:
        if dispersion is None:
            dispersion = "std" if self.training_args.analyze_seed_stability else "none"
        return MultiSeedTrainerView(
            self,
            selection=selection,
            aggregate=aggregate,
            dispersion=dispersion,
            return_per_seed=return_per_seed,
        )


def test_seed_model_group_works_with_similarity_matrix():
    group_a = SeedModelGroup(
        [_TopKModel([1, 2]), _TopKModel([1, 3])],
        selection="all_convergent",
        seed_values=[10, 11],
    )
    group_b = SeedModelGroup(
        [_TopKModel([2, 3]), _TopKModel([3, 4])],
        selection="all_convergent",
        seed_values=[10, 11],
    )
    result = compute_similarity_matrix(
        {"a": group_a, "b": group_b},
        measure="topk_overlap",
        topk=2,
        dispersion="std",
    )
    assert result["multi_seed"] is True
    assert result["matrix"][0][1] == pytest.approx(0.375)
    assert len(group_a) == 2
    assert group_a.primary.get_topk_weights(topk=2) == [1, 2]


def test_get_model_returns_seed_group_for_multi_seed_view():
    temp_dir = _local_temp("get_model_seed_group")
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
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        mock_model = MagicMock(name="seed_model")
        with patch.object(trainer, "load_model", return_value=mock_model) as load_model:
            view = trainer.multi_seed()
            result = view.get_model()

        assert isinstance(result, SeedModelGroup)
        assert len(result) == 2
        assert load_model.call_count == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_model_single_seed_returns_model_not_group():
    temp_dir = _local_temp("get_model_single")
    try:
        seed_a = os.path.join(temp_dir, "seed_10")
        os.makedirs(seed_a)
        _write_seed_report(
            temp_dir,
            [{"seed": 10, "output_dir": seed_a, "converged": True}],
        )
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir),
        )
        mock_model = MagicMock(name="seed_model")
        with patch.object(trainer, "load_model", return_value=mock_model):
            view = trainer.multi_seed(selection="best")
            result = view.get_model()

        assert isinstance(result, MagicMock)
        assert not isinstance(result, SeedModelGroup)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_models_for_comparison_returns_group_when_stability_enabled():
    temp_dir = _local_temp("models_for_comparison_group")
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
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        mock_model = MagicMock(name="seed_model")
        with patch.object(trainer, "load_model", return_value=mock_model):
            view = enter_analysis_mode(trainer)
            model_value, _, _ = models_for_comparison(view, gradiend_only=False)

        assert isinstance(model_value, SeedModelGroup)
        assert len(model_value) == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evaluate_encoder_for_comparison_delegates_to_view():
    temp_dir = _local_temp("eval_for_comparison")
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
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        calls = {"n": 0}

        def _fake_evaluate_encoder(**kwargs):
            calls["n"] += 1
            value = 1.0 if calls["n"] == 1 else 0.0
            return {
                "correlation": 0.5,
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "masked": "m",
                            "source_id": "A",
                            "target_id": "B",
                            "encoded": value,
                        }
                    ]
                ),
            }

        with patch.object(trainer, "evaluate_encoder", side_effect=_fake_evaluate_encoder):
            with patch.object(trainer, "load_model", return_value=MagicMock(base_model=MagicMock(), tokenizer=MagicMock())):
                result = evaluate_encoder_for_comparison(trainer, split="test", return_df=True)

        assert calls["n"] == 2
        assert float(result["encoder_df"].iloc[0]["encoded"]) == pytest.approx(0.5)
        assert result["seeds"]["n"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_enter_analysis_mode_for_trainers_mixed_stability():
    temp_dir = _local_temp("enter_analysis_mixed")
    try:
        stable = MockTrainerForTest(
            model=os.path.join(temp_dir, "stable"),
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        plain = MockTrainerForTest(
            model=os.path.join(temp_dir, "plain"),
            args=TrainingArguments(experiment_dir=temp_dir),
        )
        wrapped = enter_analysis_mode_for_trainers({"stable": stable, "plain": plain})
        assert is_multi_seed_view(wrapped["stable"])
        assert wrapped["plain"] is plain
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_analysis_seed_entries_from_view_matches_selection():
    temp_dir = _local_temp("analysis_seed_entries")
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
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        view = trainer.multi_seed()
        assert analysis_seed_entries(view) == [(10, seed_a), (11, seed_b)]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_comparison_seed_metadata_switches_when_any_trainer_is_stable():
    temp_dir = _local_temp("comparison_seed_metadata")
    try:
        stable = MockTrainerForTest(
            model=os.path.join(temp_dir, "s"),
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        plain = MockTrainerForTest(
            model=os.path.join(temp_dir, "p"),
            args=TrainingArguments(experiment_dir=temp_dir),
        )
        meta = comparison_seed_metadata({"a": stable, "b": plain})
        assert meta["multi_seed"] is True
        assert meta["seed_selection"] == "all_convergent"
        assert meta["dispersion"] == "std"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_cross_task_encoder_summary_multi_seed_aggregates(monkeypatch):
    temp_dir = _local_temp("cross_task_multi_seed")
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
        rows = [
            _unified_row(
                masked="[MASK] he",
                factual="he",
                alternative="she",
                factual_class="3SG",
                alternative_class="3PL",
            ),
        ]
        trainer = _StabilityTrainer(temp_dir, "pronoun_3SG_3PL", ("3SG", "3PL"), rows)
        loaded_paths: list[str | None] = []

        def _fake_build_cross_task_encoder_df_for_seed(
            base_trainer,
            *,
            trainer_id,
            df,
            split,
            max_size,
            use_cache_effective,
            expected_transitions,
            load_directory=None,
            allow_disk_cache=True,
        ):
            loaded_paths.append(load_directory)
            encoded = 1.0 if load_directory and "seed_10" in load_directory else 0.0
            return pd.DataFrame(
                [
                    {
                        "masked": "[MASK] he",
                        "source_id": "3SG",
                        "target_id": "3PL",
                        "encoded": encoded,
                        "type": "training",
                    }
                ]
            )

        monkeypatch.setattr(
            feature_cross_encoding_module,
            "_build_cross_task_encoder_df_for_seed",
            _fake_build_cross_task_encoder_df_for_seed,
        )

        summary = build_cross_task_encoder_summary(
            {trainer.run_id: enter_analysis_mode(trainer)},
            [],
            split="test",
            use_cache=False,
        )
        payload = summary[trainer.run_id]
        assert payload["multi_seed"] is True
        assert payload["n_seeds"] == 2
        assert len(loaded_paths) == 2
        assert float(payload["encoder_df"].iloc[0]["encoded"]) == pytest.approx(0.5)
        assert len(payload["encoder_dfs"]) == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_transition_cross_encoding_matrix_uses_multi_seed_summary(monkeypatch):
    temp_dir = _local_temp("transition_matrix_multi_seed")
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
        rows = [
            _unified_row(
                masked="[MASK] he",
                factual="he",
                alternative="she",
                factual_class="3SG",
                alternative_class="3PL",
            ),
        ]
        trainer = _StabilityTrainer(temp_dir, "pronoun_3SG_3PL", ("3SG", "3PL"), rows)

        def _fake_build_cross_task_encoder_df_for_seed(base_trainer, **kwargs):
            load_directory = kwargs.get("load_directory")
            encoded = 1.0 if load_directory and "seed_10" in load_directory else 0.0
            return pd.DataFrame(
                [
                    {
                        "masked": "[MASK] he",
                        "source_id": "3SG",
                        "target_id": "3PL",
                        "encoded": encoded,
                        "type": "training",
                    }
                ]
            )

        monkeypatch.setattr(
            feature_cross_encoding_module,
            "_build_cross_task_encoder_df_for_seed",
            _fake_build_cross_task_encoder_df_for_seed,
        )

        trainers = {trainer.run_id: enter_analysis_mode(trainer)}
        result = compute_gradiend_transition_cross_encoding_matrix(
            trainers,
            trainer_order=[trainer.run_id],
            use_cache=False,
        )
        assert result["multi_seed"] is True
        assert result["matrix"][0][0] == pytest.approx(0.5)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_anchor_aligned_matrix_uses_mean_encoder_summary():
    encoder_summary = {
        "ab": {
            "encoder_df": pd.DataFrame(),
            "encoder_dfs": [
                pd.DataFrame(
                    [
                        {
                            "masked": "m",
                            "source_id": "A",
                            "target_id": "B",
                            "encoded": 1.0,
                            "type": "training",
                        },
                        {
                            "masked": "m",
                            "source_id": "B",
                            "target_id": "A",
                            "encoded": -1.0,
                            "type": "training",
                        },
                    ]
                ),
                pd.DataFrame(
                    [
                        {
                            "masked": "m",
                            "source_id": "A",
                            "target_id": "B",
                            "encoded": 0.2,
                            "type": "training",
                        },
                        {
                            "masked": "m",
                            "source_id": "B",
                            "target_id": "A",
                            "encoded": -0.2,
                            "type": "training",
                        },
                    ]
                ),
            ],
            "multi_seed": True,
        }
    }
    result = compute_anchor_aligned_encoding_matrix(
        pair_by_id={"ab": ("A", "B")},
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        aggregate="mean",
    )
    matrix = pd.DataFrame(result["matrix"], index=result["rows"], columns=result["columns"])
    assert matrix.loc["A", "A"] == pytest.approx(0.6)
    assert matrix.loc["A", "B"] == pytest.approx(-0.6)
