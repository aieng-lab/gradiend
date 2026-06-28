"""Tests for comparison seed policy helpers."""

import os
import shutil
from unittest.mock import patch

import pytest

from gradiend.comparison.seed_policy import (
    enter_analysis_mode,
    models_for_comparison,
    wants_multi_seed_analysis,
)
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.multi_seed import MultiSeedTrainerView, is_multi_seed_view, resolve_default_seed_selection
from gradiend.trainer.core.seed_models import SeedModelGroup
from tests.test_multi_seed_view import _local_temp
from tests.test_trainer_model import MockTrainerForTest


def test_enter_analysis_mode_wraps_when_stability_enabled():
    temp_dir = _local_temp("enter_analysis_mode")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        wrapped = enter_analysis_mode(trainer)
        assert is_multi_seed_view(wrapped)
        assert wants_multi_seed_analysis(wrapped)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_enter_analysis_mode_noop_without_stability():
    temp_dir = _local_temp("enter_analysis_mode_noop")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir),
        )
        assert enter_analysis_mode(trainer) is trainer
        assert not wants_multi_seed_analysis(trainer)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_multi_seed_view_multi_seed_returns_self():
    temp_dir = _local_temp("multi_seed_idempotent")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        view = trainer.multi_seed()
        assert view.multi_seed() is view
        assert isinstance(view, MultiSeedTrainerView)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_default_seed_selection_on_view_returns_selection():
    temp_dir = _local_temp("resolve_on_view")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        view = trainer.multi_seed(selection="all_convergent")
        assert resolve_default_seed_selection(view, None) == "all_convergent"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_enter_analysis_mode_force_wraps_without_stability_flag():
    temp_dir = _local_temp("enter_analysis_force")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir),
        )
        wrapped = enter_analysis_mode(trainer, force=True)
        assert is_multi_seed_view(wrapped)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_models_for_comparison_best_returns_single_model():
    temp_dir = _local_temp("models_for_comparison_best")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir),
        )
        mock_model = object()
        with patch.object(trainer, "get_model", return_value=mock_model):
            model_value, _, _ = models_for_comparison(trainer, seed_selection="best")

        assert model_value is mock_model
        assert not isinstance(model_value, SeedModelGroup)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
