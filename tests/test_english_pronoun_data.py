"""Tests for English pronoun demo data completeness checks."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gradiend.examples.create_english_pronoun_data import (
    PRONOUN_CLASSES,
    _incomplete_classes_sidecar_path,
    pronoun_training_data_is_complete,
)


def _write_training_csv(path: Path, classes: list[str]) -> None:
    rows = []
    for class_id in classes:
        rows.append(
            {
                "masked": f"[MASK] {class_id}",
                "split": "train",
                "label_class": class_id,
                "label": class_id.lower(),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pronoun_training_data_is_complete_requires_all_classes():
    with tempfile.TemporaryDirectory() as tmp:
        training_path = Path(tmp) / "training.csv"
        _write_training_csv(training_path, ["1SG", "3SG", "3PL"])
        assert not pronoun_training_data_is_complete(tmp)


def test_pronoun_training_data_is_complete_rejects_incomplete_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        training_path = Path(tmp) / "training.csv"
        _write_training_csv(training_path, PRONOUN_CLASSES)
        sidecar = _incomplete_classes_sidecar_path(training_path)
        pd.DataFrame([{"label_class": "1PL"}]).to_csv(sidecar, index=False)
        assert not pronoun_training_data_is_complete(tmp)


def test_pronoun_training_data_is_complete_when_all_classes_present():
    with tempfile.TemporaryDirectory() as tmp:
        training_path = Path(tmp) / "training.csv"
        _write_training_csv(training_path, PRONOUN_CLASSES)
        assert pronoun_training_data_is_complete(tmp)


def test_build_pronoun_suite_requires_ten_trainers(monkeypatch):
    import argparse
    import sys

    with tempfile.TemporaryDirectory() as tmp:
        from experiments.multilingual_gradiend_demo import build_experiment_config, build_pronoun_suite, parse_args

        training_path = Path(tmp) / "training.csv"
        neutral_path = Path(tmp) / "neutral.csv"
        _write_training_csv(training_path, ["1SG", "3SG", "3PL"])
        pd.DataFrame([{"masked": "neutral", "split": "test"}]).to_csv(neutral_path, index=False)
        sidecar = _incomplete_classes_sidecar_path(training_path)
        pd.DataFrame(
            [
                {"masked": "[MASK] we", "split": "train", "label_class": "1PL", "label": "we"},
                {"masked": "[MASK] you", "split": "train", "label_class": "2SGPL", "label": "you"},
            ]
        ).to_csv(sidecar, index=False)

        monkeypatch.setattr(
            "experiments.multilingual_gradiend_demo._pronoun_data_paths",
            lambda: (training_path, neutral_path),
        )
        monkeypatch.setattr(sys, "argv", ["multilingual_gradiend_demo.py"])

        cli = parse_args()
        config = build_experiment_config(argparse.Namespace(**{**vars(cli), "model": None}))
        with pytest.raises(ValueError, match="expected 10 trainers"):
            build_pronoun_suite(config, retain_models_in_memory=False)
