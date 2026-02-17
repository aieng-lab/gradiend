"""
Verified test: GPT-2 on race setup.

Runs race (white/black) with gpt2; more iterations than examples;
verifies file existence and correlation >= threshold.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from gradiend import TextPredictionTrainer

from .verify_utils import (
    BENCH_TRAIN_CONFIG,
    assert_correlation_threshold,
    assert_model_files_exist,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_output_dir():
    d = tempfile.mkdtemp()
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.integration
def test_gpt2_race_verified(temp_output_dir):
    """GPT-2 on race (white/black); verify files and correlation."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    model_name = "gpt2"
    pair = ("white", "black")
    other_classes = ["asian"]
    classes = list(pair) + other_classes
    trainer = TextPredictionTrainer(
        model=model_name,
        run_id=f"race_{pair[0]}_{pair[1]}",
        data="aieng-lab/gradiend_race_data",
        classes=classes,
        pair=pair,
        masked_col="masked",
        add_identity_for_other_classes=False,
        eval_neutral_data="aieng-lab/biasneutral",
    )
    output = os.path.join(temp_output_dir, trainer.run_id or "run", model_name)

    trainer.train(output=output, **BENCH_TRAIN_CONFIG)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    score = assert_correlation_threshold(result_path, min_correlation=0.5, tolerance=0.15)
    print(f"  gpt2 race {pair[0]}/{pair[1]}: score={score:.4f}")
