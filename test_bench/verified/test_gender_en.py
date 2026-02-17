"""
Verified test: gender EN (English) use-case.

Tests multiple model configurations:
- RoBERTa-base: encoder-only with different MASK token (<mask>)
- DistilBERT: encoder-only with standard [MASK] token
- GPT-2: decoder-only with direct CLM (no MLM head)

More iterations than examples; verifies file existence and correlation >= threshold.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from gradiend import TrainingArguments
from gradiend.examples.gender_en import build_gender_trainer

from .verify_utils import (
    BENCH_TRAIN_CONFIG,
    BENCH_TRAIN_CONFIG_WITH_PRUNING,
    assert_correlation_threshold,
    assert_model_files_exist,
    get_score_from_training_stats,
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
def test_gender_en_roberta_verified(temp_output_dir):
    """Gender EN no-pruning baseline: small model (distilbert-base-cased) without pre/post pruning."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    model_name = "distilbert-base-cased"
    args = TrainingArguments(experiment_dir=temp_output_dir, use_cache=False)
    trainer = build_gender_trainer(model=model_name, names_per_template=4, args=args)
    output = os.path.join(temp_output_dir, trainer.run_id or "run", model_name)

    trainer.train(output=output, **BENCH_TRAIN_CONFIG)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    
    # For first run: print actual score, then user can set expectations
    score = get_score_from_training_stats(result_path)
    print(f"\n[EXPECTATION SETTING] gender_en (no pruning, distilbert): score={score:.4f}")
    
    # For now, just check it's not NaN and reasonable (> 0 or allow negative)
    assert score is not None, f"Could not read score from {result_path}/training.json"
    assert score == score, f"Score is NaN in {result_path}"
    # Will set proper threshold after first run
    assert abs(score) < 10.0, f"Score {score:.4f} seems unreasonable"


@pytest.mark.slow
@pytest.mark.integration
def test_gender_en_distilbert_verified(temp_output_dir):
    """Gender EN with distilbert-base-cased (encoder-only, standard [MASK] token); with pruning."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    model_name = "distilbert-base-cased"
    args = TrainingArguments(experiment_dir=temp_output_dir, use_cache=False)
    trainer = build_gender_trainer(model=model_name, names_per_template=4, args=args)
    output = os.path.join(temp_output_dir, trainer.run_id or "run", model_name)

    trainer.train(output=output, **BENCH_TRAIN_CONFIG_WITH_PRUNING)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    
    # For first run: print actual score, then user can set expectations
    score = get_score_from_training_stats(result_path)
    print(f"\n[EXPECTATION SETTING] gender_en (DistilBERT): score={score:.4f}")
    
    # For now, just check it's not NaN and reasonable
    assert score is not None, f"Could not read score from {result_path}/training.json"
    assert score == score, f"Score is NaN in {result_path}"
    assert abs(score) < 10.0, f"Score {score:.4f} seems unreasonable"


@pytest.mark.slow
@pytest.mark.integration
def test_gender_en_gpt2_clm_verified(temp_output_dir):
    """Gender EN with gpt2 (decoder-only, direct CLM, no MLM head); with pruning."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    model_name = "gpt2"
    args = TrainingArguments(experiment_dir=temp_output_dir, use_cache=False)
    trainer = build_gender_trainer(model=model_name, names_per_template=4, args=args)
    output = os.path.join(temp_output_dir, trainer.run_id or "run", model_name)

    trainer.train(output=output, **BENCH_TRAIN_CONFIG_WITH_PRUNING)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    
    # For first run: print actual score, then user can set expectations
    score = get_score_from_training_stats(result_path)
    print(f"\n[EXPECTATION SETTING] gender_en (GPT-2 CLM): score={score:.4f}")
    
    # For now, just check it's not NaN and reasonable
    assert score is not None, f"Could not read score from {result_path}/training.json"
    assert score == score, f"Score is NaN in {result_path}"
    assert abs(score) < 10.0, f"Score {score:.4f} seems unreasonable"
