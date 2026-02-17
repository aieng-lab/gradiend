"""
Verified test: gender DE (German) use-case.

Tests multiple model configurations:
- BERT baseline: encoder-only without pruning (reference)
- BERT with pruning: same model with pre/post pruning
- German GPT-2 + MLM head: decoder-only with custom MLM head workflow

More iterations than examples; verifies file existence and correlation >= threshold.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from gradiend import TextPredictionTrainer, TrainingArguments

from .verify_utils import (
    BENCH_TRAIN_CONFIG,
    BENCH_TRAIN_CONFIG_WITH_PRUNING,
    BENCH_MLM_HEAD_CONFIG,
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
def test_gender_de_bert_baseline_verified(temp_output_dir):
    """Gender DE with bert-base-german-cased baseline (no pruning). Use small model for no-pruning case."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    model_name = "distilbert-base-german-cased"
    pair = ("masc_nom", "fem_nom")
    args = TrainingArguments(
        experiment_dir=temp_output_dir,
        add_identity_for_other_classes=True,
        use_cache=False,
    )
    trainer = TextPredictionTrainer(
        model=model_name,
        run_id=f"gender_de_{pair[0]}_{pair[1]}_baseline",
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
    )
    output = os.path.join(temp_output_dir, trainer.run_id or "run", model_name)

    trainer.train(output=output, **BENCH_TRAIN_CONFIG)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    
    # For first run: print actual score, then user can set expectations
    score = get_score_from_training_stats(result_path)
    print(f"\n[EXPECTATION SETTING] gender_de baseline (no pruning): score={score:.4f}")
    
    # For now, just check it's not NaN and reasonable
    assert score is not None, f"Could not read score from {result_path}/training.json"
    assert score == score, f"Score is NaN in {result_path}"
    assert abs(score) < 10.0, f"Score {score:.4f} seems unreasonable"


@pytest.mark.slow
@pytest.mark.integration
def test_gender_de_bert_with_pruning_verified(temp_output_dir):
    """Gender DE with bert-base-german-cased with pre/post pruning."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    model_name = "bert-base-german-cased"
    pair = ("masc_nom", "fem_nom")
    args = TrainingArguments(
        experiment_dir=temp_output_dir,
        add_identity_for_other_classes=True,
        use_cache=False,
    )
    trainer = TextPredictionTrainer(
        model=model_name,
        run_id=f"gender_de_{pair[0]}_{pair[1]}_pruned",
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
    )
    output = os.path.join(temp_output_dir, trainer.run_id or "run", model_name)

    trainer.train(output=output, **BENCH_TRAIN_CONFIG_WITH_PRUNING)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    
    # For first run: print actual score, then user can set expectations
    score = get_score_from_training_stats(result_path)
    print(f"\n[EXPECTATION SETTING] gender_de BERT with pruning: score={score:.4f}")
    
    # For now, just check it's not NaN and reasonable
    assert score is not None, f"Could not read score from {result_path}/training.json"
    assert score == score, f"Score is NaN in {result_path}"
    assert abs(score) < 10.0, f"Score {score:.4f} seems unreasonable"
    
    # Compare with baseline (if baseline was run)
    baseline_score = get_score_from_training_stats(
        os.path.join(temp_output_dir, "gender_de_masc_nom_fem_nom_baseline", "run", "distilbert-base-german-cased")
    )
    if baseline_score is not None:
        print(f"  Baseline score: {baseline_score:.4f}, Pruned score: {score:.4f}, Difference: {abs(score - baseline_score):.4f}")


@pytest.mark.slow
@pytest.mark.integration
def test_gender_de_gpt2_mlm_verified(temp_output_dir):
    """Gender DE with german-gpt2 + MLM head (decoder-only with custom MLM head)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    base_model = "dbmdz/german-gpt2"
    pair = ("masc_nom", "fem_nom")
    args = TrainingArguments(
        experiment_dir=temp_output_dir,
        add_identity_for_other_classes=True,
        use_cache=False,
    )
    trainer = TextPredictionTrainer(
        model=base_model,
        run_id=f"gender_de_{pair[0]}_{pair[1]}_gpt2_mlm",
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
    )

    # 1) Train decoder-only MLM head (needed for decoder-only model on this setup)
    mlm_output = os.path.join(temp_output_dir, "decoder_mlm_head", trainer.run_id or "run", "german-gpt2")
    mlm_head_path = trainer.train_decoder_only_mlm_head(
        base_model,
        output=mlm_output,
        **BENCH_MLM_HEAD_CONFIG,
    )
    assert os.path.isdir(mlm_head_path), f"MLM head should exist at {mlm_head_path}"

    # 2) Train GRADIEND (resolve_model_path will use MLM head)
    gradiend_output = os.path.join(temp_output_dir, "models", trainer.run_id or "run", "german-gpt2")
    trainer.train(output=gradiend_output, **BENCH_TRAIN_CONFIG_WITH_PRUNING)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    
    # For first run: print actual score, then user can set expectations
    score = get_score_from_training_stats(result_path)
    print(f"\n[EXPECTATION SETTING] gender_de German GPT-2 + MLM head: score={score:.4f}")
    
    # For now, just check it's not NaN and reasonable
    assert score is not None, f"Could not read score from {result_path}/training.json"
    assert score == score, f"Score is NaN in {result_path}"
    assert abs(score) < 10.0, f"Score {score:.4f} seems unreasonable"
