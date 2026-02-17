"""
Verified test: German GPT-2 for gender DE with MLM head.

Trains decoder-only MLM head then GRADIEND on gender_de (masc_nom vs fem_nom);
verifies file existence and correlation >= threshold.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from gradiend import TextPredictionTrainer, TrainingArguments

from .verify_utils import (
    BENCH_MLM_HEAD_CONFIG,
    BENCH_TRAIN_CONFIG_WITH_PRUNING,
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
def test_german_gpt2_mlm_gender_de_verified(temp_output_dir):
    """German GPT-2 + MLM head for gender DE; verify files and correlation."""
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
        run_id="gender_de_masc_nom_fem_nom",
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

    # 2) Train GRADIEND (resolve_model_path will use MLM head); use pruning
    gradiend_output = os.path.join(temp_output_dir, "models", trainer.run_id or "run", "german-gpt2")
    trainer.train(output=gradiend_output, **BENCH_TRAIN_CONFIG_WITH_PRUNING)
    result_path = trainer.model_path

    assert result_path is not None
    assert os.path.isdir(result_path)
    assert_model_files_exist(result_path)
    score = assert_correlation_threshold(result_path, min_correlation=0.5, tolerance=0.15)
    print(f"  german-gpt2 MLM gender_de: score={score:.4f}")
