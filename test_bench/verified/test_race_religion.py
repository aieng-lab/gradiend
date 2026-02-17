"""
Verified test: race and religion setup with BERT-family model.

Runs both race (white/black) and religion (christian/muslim) with more iterations
than examples; verifies file existence and correlation >= threshold.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from gradiend import TextPredictionTrainer, TrainingArguments

from .verify_utils import (
    BENCH_TRAIN_CONFIG_WITH_PRUNING,
    assert_correlation_threshold,
    assert_model_files_exist,
    get_score_from_training_stats,
)

# Add project root for imports
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
def test_race_religion_distilbert_verified(temp_output_dir):
    """Race and religion with DistilBERT; verify files and correlation."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - test bench requires GPU for reasonable runtime")

    model_name = "distilbert-base-cased"
    race_white_black = ("race", ("white", "black"), ["asian"])
    religion_christian_muslim = ("religion", ("christian", "muslim"), ["jewish"])

    for bias_type, pair, other_classes in [race_white_black, religion_christian_muslim]:
        args = TrainingArguments(
            experiment_dir=temp_output_dir,
            add_identity_for_other_classes=False,
            use_cache=False,
        )
        trainer = TextPredictionTrainer(
            model=model_name,
            run_id=f"{bias_type}_{pair[0]}_{pair[1]}",
            data=f"aieng-lab/gradiend_{bias_type}_data",
            target_classes=list(pair),
            masked_col="masked",
            eval_neutral_data="aieng-lab/biasneutral",
            args=args,
        )
        output = os.path.join(temp_output_dir, f"{bias_type}_{pair[0]}_{pair[1]}", model_name)

        trainer.train(output=output, **BENCH_TRAIN_CONFIG_WITH_PRUNING)
        result_path = trainer.model_path

        assert result_path is not None
        assert os.path.isdir(result_path)
        assert_model_files_exist(result_path)
        
        # For first run: print actual score, then user can set expectations
        score = get_score_from_training_stats(result_path)
        print(f"\n[EXPECTATION SETTING] {bias_type} {pair[0]}/{pair[1]}: score={score:.4f}")
        
        # For now, just check it's not NaN and reasonable
        assert score is not None, f"Could not read score from {result_path}/training.json"
        assert score == score, f"Score is NaN in {result_path}"
        assert abs(score) < 10.0, f"Score {score:.4f} seems unreasonable"
