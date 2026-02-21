"""
Tests for decoder probability computation in evaluate_base_model.

Tests that probabilities are correctly computed for target classes,
with proper handling of counterfactual evaluation and multiple tokens per class.
"""

import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any

from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer, TextPredictionConfig
from gradiend.trainer.text.prediction.decoder_eval_utils import evaluate_probability_shift_score
from tests.conftest import MockTokenizer, SimpleMockModel


class TestEvaluateBaseModelProbabilityComputation:
    """Test probability computation in evaluate_base_model."""
    
    def test_evaluate_base_model_returns_probs_dict(self):
        """Test that evaluate_base_model returns probs dict with target class probabilities."""
        # Create trainer with minimal config
        config = TextPredictionConfig(
            data=pd.DataFrame([
                {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he"},
                {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they"},
            ]),
            target_classes=["3SG", "3PL"],
            decoder_eval_targets={
                "3SG": ["he", "He"],
                "3PL": ["they", "They"],
            },
            masked_col="masked",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        
        # Mock model and tokenizer
        model = SimpleMockModel()
        tokenizer = MockTokenizer()
        
        # Create training_like_df with alternative_id for counterfactual evaluation
        training_like_df = pd.DataFrame([
            {
                "masked": "[MASK] here",
                "label_class": "3SG",
                "label": "he",
                "alternative_id": "3PL",  # Counterfactual: 3SG data evaluated for 3PL
            },
            {
                "masked": "[MASK] there",
                "label_class": "3PL",
                "label": "they",
                "alternative_id": "3SG",  # Counterfactual: 3PL data evaluated for 3SG
            },
        ])
        
        neutral_df = pd.DataFrame([{"text": "neutral text"}])
        
        # Mock evaluate_probability_shift_score to return predictable results
        with patch('gradiend.trainer.text.prediction.trainer.evaluate_probability_shift_score') as mock_eval:
            mock_eval.return_value = {
                "3SG": 0.7,  # Probability of 3SG tokens on counterfactual (3PL) data
                "3PL": 0.3,  # Probability of 3PL tokens on counterfactual (3SG) data
            }
            
            with patch('gradiend.trainer.text.prediction.trainer.compute_lms') as mock_lms:
                mock_lms.return_value = {"lms": 0.5}
                
                result = trainer.evaluate_base_model(
                    model=model,
                    tokenizer=tokenizer,
                    training_like_df=training_like_df,
                    neutral_df=neutral_df,
                    use_cache=False,
                )
        
        # Verify structure
        assert "probs" in result
        assert "lms" in result
        
        # Verify probs dict contains target classes
        assert isinstance(result["probs"], dict)
        assert "3SG" in result["probs"]
        assert "3PL" in result["probs"]
        
        # Verify probabilities are floats
        assert isinstance(result["probs"]["3SG"], float)
        assert isinstance(result["probs"]["3PL"], float)
        
        # Verify LMS structure
        assert isinstance(result["lms"], dict)
    
    def test_evaluate_base_model_counterfactual_evaluation(self):
        """Test that counterfactual evaluation filters correctly when decoder_eval_prob_on_other_class=True."""
        config = TextPredictionConfig(
            data=pd.DataFrame([
                {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he"},
                {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they"},
            ]),
            target_classes=["3SG", "3PL"],
            decoder_eval_targets={
                "3SG": ["he"],
                "3PL": ["they"],
            },
            decoder_eval_prob_on_other_class=True,  # Enable counterfactual evaluation
            masked_col="masked",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        
        model = SimpleMockModel()
        tokenizer = MockTokenizer()
        
        # Create training_like_df with alternative_id
        training_like_df = pd.DataFrame([
            {
                "masked": "[MASK] here",
                "label_class": "3SG",
                "alternative_id": "3PL",  # This row belongs to 3SG dataset
            },
            {
                "masked": "[MASK] there",
                "label_class": "3PL",
                "alternative_id": "3SG",  # This row belongs to 3PL dataset
            },
        ])
        
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        
        # Track calls to evaluate_probability_shift_score
        with patch('gradiend.trainer.text.prediction.trainer.evaluate_probability_shift_score') as mock_eval:
            mock_eval.return_value = {"3SG": 0.6, "3PL": 0.4}
            
            with patch('gradiend.trainer.text.prediction.trainer.compute_lms') as mock_lms:
                mock_lms.return_value = {"lms": 0.5}
                
                result = trainer.evaluate_base_model(
                    model=model,
                    tokenizer=tokenizer,
                    training_like_df=training_like_df,
                    neutral_df=neutral_df,
                    use_cache=False,
                )
            
            # Verify evaluate_probability_shift_score was called with data_class_col
            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["data_class_col"] == "alternative_id"  # Should use alternative_id for counterfactual
    
    def test_evaluate_base_model_multiple_tokens_per_class(self):
        """Test that multiple tokens per class are correctly summed."""
        config = TextPredictionConfig(
            data=pd.DataFrame([
                {"masked": "[MASK] here", "split": "train", "label_class": "masc_nom", "label": "der"},
                {"masked": "[MASK] there", "split": "train", "label_class": "fem_nom", "label": "die"},
            ]),
            target_classes=["masc_nom", "fem_nom"],
            decoder_eval_targets={
                "masc_nom": ["der", "Der"],  # Multiple tokens per class
                "fem_nom": ["die", "Die"],   # Multiple tokens per class
            },
            masked_col="masked",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        
        model = SimpleMockModel()
        tokenizer = MockTokenizer()
        
        training_like_df = pd.DataFrame([
            {"masked": "[MASK] here", "label_class": "masc_nom", "alternative_id": "fem_nom"},
            {"masked": "[MASK] there", "label_class": "fem_nom", "alternative_id": "masc_nom"},
        ])
        
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        
        # Mock evaluate_probability_shift_score - it should sum probabilities for multiple tokens
        with patch('gradiend.trainer.text.prediction.trainer.evaluate_probability_shift_score') as mock_eval:
            # The function should return summed probabilities (P("der") + P("Der") for masc_nom)
            mock_eval.return_value = {
                "masc_nom": 0.8,  # Sum of P("der") + P("Der")
                "fem_nom": 0.2,   # Sum of P("die") + P("Die")
            }
            
            with patch('gradiend.trainer.text.prediction.trainer.compute_lms') as mock_lms:
                mock_lms.return_value = {"lms": 0.5}
                
                result = trainer.evaluate_base_model(
                    model=model,
                    tokenizer=tokenizer,
                    training_like_df=training_like_df,
                    neutral_df=neutral_df,
                    use_cache=False,
                )
            
            # Verify that evaluate_probability_shift_score received targets with multiple tokens
            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args[1]
            targets = call_kwargs["targets"]
            assert "masc_nom" in targets
            assert "fem_nom" in targets
            assert len(targets["masc_nom"]) == 2  # Should have 2 tokens
            assert len(targets["fem_nom"]) == 2   # Should have 2 tokens
            
            # Verify result contains summed probabilities
            assert result["probs"]["masc_nom"] == 0.8
            assert result["probs"]["fem_nom"] == 0.2
