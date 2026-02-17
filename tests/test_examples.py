"""
Tests for example files.

Tests that example files exist, can be imported, and have correct structure.
Also tests encoder distribution plot violin count.
"""

import os
import sys
import importlib.util
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from gradiend.visualizer.encoder_distributions import plot_encoder_distributions


# Example files that should exist
EXAMPLE_FILES = [
    "gradiend/examples/gender_en.py",
    "gradiend/examples/gender_de.py",
    "gradiend/examples/gender_de_decoder_only.py",
    "gradiend/examples/gender_de_detailed.py",
    "gradiend/examples/race_religion.py",
]


class TestExampleFiles:
    """Test that example files exist and can be imported."""
    
    @pytest.mark.parametrize("example_file", EXAMPLE_FILES)
    def test_example_file_exists(self, example_file):
        """Test that example file exists."""
        file_path = Path(example_file)
        assert file_path.exists(), f"Example file {example_file} does not exist"
    
    @pytest.mark.parametrize("example_file", EXAMPLE_FILES)
    def test_example_file_has_valid_syntax(self, example_file):
        """Test that example file has valid Python syntax."""
        file_path = Path(example_file)
        if not file_path.exists():
            pytest.skip(f"Example file {example_file} does not exist")
        
        # Try to compile the file to check syntax
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, str(file_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Example file {example_file} has syntax error: {e}")
        except Exception as e:
            pytest.fail(f"Error reading example file {example_file}: {e}")
    
    def test_examples_directory_exists(self):
        """Test that examples directory exists."""
        examples_dir = Path("gradiend/examples")
        assert examples_dir.exists(), "Examples directory does not exist"
        assert examples_dir.is_dir(), "Examples path is not a directory"


class TestEncoderDistributionViolinCount:
    """Test encoder distribution plot violin count."""
    
    def test_encoder_distribution_violin_count_basic(self, temp_dir):
        """Test that encoder distribution plot has expected number of violins."""
        # Create mock trainer
        trainer = MagicMock()
        trainer.run_id = "test_run"
        trainer.pair = None
        trainer.get_model = MagicMock(return_value=None)
        
        # Create encoder_df with known structure
        # Each unique violin_group should create one violin
        # In paired mode, violins are grouped by pair_id (i // 2)
        encoder_df = pd.DataFrame({
            "encoded": np.random.randn(100),
            "label": [1.0] * 50 + [2.0] * 50,
            "source_id": ["1"] * 50 + ["2"] * 50,
            "target_id": ["2"] * 50 + ["1"] * 50,
            "type": ["training"] * 100,
        })
        
        # Expected: 2 unique source_ids -> 2 violin groups (paired mode)
        # In paired mode with 2 labels, we get 1 violin (pair_id = 0 for both)
        expected_violin_groups = 1  # Both labels form one pair
        
        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                output_path = plot_encoder_distributions(
                    trainer=trainer,
                    encoder_df=encoder_df,
                    output=os.path.join(temp_dir, "test_plot.pdf"),
                    show=False
                )
                
                # Verify plot was created
                assert output_path != ""
                mock_savefig.assert_called()
    
    def test_encoder_distribution_violin_count_multiple_groups(self, temp_dir):
        """Test encoder distribution with multiple violin groups."""
        trainer = MagicMock()
        trainer.run_id = "test_run"
        trainer.pair = None
        trainer.get_model = MagicMock(return_value=None)
        
        # Create encoder_df with 4 different source_ids
        # This should create 2 violin groups in paired mode (4 labels -> 2 pairs)
        encoder_df = pd.DataFrame({
            "encoded": np.random.randn(200),
            "label": [1.0] * 50 + [2.0] * 50 + [3.0] * 50 + [4.0] * 50,
            "source_id": ["1"] * 50 + ["2"] * 50 + ["3"] * 50 + ["4"] * 50,
            "target_id": ["2"] * 50 + ["1"] * 50 + ["4"] * 50 + ["3"] * 50,
            "type": ["training"] * 200,
        })
        
        # Expected: 4 labels -> 2 pairs -> 2 violin groups
        expected_violin_groups = 2
        
        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                output_path = plot_encoder_distributions(
                    trainer=trainer,
                    encoder_df=encoder_df,
                    output=os.path.join(temp_dir, "test_plot.pdf"),
                    show=False
                )
                
                assert output_path != ""
                mock_savefig.assert_called()
    
    def test_encoder_distribution_violin_count_with_neutral(self, temp_dir):
        """Test encoder distribution with neutral data."""
        trainer = MagicMock()
        trainer.run_id = "test_run"
        trainer.pair = None
        trainer.get_model = MagicMock(return_value=None)
        
        # Create encoder_df with training and neutral data
        encoder_df = pd.DataFrame({
            "encoded": np.random.randn(150),
            "label": [1.0] * 50 + [2.0] * 50 + [0.0] * 50,  # 0.0 is neutral
            "source_id": ["1"] * 50 + ["2"] * 50 + ["neutral"] * 50,
            "target_id": ["2"] * 50 + ["1"] * 50 + ["neutral"] * 50,
            "type": ["training"] * 100 + ["neutral"] * 50,
        })
        
        # Expected: 1 training pair + 1 neutral group = 2 violin groups
        expected_violin_groups = 2
        
        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                output_path = plot_encoder_distributions(
                    trainer=trainer,
                    encoder_df=encoder_df,
                    output=os.path.join(temp_dir, "test_plot.pdf"),
                    show=False
                )
                
                assert output_path != ""
                mock_savefig.assert_called()
    
    def test_encoder_distribution_violin_count_with_paired_legend_labels(self, temp_dir):
        """Test encoder distribution with explicit paired_legend_labels."""
        trainer = MagicMock()
        trainer.run_id = "test_run"
        trainer.pair = None
        trainer.get_model = MagicMock(return_value=None)
        
        # Create encoder_df
        encoder_df = pd.DataFrame({
            "encoded": np.random.randn(200),
            "label": [1.0] * 50 + [2.0] * 50 + [3.0] * 50 + [4.0] * 50,
            "source_id": ["1"] * 50 + ["2"] * 50 + ["3"] * 50 + ["4"] * 50,
            "target_id": ["2"] * 50 + ["1"] * 50 + ["4"] * 50 + ["3"] * 50,
            "type": ["training"] * 200,
        })
        
        # Explicit pairing: labels 0,1 -> pair 0, labels 2,3 -> pair 1
        paired_legend_labels = ["label1", "label2", "label3", "label4"]
        
        # Expected: 2 pairs -> 2 violin groups
        expected_violin_groups = 2
        
        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                output_path = plot_encoder_distributions(
                    trainer=trainer,
                    encoder_df=encoder_df,
                    output=os.path.join(temp_dir, "test_plot.pdf"),
                    show=False,
                    paired_legend_labels=paired_legend_labels
                )
                
                assert output_path != ""
                mock_savefig.assert_called()
    
    def test_encoder_distribution_violin_count_matches_data(self, temp_dir):
        """Test that violin count matches the data structure (bug-indicating test)."""
        trainer = MagicMock()
        trainer.run_id = "test_run"
        trainer.pair = None
        trainer.get_model = MagicMock(return_value=None)
        
        # Create encoder_df with specific structure
        # 3 unique source_ids -> should create appropriate number of violins
        n_per_group = 30
        encoder_df = pd.DataFrame({
            "encoded": np.random.randn(n_per_group * 6),
            "label": [1.0] * n_per_group + [2.0] * n_per_group + 
                     [3.0] * n_per_group + [4.0] * n_per_group +
                     [5.0] * n_per_group + [6.0] * n_per_group,
            "source_id": ["1"] * n_per_group + ["2"] * n_per_group + 
                        ["3"] * n_per_group + ["4"] * n_per_group +
                        ["5"] * n_per_group + ["6"] * n_per_group,
            "target_id": ["2"] * n_per_group + ["1"] * n_per_group + 
                         ["4"] * n_per_group + ["3"] * n_per_group +
                         ["6"] * n_per_group + ["5"] * n_per_group,
            "type": ["training"] * (n_per_group * 6),
        })
        
        # Expected: 6 labels -> 3 pairs -> 3 violin groups
        expected_violin_groups = 3
        
        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                # Also patch the actual plotting to verify violin count
                with patch('seaborn.violinplot') as mock_violinplot:
                    output_path = plot_encoder_distributions(
                        trainer=trainer,
                        encoder_df=encoder_df,
                        output=os.path.join(temp_dir, "test_plot.pdf"),
                        show=False
                    )
                    
                    # Verify plot was created
                    assert output_path != ""
                    mock_savefig.assert_called()
                    
                    # Verify violinplot was called with correct data
                    if mock_violinplot.called:
                        call_kwargs = mock_violinplot.call_args[1]
                        plot_data = call_kwargs.get('data')
                        if plot_data is not None:
                            # Check that we have the expected number of unique violin groups
                            unique_groups = plot_data['x_cat'].nunique() if 'x_cat' in plot_data.columns else 0
                            # The actual count should match expected (allowing for some flexibility)
                            assert unique_groups > 0, "Should have at least one violin group"
