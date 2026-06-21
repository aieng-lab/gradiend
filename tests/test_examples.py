"""
Tests for example files.

Tests that example files exist, can be imported, and have correct structure.
Also tests encoder distribution plot violin count.
"""

import os
import importlib.util
import ast
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from gradiend.visualizer.encoder_distributions import plot_encoder_distributions


EXAMPLE_FILES = sorted(
    str(path).replace("\\", "/")
    for path in Path("gradiend/examples").glob("*.py")
    if path.name != "__init__.py"
)


def _violin_group_count(trainer, encoder_df, output_path, **plot_kwargs) -> int:
    """Run plot_encoder_distributions and return the number of violin x-axis groups."""
    pytest.importorskip("seaborn")
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch(
        "seaborn.violinplot"
    ) as mock_violinplot:
        plot_encoder_distributions(
            trainer=trainer,
            encoder_df=encoder_df,
            output=output_path,
            show=False,
            **plot_kwargs,
        )
        assert mock_violinplot.called, "Expected seaborn.violinplot to be called"
        plot_data = mock_violinplot.call_args.kwargs["data"]
        return int(plot_data["x_cat"].nunique())

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

    @pytest.mark.parametrize("example_file", EXAMPLE_FILES)
    def test_example_file_imports_without_running_workflow(self, example_file):
        """Example modules should import cleanly without starting expensive workflows."""
        module_name = f"_gradiend_example_smoke_{Path(example_file).stem}"
        spec = importlib.util.spec_from_file_location(module_name, example_file)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        old_module = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module
    
    def test_examples_directory_exists(self):
        """Test that examples directory exists."""
        examples_dir = Path("gradiend/examples")
        assert examples_dir.exists(), "Examples directory does not exist"
        assert examples_dir.is_dir(), "Examples path is not a directory"

    @pytest.mark.parametrize("example_file", EXAMPLE_FILES)
    def test_training_arguments_fail_on_non_convergence(self, example_file):
        """Executable training examples must fail loudly when GRADIEND does not converge."""
        tree = ast.parse(Path(example_file).read_text(encoding="utf-8"), filename=example_file)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "TrainingArguments")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "TrainingArguments")
            )
        ]
        for call in calls:
            value = None
            for kw in call.keywords:
                if kw.arg == "fail_on_non_convergence":
                    value = kw.value
                    break
            assert isinstance(value, ast.Constant) and value.value is True, (
                f"{example_file}: TrainingArguments must set fail_on_non_convergence=True"
            )


class TestEncoderDistributionViolinCount:
    """Test encoder distribution plot violin count."""

    def test_encoder_distribution_violin_count_basic(self, temp_dir):
        """Test that encoder distribution plot has expected number of violins."""
        pytest.importorskip("matplotlib")
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

        group_count = _violin_group_count(
            trainer,
            encoder_df,
            os.path.join(temp_dir, "test_plot.pdf"),
        )
        assert group_count == expected_violin_groups

    def test_encoder_distribution_class_label_mapping_updates_transition_labels(self, temp_dir):
        """Individual class labels can be mapped before transition legend labels are built."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        trainer = MagicMock()
        trainer.run_id = "test_run"
        trainer.pair = ("NM", "NF")
        trainer.get_model = MagicMock(return_value=None)
        trainer._id2label = {}
        trainer.config = None

        encoder_df = pd.DataFrame({
            "encoded": np.random.randn(40),
            "label": [1.0] * 20 + [-1.0] * 20,
            "source_id": ["NM"] * 20 + ["NF"] * 20,
            "target_id": ["NF"] * 20 + ["NM"] * 20,
            "type": ["training"] * 40,
        })

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.pyplot.close"):
                with patch("matplotlib.pyplot.savefig"):
                    output_path = plot_encoder_distributions(
                        trainer=trainer,
                        encoder_df=encoder_df,
                        output=os.path.join(temp_dir, "mapped_labels.pdf"),
                        class_label_mapping={"NM": "Masc. Nom.", "NF": "Fem. Nom."},
                        show=False,
                    )

                    ax = plt.gcf().axes[0]
                    legend = ax.get_legend()
                    labels = [txt.get_text() for txt in legend.get_texts()]

        assert output_path != ""
        assert "Masc. Nom. -> Fem. Nom." in labels
        assert "Fem. Nom. -> Masc. Nom." in labels


class TestEncoderDistributionViolinVariants:
    def test_encoder_distribution_violin_count_multiple_groups(self, temp_dir):
        """Test encoder distribution with multiple violin groups."""
        pytest.importorskip("matplotlib")
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

        group_count = _violin_group_count(
            trainer,
            encoder_df,
            os.path.join(temp_dir, "test_plot.pdf"),
        )
        assert group_count == expected_violin_groups

    def test_encoder_distribution_violin_count_with_neutral(self, temp_dir):
        """Test encoder distribution with neutral data."""
        pytest.importorskip("matplotlib")
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
            "type": ["training"] * 100 + ["neutral_dataset"] * 50,
        })
        
        # Expected: 1 training pair + 1 neutral x-axis group
        expected_violin_groups = 2

        group_count = _violin_group_count(
            trainer,
            encoder_df,
            os.path.join(temp_dir, "test_plot.pdf"),
        )
        assert group_count == expected_violin_groups
    
    def test_encoder_distribution_violin_count_with_paired_legend_labels(self, temp_dir):
        """Test encoder distribution with explicit paired_legend_labels."""
        pytest.importorskip("matplotlib")
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
        
        # Legend labels must match the transitions present in encoder_df.
        paired_legend_labels = ["1 -> 2", "2 -> 1", "3 -> 4", "4 -> 3"]
        
        # Expected: 2 pairs -> 2 violin groups
        expected_violin_groups = 2

        group_count = _violin_group_count(
            trainer,
            encoder_df,
            os.path.join(temp_dir, "test_plot.pdf"),
            paired_legend_labels=paired_legend_labels,
        )
        assert group_count == expected_violin_groups
    
    def test_encoder_distribution_violin_count_matches_data(self, temp_dir):
        """Test that violin count matches the data structure (bug-indicating test)."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
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

        group_count = _violin_group_count(
            trainer,
            encoder_df,
            os.path.join(temp_dir, "test_plot.pdf"),
        )
        assert group_count == expected_violin_groups
