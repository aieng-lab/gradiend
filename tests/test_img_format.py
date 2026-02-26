"""Tests for img_format: config default, visualizer output path, and trainer forwarding."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.visualizer.convergence import plot_training_convergence, _steps_and_values
from gradiend.visualizer.encoder_distributions import plot_encoder_distributions


class TestImgFormatConfig:
    """TextPredictionConfig img_format default and storage."""

    def test_config_default_img_format_is_pdf(self):
        config = TextPredictionConfig(data=pd.DataFrame(), target_classes=["A", "B"])
        assert config.img_format == "pdf"

    def test_config_stores_img_format(self):
        config = TextPredictionConfig(
            data=pd.DataFrame(),
            target_classes=["A", "B"],
            img_format="png",
        )
        assert config.img_format == "png"


class TestImgFormatVisualizerOutputPath:
    """Visualizers use img_format to set the output file extension."""

    def test_plot_training_convergence_output_path_uses_img_format(self, tmp_path):
        pytest.importorskip("matplotlib")
        training_stats = {
            "training_stats": {
                "mean_by_class": {0: {"0": 0.1, "1": -0.1}},
                "scores": {0: 0.5},
            },
            "best_score_checkpoint": {},
        }
        out_base = tmp_path / "convergence"
        out_base.mkdir()
        output_with_pdf = str(out_base / "plot.pdf")
        with patch("matplotlib.pyplot.show"):
            path = plot_training_convergence(
                training_stats=training_stats,
                output=output_with_pdf,
                img_format="png",
                show=False,
            )
        assert path.endswith(".png")
        assert (out_base / "plot.png").exists()

    def test_plot_encoder_distributions_output_path_uses_img_format(self, tmp_path):
        pytest.importorskip("matplotlib")
        trainer = MagicMock()
        trainer.run_id = "run1"
        trainer.pair = None
        trainer.get_model = MagicMock(return_value=None)
        encoder_df = pd.DataFrame({
            "encoded": [0.1, -0.2, 0.1, -0.2],
            "label": [1.0, -1.0, 1.0, -1.0],
            "source_id": ["1", "2", "1", "2"],
            "target_id": ["2", "1", "2", "1"],
            "type": ["training"] * 4,
        })
        output_with_pdf = str(tmp_path / "encoder.pdf")
        with patch("matplotlib.pyplot.show"):
            path = plot_encoder_distributions(
                trainer=trainer,
                encoder_df=encoder_df,
                output=output_with_pdf,
                img_format="png",
                show=False,
            )
        assert path.endswith(".png")
        assert (tmp_path / "encoder.png").exists()


class TestImgFormatTrainerForwarding:
    """TextPredictionTrainer forwards img_format to plot methods."""

    def test_plot_encoder_distributions_receives_img_format_from_trainer(self, tmp_path):
        pytest.importorskip("matplotlib")
        # Patch where the evaluator's visualizer calls through (visualizer holds _plot_encoder_distributions)
        with patch(
            "gradiend.visualizer.visualizer._plot_encoder_distributions",
            wraps=plot_encoder_distributions,
        ) as mock_plot:
            config = TextPredictionConfig(
                data=pd.DataFrame({
                    "masked": ["[MASK] here"],
                    "split": ["train"],
                    "label_class": ["3SG"],
                    "label": ["he"],
                }),
                target_classes=["3SG", "3PL"],
                img_format="png",
            )
            args = TrainingArguments(experiment_dir=str(tmp_path))
            trainer = TextPredictionTrainer(model="bert-base-uncased", config=config, args=args)
            trainer.run_id = "test_run"
            # Avoid loading the real model (plot_encoder_distributions calls trainer.get_model())
            trainer.get_model = MagicMock(return_value=None)
            # pair is derived from config.target_classes (read-only); already ("3SG", "3PL")
            # Use source_id/target_id that match trainer.pair (3SG, 3PL) so target_and_neutral_only keeps them
            encoder_df = pd.DataFrame({
                "encoded": [0.1, -0.2],
                "label": [1.0, -1.0],
                "source_id": ["3SG", "3PL"],
                "target_id": ["3PL", "3SG"],
                "type": ["training", "training"],
            })
            with patch("matplotlib.pyplot.show"):
                trainer.plot_encoder_distributions(
                    encoder_df=encoder_df,
                    output=os.path.join(tmp_path, "enc.pdf"),
                    show=False,
                )
            mock_plot.assert_called_once()
            call_kwargs = mock_plot.call_args[1]
            assert call_kwargs.get("img_format") == "png"

    def test_plot_training_convergence_receives_img_format_from_trainer(self, tmp_path):
        pytest.importorskip("matplotlib")
        # Patch where the evaluator's visualizer calls through
        with patch(
            "gradiend.visualizer.visualizer._plot_training_convergence",
            wraps=plot_training_convergence,
        ) as mock_plot:
            config = TextPredictionConfig(
                data=pd.DataFrame({
                    "masked": ["[MASK] here"],
                    "split": ["train"],
                    "label_class": ["3SG"],
                    "label": ["he"],
                }),
                target_classes=["3SG", "3PL"],
                img_format="svg",
            )
            args = TrainingArguments(experiment_dir=str(tmp_path))
            trainer = TextPredictionTrainer(model="bert-base-uncased", config=config, args=args)
            trainer.get_model = MagicMock(return_value=None)
            training_stats = {
                "training_stats": {"mean_by_class": {0: {"0": 0.1}}, "scores": {0: 0.5}},
                "best_score_checkpoint": {},
            }
            with patch("matplotlib.pyplot.show"):
                trainer.plot_training_convergence(
                    training_stats=training_stats,
                    output=os.path.join(tmp_path, "conv.pdf"),
                    show=False,
                )
            mock_plot.assert_called_once()
            call_kwargs = mock_plot.call_args[1]
            assert call_kwargs.get("img_format") == "svg"


class TestConvergencePlotAutoSave:
    """Training convergence plot is auto-saved when experiment_dir is set."""

    def test_explicit_plot_training_convergence_saves_when_experiment_dir_set(self, tmp_path):
        """trainer.plot_training_convergence() saves to experiment_dir when experiment_dir is set."""
        pytest.importorskip("matplotlib")
        config = TextPredictionConfig(
            data=pd.DataFrame({
                "masked": ["[MASK] here"],
                "split": ["train"],
                "label_class": ["3SG"],
                "label": ["he"],
            }),
            target_classes=["3SG", "3PL"],
        )
        args = TrainingArguments(experiment_dir=str(tmp_path))
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config, args=args)
        trainer.get_model = MagicMock(return_value=None)
        trainer.get_training_stats = MagicMock(
            return_value={
                "training_stats": {"mean_by_class": {0: {"0": 0.1}}, "scores": {0: 0.5}},
                "best_score_checkpoint": {},
            }
        )
        with patch("matplotlib.pyplot.show"):
            path = trainer.plot_training_convergence(show=False)
        assert path
        assert path.endswith(".pdf")
        assert os.path.exists(path)
        assert "training_convergence.pdf" in path

    def test_plot_automatically_called_after_training_when_experiment_dir_set(self, tmp_path):
        """train() automatically saves convergence plot when experiment_dir is set."""
        pytest.importorskip("matplotlib")
        config = TextPredictionConfig(
            data=pd.DataFrame({
                "masked": ["[MASK] here"],
                "split": ["train"],
                "label_class": ["3SG"],
                "label": ["he"],
            }),
            target_classes=["3SG", "3PL"],
        )
        args = TrainingArguments(experiment_dir=str(tmp_path), max_seeds=1)
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config, args=args)

        output_dir = str(tmp_path / "model")
        os.makedirs(output_dir, exist_ok=True)
        # Write minimal training.json so plot has data to render
        import json
        with open(os.path.join(output_dir, "training.json"), "w") as f:
            json.dump({
                "training_stats": {"mean_by_class": {0: {"0": 0.1}}, "scores": {0: 0.5}},
                "best_score_checkpoint": {},
            }, f)

        with patch.object(trainer, "_train", return_value=output_dir):
            with patch("matplotlib.pyplot.show"):
                trainer.train(use_cache=False)

        plot_path = tmp_path / "training_convergence.pdf"
        assert plot_path.exists(), f"Expected convergence plot at {plot_path}"

    def test_convergence_plot_includes_identity_classes(self):
        """Convergence plot must include identity classes (label 0) and identity feature classes."""
        training_stats = {
            "mean_by_class": {
                0: {"0": 0.02, "1": 0.1, "-1": -0.1},
            },
            "mean_by_feature_class": {
                0: {"masc_nom": 0.1, "fem_nom": -0.1, "neut_nom": 0.02},
            },
            "scores": {0: 0.5},
        }
        run_info = {"training_stats": training_stats, "best_score_checkpoint": {}}
        steps, series_by_class, series_by_fc = _steps_and_values(
            run_info["training_stats"], mean_by_class=True, mean_by_feature_class=True
        )
        assert "0" in series_by_class, "mean_by_class must include label 0 (identity) in convergence plot"
        assert "neut_nom" in series_by_fc, "mean_by_feature_class must include identity class neut_nom"
