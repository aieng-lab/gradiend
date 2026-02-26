"""
Tests for decoder probability shifts plotting (plot_probability_shifts).

Covers the tricky parts: counterfactual selection (P(other) on target_class dataset),
vertical line at selected LR, x-axis clamped to data range, and plot structure.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from gradiend.visualizer.probability_shifts import plot_probability_shifts


def _make_plotting_data(lrs=(1e-5, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0), ff=-1.0):
    """Build minimal grid for plot_probability_shifts."""
    grid = {
        "base": {
            "probs_by_dataset": {
                "3SG": {"3SG": 0.8, "3PL": 0.2},
                "3PL": {"3SG": 0.2, "3PL": 0.8},
            },
            "lms": {"lms": 0.5},
        },
    }
    for lr in lrs:
        # Simulate: higher lr pushes 3PL more on 3PL data, 3SG more on 3SG data
        grid[(ff, lr)] = {
            "id": {"feature_factor": ff, "learning_rate": lr},
            "probs_by_dataset": {
                "3SG": {"3SG": 0.9 - lr * 0.0001, "3PL": 0.1 + lr * 0.0001},
                "3PL": {"3SG": 0.1 + lr * 0.0005, "3PL": 0.9 - lr * 0.0005},
            },
            "lms": {"lms": max(0.01, 0.5 - lr * 0.0001)},
        }
    return {"plotting_data": grid, "grid": grid}


def _make_decoder_results(selected_lr=0.1, selected_ff=-1.0):
    """Build decoder_results in flat format (class keys at top level) for target_class 3PL."""
    return {
        "3SG": {"learning_rate": selected_lr, "feature_factor": selected_ff, "value": 0.5},
        "3PL": {"learning_rate": selected_lr, "feature_factor": selected_ff, "value": 0.5},
        "grid": {},
    }


class TestPlotProbabilityShifts:
    """Test plot_probability_shifts behavior."""

    def test_plot_runs_with_minimal_mock_data(self, tmp_path):
        """Plot runs without error and returns output path when output is set."""
        pytest.importorskip("matplotlib")
        plotting_data = _make_plotting_data()
        decoder_results = _make_decoder_results()
        output = str(tmp_path / "shifts.pdf")
        with patch("matplotlib.pyplot.show"):
            path = plot_probability_shifts(
                decoder_results=decoder_results,
                plotting_data=plotting_data,
                class_ids=["3SG", "3PL"],
                target_class="3PL",
                output=output,
                show=False,
            )
        assert path == output
        assert os.path.exists(output)

    def test_counterfactual_uses_p_other_on_metric_dataset(self, tmp_path):
        """
        Selection star is drawn on P(other_class) on target_class dataset.
        For target_class=3PL: use P(3SG) on 3PL (orange line in Dataset 3PL subplot).
        """
        pytest.importorskip("matplotlib")
        # Distinct probs so we can verify which value is used for the star
        grid = _make_plotting_data()["plotting_data"]
        grid[(1.0, 0.1)] = {
            "id": {"feature_factor": 1.0, "learning_rate": 0.1},
            "probs_by_dataset": {
                "3SG": {"3SG": 0.9, "3PL": 0.1},
                "3PL": {"3SG": 0.4, "3PL": 0.6},  # P(3SG) on 3PL = 0.4 (counterfactual)
            },
            "lms": {"lms": 0.5},
        }
        plotting_data = {"plotting_data": grid}
        decoder_results = _make_decoder_results(selected_lr=0.1)

        scatter_calls = []

        def track_scatter(*args, **kwargs):
            scatter_calls.append((args, kwargs))

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.axes.Axes.scatter", side_effect=track_scatter):
                plot_probability_shifts(
                    decoder_results=decoder_results,
                    plotting_data=plotting_data,
                    class_ids=["3SG", "3PL"],
                    target_class="3PL",
                    output=str(tmp_path / "out.pdf"),
                    show=False,
                )

        # Should have scatter for the selection star (marker="*")
        star_scatters = [c for c in scatter_calls if len(c[0]) >= 2 and "marker" in c[1] and c[1]["marker"] == "*"]
        assert len(star_scatters) >= 1, "Expected at least one star scatter for selected config"

    def test_vertical_line_at_selected_lr(self, tmp_path):
        """Vertical line (axvline) is drawn at selected learning rate on all subplots."""
        pytest.importorskip("matplotlib")
        plotting_data = _make_plotting_data()
        selected_lr = 10.0
        decoder_results = _make_decoder_results(selected_lr=selected_lr)

        axvline_calls = []

        def track_axvline(x, **kwargs):
            axvline_calls.append(x)

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.axes.Axes.axvline", side_effect=track_axvline):
                plot_probability_shifts(
                    decoder_results=decoder_results,
                    plotting_data=plotting_data,
                    class_ids=["3SG", "3PL"],
                    target_class="3PL",
                    output=str(tmp_path / "out.pdf"),
                    show=False,
                )

        assert len(axvline_calls) >= 1, "Expected axvline for selected LR"
        assert all(x == selected_lr for x in axvline_calls), "axvline should be at selected_lr"

    def test_xaxis_limited_to_data_range(self, tmp_path):
        """X-axis right limit is at most max_lr * 1.5, not 1e7."""
        pytest.importorskip("matplotlib")
        lrs = (1e-5, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)  # max = 1000
        plotting_data = _make_plotting_data(lrs=lrs)
        decoder_results = _make_decoder_results()

        set_xlim_calls = []

        def track_set_xlim(*args, **kwargs):
            set_xlim_calls.append({"args": args, "kwargs": kwargs})

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.axes.Axes.set_xlim", side_effect=track_set_xlim):
                plot_probability_shifts(
                    decoder_results=decoder_results,
                    plotting_data=plotting_data,
                    class_ids=["3SG", "3PL"],
                    target_class="3PL",
                    output=str(tmp_path / "out.pdf"),
                    show=False,
                )

        max_lr = max(lrs)
        for call in set_xlim_calls:
            kwargs = call["kwargs"]
            if "right" in kwargs:
                assert kwargs["right"] <= max_lr * 2, "xlim right should not extend far beyond max_lr"
                assert kwargs["right"] < 1e6, "xlim right should not be 1e7"

    def test_plot_subplot_count(self, tmp_path):
        """Plot has LMS + one subplot per dataset class (no separate selection-metrics subplot)."""
        pytest.importorskip("matplotlib")
        plotting_data = _make_plotting_data()
        decoder_results = _make_decoder_results()

        subplot_calls = []

        real_subplots = __import__("matplotlib").pyplot.subplots

        def track_subplots(nrows, *args, **kwargs):
            subplot_calls.append(nrows)
            return real_subplots(nrows, *args, **kwargs)

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.pyplot.subplots", side_effect=track_subplots):
                plot_probability_shifts(
                    decoder_results=decoder_results,
                    plotting_data=plotting_data,
                    class_ids=["3SG", "3PL"],
                    target_class="3PL",
                    output=str(tmp_path / "out.pdf"),
                    show=False,
                )

        # n_subplots = 1 + len(dataset_classes) = 1 + 2 = 3 (LMS + 3SG + 3PL)
        assert subplot_calls, "subplots should be called"
        assert subplot_calls[0] == 3, "Expected 3 subplots (LMS + 2 dataset classes)"

    def test_evaluate_base_model_counterfactual_p_other_on_metric_dataset(self):
        """Trainer produces probs[class] = P(other) on target_class dataset for selection."""
        pytest.importorskip("matplotlib")
        from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer, TextPredictionConfig

        config = TextPredictionConfig(
            data=__import__("pandas").DataFrame([
                {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he"},
                {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they"},
            ]),
            target_classes=["3SG", "3PL"],
            decoder_eval_targets={"3SG": ["he"], "3PL": ["they"]},
            decoder_eval_prob_on_other_class=True,
            masked_col="masked",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()

        # probs_by_dataset[3PL][3SG] = P(3SG) on 3PL data
        # So probs["3PL"] should equal that (counterfactual for metric 3PL)
        with patch("gradiend.trainer.text.prediction.trainer.evaluate_probability_shift_score") as mock_eval:
            mock_eval.return_value = {
                "3SG": {"3SG": 0.8, "3PL": 0.2},
                "3PL": {"3SG": 0.3, "3PL": 0.7},
            }
            with patch("gradiend.trainer.text.prediction.trainer.compute_lms", return_value={"lms": 0.5}):
                from tests.conftest import SimpleMockModel, MockTokenizer
                result = trainer.evaluate_base_model(
                    model=SimpleMockModel(),
                    tokenizer=MockTokenizer(),
                    training_like_df=__import__("pandas").DataFrame([
                        {"masked": "x", "label_class": "3SG", "alternative_id": "3PL"},
                        {"masked": "y", "label_class": "3PL", "alternative_id": "3SG"},
                    ]),
                    neutral_df=__import__("pandas").DataFrame([{"text": "z"}]),
                    use_cache=False,
                )

        # probs["3PL"] = P(3SG) on 3PL = 0.3
        assert result["probs"]["3PL"] == 0.3
        # probs["3SG"] = P(3PL) on 3SG = 0.2
        assert result["probs"]["3SG"] == 0.2
