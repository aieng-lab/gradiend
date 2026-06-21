"""Tests for neutral encoder metadata and multi-split plot framing."""

from __future__ import annotations

import pandas as pd
import pytest

from gradiend.visualizer.encoder_neutral import (
    NEUTRAL_TYPE_VIOLIN_LABELS,
    append_multi_split_neutral_frames,
    build_multi_split_encoder_plot_frame,
    encoder_plot_xlabel,
    neutral_encoder_row_metadata,
    neutral_hue_label,
)


class TestNeutralEncoderMetadata:
    def test_row_metadata_per_type(self):
        masked = neutral_encoder_row_metadata("neutral_training_masked")
        dataset = neutral_encoder_row_metadata("neutral_dataset")
        assert masked == {
            "data_split": "test",
            "neutral_variant": "neutral_training_masked",
        }
        assert dataset == {
            "data_split": "test",
            "neutral_variant": "neutral_dataset",
        }
        assert masked["neutral_variant"] != dataset["neutral_variant"]

    def test_hue_labels_differ_by_type(self):
        assert neutral_hue_label("neutral_training_masked") == "test — training masked"
        assert neutral_hue_label("neutral_dataset") == "test — dataset"
        assert neutral_hue_label("neutral_training_masked") != neutral_hue_label("neutral_dataset")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown neutral encoder type"):
            neutral_encoder_row_metadata("neutral_other")


class TestMultiSplitNeutralPlotFrame:
    @staticmethod
    def _sample_encoder_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "encoded": [0.5, -0.5, 0.1, 0.0, 0.05],
                "label": [1.0, -1.0, 1.0, 0.0, 0.0],
                "source_id": ["white", "black", "white", "neutral", "neutral"],
                "target_id": ["black", "white", "black", "neutral", "neutral"],
                "type": [
                    "training",
                    "training",
                    "training",
                    "neutral_dataset",
                    "neutral_training_masked",
                ],
                "data_split": ["train", "test", "validation", "test", "test"],
                "neutral_variant": [None, None, None, "neutral_dataset", "neutral_training_masked"],
            }
        )

    def test_append_neutral_frames_keeps_distinct_violin_groups(self):
        frames, groups, hue_labels = append_multi_split_neutral_frames(self._sample_encoder_df())
        assert len(frames) == 2
        assert groups == [
            NEUTRAL_TYPE_VIOLIN_LABELS["neutral_training_masked"],
            NEUTRAL_TYPE_VIOLIN_LABELS["neutral_dataset"],
        ]
        assert len(set(hue_labels)) == 2

    def test_build_plot_frame_excludes_neutral_by_default(self):
        encoder_df = self._sample_encoder_df()
        df_train = encoder_df[encoder_df["type"] == "training"].copy()
        df_train["violin_group"] = df_train["source_id"]

        df_plot, group_order, facet_splits, dodge_hues, includes_neutral = (
            build_multi_split_encoder_plot_frame(df_train, encoder_df)
        )

        assert includes_neutral is False
        assert group_order == ["black", "white"]
        assert facet_splits == ["train", "validation", "test"]
        assert dodge_hues == ["train", "validation", "test"]
        assert set(df_plot["type"].unique()) == {"training"}

    def test_build_plot_frame_assigns_distinct_neutral_axes_and_hues(self):
        encoder_df = self._sample_encoder_df()
        df_train = encoder_df[encoder_df["type"] == "training"].copy()
        df_train["violin_group"] = df_train["source_id"]

        df_plot, group_order, facet_splits, dodge_hues, includes_neutral = (
            build_multi_split_encoder_plot_frame(
                df_train,
                encoder_df,
                include_neutral=True,
            )
        )

        assert includes_neutral is True
        assert group_order == [
            "black",
            "white",
            NEUTRAL_TYPE_VIOLIN_LABELS["neutral_training_masked"],
            NEUTRAL_TYPE_VIOLIN_LABELS["neutral_dataset"],
        ]
        assert facet_splits == ["train", "validation", "test"]
        assert "test — training masked" in dodge_hues
        assert "test — dataset" in dodge_hues

        test_panel = df_plot[df_plot["data_split"] == "test"]
        neutral_groups = set(test_panel["violin_group"].astype(str).unique())
        assert NEUTRAL_TYPE_VIOLIN_LABELS["neutral_training_masked"] in neutral_groups
        assert NEUTRAL_TYPE_VIOLIN_LABELS["neutral_dataset"] in neutral_groups
        assert test_panel["neutral_variant"].tolist().count("neutral_training_masked") == 1
        assert test_panel["neutral_variant"].tolist().count("neutral_dataset") == 1

    def test_single_neutral_type_omits_missing_variant(self):
        encoder_df = self._sample_encoder_df()
        encoder_df = encoder_df[encoder_df["type"] != "neutral_training_masked"]
        df_train = encoder_df[encoder_df["type"] == "training"].copy()
        df_train["violin_group"] = df_train["source_id"]

        _, group_order, _, dodge_hues, _ = build_multi_split_encoder_plot_frame(
            df_train,
            encoder_df,
            include_neutral=True,
        )
        assert group_order == ["black", "white", NEUTRAL_TYPE_VIOLIN_LABELS["neutral_dataset"]]
        assert "test — dataset" in dodge_hues
        assert "test — training masked" not in dodge_hues

    def test_encoder_plot_xlabel(self):
        assert encoder_plot_xlabel(includes_neutral_groups=False) == "Feature class"
        assert encoder_plot_xlabel(includes_neutral_groups=True) == "Target"


class TestMultiSplitEncoderPlots:
    def test_facet_plot_puts_each_neutral_type_in_test_panel(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        from unittest.mock import MagicMock, patch

        import matplotlib.pyplot as plt
        import seaborn as sns

        from gradiend.visualizer.encoder_distributions import plot_encoder_distributions

        encoder_df = pd.DataFrame(
            {
                "encoded": [0.5, -0.5, 0.1, 0.0, 0.05],
                "label": [1.0, -1.0, 1.0, 0.0, 0.0],
                "source_id": ["white", "black", "white", "neutral", "neutral"],
                "target_id": ["black", "white", "black", "neutral", "neutral"],
                "type": [
                    "training",
                    "training",
                    "training",
                    "neutral_dataset",
                    "neutral_training_masked",
                ],
                "data_split": ["train", "test", "validation", "test", "test"],
                "neutral_variant": [None, None, None, "neutral_dataset", "neutral_training_masked"],
            }
        )
        trainer = MagicMock()
        trainer.run_id = "race_white_black"
        trainer.pair = ("white", "black")
        trainer.get_model = MagicMock(return_value=None)
        trainer._id2label = {}
        trainer.config = None
        trainer.experiment_dir = None

        facet_panels: list[pd.DataFrame] = []
        original_violin = sns.violinplot

        def _record_violin(*args, **kwargs):
            data = kwargs.get("data")
            if data is not None:
                facet_panels.append(data.copy())
            return original_violin(*args, **kwargs)

        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch("seaborn.violinplot", side_effect=_record_violin):
            plot_encoder_distributions(
                trainer=trainer,
                encoder_df=encoder_df,
                split_plot_mode="facet",
                include_neutral=True,
                show=False,
                title=False,
                output="facet_test.png",
            )
        plt.close("all")

        test_panel = next(panel for panel in facet_panels if (panel["data_split"] == "test").any())
        groups = set(test_panel["violin_group"].astype(str).unique())
        assert NEUTRAL_TYPE_VIOLIN_LABELS["neutral_training_masked"] in groups
        assert NEUTRAL_TYPE_VIOLIN_LABELS["neutral_dataset"] in groups

    def test_removed_overlay_modes_use_facet_panels(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        from unittest.mock import MagicMock, patch

        import matplotlib.pyplot as plt
        import seaborn as sns

        from gradiend.visualizer.encoder_distributions import plot_encoder_distributions

        encoder_df = pd.DataFrame(
            {
                "encoded": [0.5, -0.5, 0.0, 0.05],
                "label": [1.0, -1.0, 0.0, 0.0],
                "source_id": ["white", "black", "neutral", "neutral"],
                "target_id": ["black", "white", "neutral", "neutral"],
                "type": ["training", "training", "neutral_dataset", "neutral_training_masked"],
                "data_split": ["train", "test", "test", "test"],
                "neutral_variant": [None, None, "neutral_dataset", "neutral_training_masked"],
            }
        )
        trainer = MagicMock()
        trainer.run_id = "race_white_black"
        trainer.pair = ("white", "black")
        trainer.get_model = MagicMock(return_value=None)
        trainer._id2label = {}
        trainer.config = None
        trainer.experiment_dir = None

        facet_panels: list[pd.DataFrame] = []
        original_violin = sns.violinplot

        def _record_violin(*args, **kwargs):
            data = kwargs.get("data")
            if data is not None:
                facet_panels.append(data.copy())
            return original_violin(*args, **kwargs)

        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch("seaborn.violinplot", side_effect=_record_violin):
            plot_encoder_distributions(
                trainer=trainer,
                encoder_df=encoder_df,
                split_plot_mode="dodge",
                include_neutral=True,
                show=False,
                title=False,
                output="dodge_test.png",
            )
        plt.close("all")

        assert facet_panels
        test_panel = next(panel for panel in facet_panels if (panel["data_split"] == "test").any())
        groups = set(test_panel["violin_group"].astype(str).unique())
        assert NEUTRAL_TYPE_VIOLIN_LABELS["neutral_training_masked"] in groups
        assert NEUTRAL_TYPE_VIOLIN_LABELS["neutral_dataset"] in groups
