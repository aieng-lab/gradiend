import json
from unittest.mock import MagicMock, patch

import pandas as pd

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.classification.config import TextClassificationConfig
from gradiend.trainer.text.classification.trainer import TextClassificationTrainer
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer


def test_annotate_data_prediction_writes_csv_and_json(tmp_path):
    args = TrainingArguments(experiment_dir=str(tmp_path))
    config = TextPredictionConfig(target_classes=["male", "female"])
    trainer = TextPredictionTrainer(
        model="bert-base-uncased",
        config=config,
        args=args,
    )
    base_df = pd.DataFrame(
        {
            "masked": ["[MASK] is here", "[MASK] is there"],
            "factual": ["he", "she"],
            "alternative": ["she", "he"],
            "factual_id": ["male", "female"],
            "alternative_id": ["female", "male"],
        }
    )
    annotated_df = base_df.copy()
    annotated_df["base_p_target_he"] = [0.7, 0.2]
    annotated_df["base_p_target_she"] = [0.1, 0.8]
    annotated_df["base_p_class_male"] = [0.7, 0.2]
    annotated_df["base_p_class_female"] = [0.1, 0.8]
    token_map = {"he": "he", "she": "she"}
    token_columns = {"he": "base_p_target_he", "she": "base_p_target_she"}
    class_columns = {
        "male": "base_p_class_male",
        "female": "base_p_class_female",
    }

    fake_model = MagicMock()
    fake_model.tokenizer = MagicMock()
    fake_model.base_model = fake_model

    with patch.object(trainer, "_load_base_annotation_model", return_value=fake_model), \
         patch.object(trainer, "_get_decoder_eval_dataframe", return_value=(base_df, base_df.copy())), \
         patch.object(trainer, "_get_decoder_eval_targets", return_value={"male": ["he"], "female": ["she"]}), \
         patch.object(
             trainer,
             "_annotate_text_prediction_rows",
             return_value=(annotated_df, token_map, token_columns, class_columns),
         ):
        result = trainer.annotate_data()

    assert (tmp_path / "annotated_data.csv").exists()
    assert (tmp_path / "annotated_data.json").exists()
    assert result["summary"]["mean_p_target"] == {"he": 0.45, "she": 0.45}
    assert result["summary"]["mean_p_target_factual"] == {"he": 0.7, "she": 0.8}
    assert result["summary"]["mean_p_target_alternative"] == {"he": 0.2, "she": 0.1}
    assert result["summary"]["mean_p_class_factual"] == {"male": 0.7, "female": 0.8}
    assert result["summary"]["mean_p_class_alternative"] == {"male": 0.2, "female": 0.1}

    with open(tmp_path / "annotated_data.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["target_token_map"] == token_map


def test_annotate_data_classification_uses_shared_api(tmp_path):
    args = TrainingArguments(experiment_dir=str(tmp_path))
    config = TextClassificationConfig(target_classes=["pos", "neg"])
    trainer = TextClassificationTrainer(
        model="distilbert-base-uncased",
        config=config,
        args=args,
        data=pd.DataFrame({"text": ["x", "y"], "label": ["pos", "neg"]}),
    )
    base_df = pd.DataFrame(
        {
            "text": ["good", "bad"],
            "label_class": ["pos", "neg"],
            "factual_id": ["pos", "neg"],
            "alternative_id": ["neg", "pos"],
        }
    )
    annotated_df = base_df.copy()
    annotated_df["base_p_target_pos"] = [0.9, 0.2]
    annotated_df["base_p_target_neg"] = [0.1, 0.8]
    annotated_df["base_p_class_pos"] = [0.9, 0.2]
    annotated_df["base_p_class_neg"] = [0.1, 0.8]
    token_map = {"pos": "pos", "neg": "neg"}
    token_columns = {"pos": "base_p_target_pos", "neg": "base_p_target_neg"}
    class_columns = {"pos": "base_p_class_pos", "neg": "base_p_class_neg"}

    fake_model = MagicMock()
    fake_model.tokenizer = MagicMock()
    fake_model.base_model = fake_model

    with patch.object(trainer, "_load_base_annotation_model", return_value=fake_model), \
         patch.object(trainer, "_get_decoder_eval_dataframe", return_value=(base_df, base_df.copy())), \
         patch.object(trainer, "_get_decoder_eval_targets", return_value={"pos": ["pos"], "neg": ["neg"]}), \
         patch.object(
             trainer,
             "_annotate_text_classification_rows",
             return_value=(annotated_df, token_map, token_columns, class_columns),
         ):
        result = trainer.annotate_data()

    assert result["summary"]["mean_p_target"] == {"pos": 0.55, "neg": 0.45}
    assert result["summary"]["mean_p_target_factual"] == {"pos": 0.9, "neg": 0.8}
    assert result["summary"]["mean_p_target_alternative"] == {"pos": 0.2, "neg": 0.1}
