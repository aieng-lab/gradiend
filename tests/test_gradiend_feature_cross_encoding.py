import pandas as pd
import pytest

import gradiend.comparison.feature_cross_encoding as feature_cross_encoding_module
from gradiend.comparison.feature_cross_encoding import (
    collect_unified_test_rows_by_feature_class,
    compute_gradiend_feature_cross_encoding_matrix,
    _mean_encoded_for_feature_class,
)
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
    UNIFIED_TRANSITION,
)


def _unified_row(
    *,
    masked: str,
    factual: str,
    alternative: str,
    factual_class: str,
    alternative_class: str,
    split: str = "test",
) -> dict:
    return {
        UNIFIED_MASKED: masked,
        UNIFIED_FACTUAL: factual,
        UNIFIED_ALTERNATIVE: alternative,
        UNIFIED_FACTUAL_CLASS: factual_class,
        UNIFIED_ALTERNATIVE_CLASS: alternative_class,
        UNIFIED_TRANSITION: f"{factual_class}->{alternative_class}",
        UNIFIED_SPLIT: split,
    }


class _Trainer:
    def __init__(self, trainer_id: str, target_classes, rows):
        self.run_id = trainer_id
        self.target_classes = list(target_classes)
        self.combined_data = pd.DataFrame(rows)

    def _ensure_data(self):
        return None

    def create_gradient_training_dataset(self, pairs, model):
        return pairs

    def evaluate_encoder(self, **kwargs):
        raise AssertionError("should not be called")


class _TinyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 103
    pad_token_id = 0
    model_max_length = 32

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=True,
        max_length=None,
        padding=None,
    ):
        if return_tensors == "pt":
            import torch

            ids = [101, self.mask_token_id if self.mask_token in str(text) else 200, 102]
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": [201]}

    def convert_tokens_to_ids(self, token):
        return self.mask_token_id if token == self.mask_token else 201


class _TextProbeTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_dataset_calls = []

    def create_gradient_training_dataset(self, pairs, model):
        self.gradient_dataset_calls.append((pairs, model))
        return [{"source": "gradient", "label": pairs[0]["label"], "factual_id": pairs[0]["factual_id"]}]


class _TextProbeModel:
    tokenizer = _TinyTokenizer()
    is_decoder_only_model = False
    is_seq2seq_model = False


def test_collect_unified_test_rows_by_feature_class_merges_across_trainers():
    trainers = {
        "a": _Trainer(
            "a",
            ["F", "M"],
            [
                _unified_row(
                    masked="[MASK] she",
                    factual="she",
                    alternative="he",
                    factual_class="F",
                    alternative_class="M",
                ),
            ],
        ),
        "b": _Trainer(
            "b",
            ["positive", "negative"],
            [
                _unified_row(
                    masked="[MASK] good",
                    factual="good",
                    alternative="bad",
                    factual_class="positive",
                    alternative_class="negative",
                ),
                _unified_row(
                    masked="[MASK] she",
                    factual="she",
                    alternative="he",
                    factual_class="F",
                    alternative_class="M",
                ),
            ],
        ),
    }
    by_class = collect_unified_test_rows_by_feature_class(trainers, split="test")
    assert set(by_class) == {"F", "positive"}
    assert len(by_class["F"]) == 1
    assert len(by_class["positive"]) == 1


def test_cross_encoding_probe_pairs_are_delegated_to_trainer_before_encoding(monkeypatch):
    trainer = _TextProbeTrainer(
        "gender",
        ["F", "M"],
        [],
    )
    class_df = pd.DataFrame([
        _unified_row(
            masked="[MASK] runs",
            factual="she",
            alternative="he",
            factual_class="F",
            alternative_class="M",
        )
    ])

    def _capture_encoded_rows(model, dataset):
        item = dataset[0]
        assert item["source"] == "gradient"
        assert item["factual_id"] == "F"
        return [{"encoded": 0.75}]

    monkeypatch.setattr(
        feature_cross_encoding_module,
        "encode_dataset_to_rows",
        _capture_encoded_rows,
    )

    model = _TextProbeModel()
    assert _mean_encoded_for_feature_class(
        trainer,
        model,
        class_df,
    ) == pytest.approx(0.75)
    pairs, captured_model = trainer.gradient_dataset_calls[0]
    assert captured_model is model


def test_compute_gradiend_feature_cross_encoding_matrix_is_dense(monkeypatch):
    trainers = {
        "gender": _Trainer(
            "gender",
            ["F", "M"],
            [
                _unified_row(
                    masked="[MASK] she",
                    factual="she",
                    alternative="he",
                    factual_class="F",
                    alternative_class="M",
                ),
                _unified_row(
                    masked="[MASK] he",
                    factual="he",
                    alternative="she",
                    factual_class="M",
                    alternative_class="F",
                ),
            ],
        ),
        "sentiment": _Trainer(
            "sentiment",
            ["positive", "negative"],
            [
                _unified_row(
                    masked="[MASK] good",
                    factual="good",
                    alternative="bad",
                    factual_class="positive",
                    alternative_class="negative",
                ),
                _unified_row(
                    masked="[MASK] bad",
                    factual="bad",
                    alternative="good",
                    factual_class="negative",
                    alternative_class="positive",
                ),
            ],
        ),
    }
    encoded_by_trainer_class = {
        ("gender", "F"): 0.8,
        ("gender", "M"): -0.8,
        ("gender", "positive"): 0.1,
        ("gender", "negative"): -0.1,
        ("sentiment", "F"): 0.2,
        ("sentiment", "M"): -0.2,
        ("sentiment", "positive"): 0.9,
        ("sentiment", "negative"): -0.9,
    }

    def _fake_mean(trainer, model, class_df, *, max_size=None):
        factual_class = str(class_df.iloc[0][UNIFIED_FACTUAL_CLASS])
        trainer_id = trainer.run_id
        return encoded_by_trainer_class[(trainer_id, factual_class)]

    monkeypatch.setattr(
        feature_cross_encoding_module,
        "_load_eval_model_for_trainer",
        lambda trainer, load_directory=None: object(),
    )
    monkeypatch.setattr(
        feature_cross_encoding_module,
        "_mean_encoded_for_feature_class",
        _fake_mean,
    )

    result = compute_gradiend_feature_cross_encoding_matrix(
        trainers,
        ["F", "M", "positive", "negative"],
        trainer_order=["gender", "sentiment"],
    )

    assert result["model_ids"] == ["gender", "sentiment"]
    assert result["column_ids"] == ["F", "M", "positive", "negative"]
    assert result["matrix"][0] == pytest.approx([0.8, -0.8, 0.1, -0.1])
    assert result["matrix"][1] == pytest.approx([0.2, -0.2, 0.9, -0.9])
    assert all(count > 0 for row in result["n_matrix"] for count in row)
