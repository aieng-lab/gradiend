import pandas as pd
import pytest
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.classification.config import TextClassificationConfig
from gradiend.trainer.text.classification.classification_head import _ClassificationDataset
from gradiend.trainer.text.classification.trainer import TextClassificationTrainer


def _make_trainer(df: pd.DataFrame) -> TextClassificationTrainer:
    config = TextClassificationConfig(
        data=df,
        target_classes=["commutative", "non-commutative"],
    )
    args = TrainingArguments(output_dir="unused")
    return TextClassificationTrainer(
        model="bert-base-uncased",
        args=args,
        config=config,
    )


class _DummyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, text_pair=None, truncation=True, max_length=512, padding="max_length", return_tensors="pt"):
        self.calls.append((text, text_pair))
        return {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


def test_ensure_data_raises_clear_error_for_globally_single_label_dataset():
    df = pd.DataFrame(
        [
            {"text": "a+b", "label": "commutative", "split": "train"},
            {"text": "b+a", "label": "commutative", "split": "validation"},
        ]
    )

    trainer = _make_trainer(df)

    with pytest.raises(ValueError) as exc_info:
        trainer._ensure_data()

    msg = str(exc_info.value)
    assert "dataset overall" in msg
    assert "validation split" not in msg
    assert "commutative" in msg


def test_ensure_data_raises_clear_error_when_split_is_missing_one_label():
    df = pd.DataFrame(
        [
            {"text": "a+b", "label": "commutative", "split": "train"},
            {"text": "x*y", "label": "non-commutative", "split": "train"},
            {"text": "b+a", "label": "commutative", "split": "validation"},
            {"text": "y*x", "label": "non-commutative", "split": "test"},
            {"text": "x*(y*z)", "label": "non-commutative", "split": "test"},
        ]
    )

    trainer = _make_trainer(df)

    with pytest.raises(ValueError) as exc_info:
        trainer._ensure_data()

    msg = str(exc_info.value)
    assert "validation split" in msg
    assert "dataset overall" not in msg
    assert "commutative" in msg


def test_ensure_data_accepts_single_supervised_label_when_semantic_classes_differ():
    df = pd.DataFrame(
        [
            {
                "text": "a+b",
                "text_alternative": "a*b",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "commutative",
                "alternative_cls": "non-commutative",
                "split": "train",
            },
            {
                "text": "a*b",
                "text_alternative": "a+b",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "non-commutative",
                "alternative_cls": "commutative",
                "split": "train",
            },
            {
                "text": "b+a",
                "text_alternative": "b*a",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "commutative",
                "alternative_cls": "non-commutative",
                "split": "validation",
            },
            {
                "text": "b*a",
                "text_alternative": "b+a",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "non-commutative",
                "alternative_cls": "commutative",
                "split": "validation",
            },
            {
                "text": "x+y",
                "text_alternative": "x*y",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "commutative",
                "alternative_cls": "non-commutative",
                "split": "test",
            },
            {
                "text": "x*y",
                "text_alternative": "x+y",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "non-commutative",
                "alternative_cls": "commutative",
                "split": "test",
            },
        ]
    )

    trainer = _make_trainer(df)
    trainer._ensure_data()

    assert set(trainer._combined_data["factual_cls"].astype(str)) == {"commutative", "non-commutative"}


def test_classification_head_data_still_uses_supervised_labels_for_case_three():
    df = pd.DataFrame(
        [
            {
                "text": "a+b",
                "text_alternative": "a*b",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "commutative",
                "alternative_cls": "non-commutative",
                "split": "train",
            },
            {
                "text": "a*b",
                "text_alternative": "a+b",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "non-commutative",
                "alternative_cls": "commutative",
                "split": "train",
            },
            {
                "text": "x+y",
                "text_alternative": "x*y",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "commutative",
                "alternative_cls": "non-commutative",
                "split": "validation",
            },
            {
                "text": "x*y",
                "text_alternative": "x+y",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "non-commutative",
                "alternative_cls": "commutative",
                "split": "validation",
            },
        ]
    )

    trainer = _make_trainer(df)

    head_df, label2id, id2label, num_labels, split_col = trainer._classification_head_data()

    assert num_labels == 2
    assert split_col == "split"
    assert set(head_df["label"].astype(str)) == {"commutative", "non-commutative"}
    assert set(label2id.keys()) == {"commutative", "non-commutative"}
    assert set(id2label.values()) == {"commutative", "non-commutative"}


def test_training_dataset_uses_semantic_classes_for_encoder_ids():
    df = pd.DataFrame(
        [
            {
                "text": "a+b",
                "text_alternative": "a*b",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "commutative",
                "alternative_cls": "non-commutative",
                "split": "train",
            },
            {
                "text": "a*b",
                "text_alternative": "a+b",
                "label": "commutative",
                "label_alternative": "commutative",
                "factual_cls": "non-commutative",
                "alternative_cls": "commutative",
                "split": "train",
            }
        ]
    )

    trainer = _make_trainer(df)
    dataset = trainer.create_training_data(_DummyTokenizer(), split="train")
    item = dataset[0]

    assert item["factual_id"] == "commutative"
    assert item["alternative_id"] == "non-commutative"
    assert item["feature_class_id"] == "commutative"
    assert item["label"] == pytest.approx(1.0)


def test_classification_head_dataset_supports_sequence_pairs():
    tokenizer = _DummyTokenizer()
    df = pd.DataFrame(
        [
            {
                "text": "fallback",
                "text_a": "a + b = c",
                "text_b": "c = a + b",
                "label": "commutative",
            }
        ]
    )

    dataset = _ClassificationDataset(
        df,
        tokenizer,
        {"commutative": 1},
        text_col="text",
        label_col="label",
        text_a_col="text_a",
        text_b_col="text_b",
        max_length=16,
    )

    item = dataset[0]

    assert tokenizer.calls == [("a + b = c", "c = a + b")]
    assert int(item["labels"].item()) == 1


def test_training_dataset_preserves_tuple_inputs():
    df = pd.DataFrame(
        [
            {
                "text": ("a + b", "b + a"),
                "text_alternative": ("a + b", "b * a"),
                "label": "commutative",
                "label_alternative": "non-commutative",
                "split": "train",
            },
            {
                "text": ("a + b", "b * a"),
                "text_alternative": ("a + b", "b + a"),
                "label": "non-commutative",
                "label_alternative": "commutative",
                "split": "train",
            },
        ]
    )

    tokenizer = _DummyTokenizer()
    trainer = _make_trainer(df)
    dataset = trainer.create_training_data(tokenizer, split="train")
    _ = dataset[0]

    assert tokenizer.calls == [("a + b", "b + a"), ("a + b", "b * a")]
