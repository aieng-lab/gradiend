"""``mask_placeholder``: the prediction-slot marker of the dataset schema.

The marker inside masked templates is independent of ``tokenizer.mask_token`` (a causal LM has none,
an MLM has its own special). Data that marks the slot differently (``[PRONOUN]``) is declared via
``TextPredictionConfig.mask_placeholder`` / ``TrainingArguments.mask_placeholder``; the tokenizer's
special is substituted only at tokenization time.
"""

import pandas as pd
import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.dataset import (
    DEFAULT_DATASET_MASK_PLACEHOLDER,
    TextTrainingDataset,
    create_masked_pair_from_text,
    resolve_mask_placeholder,
    validate_masked_templates_in_dataframe,
)
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.testing_mocks import MockTokenizer


def _frame(masked, factual="token_1", alternative="token_2"):
    return pd.DataFrame({
        "masked": [masked],
        "factual": [factual],
        "alternative": [alternative],
        "factual_class": ["class1"],
        "alternative_class": ["class2"],
        "factual_id": [1],
        "alternative_id": [2],
        "label": ["positive"],
        "feature_class_id": [1],
        "feature_pole": ["pos"],
    })


class TestTrainingArguments:
    def test_default_and_serialization(self):
        assert TrainingArguments().mask_placeholder == "[MASK]"
        args = TrainingArguments(mask_placeholder="[PRONOUN]")
        assert TrainingArguments.from_dict(args.to_dict()).mask_placeholder == "[PRONOUN]"

    def test_validation(self):
        with pytest.raises(TypeError, match="mask_placeholder"):
            TrainingArguments(mask_placeholder=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mask_placeholder"):
            TrainingArguments(mask_placeholder="")


class TestResolve:
    def test_non_default_config_wins_over_default_args(self):
        config = TextPredictionConfig(mask_placeholder="[PRONOUN]")
        assert resolve_mask_placeholder(config=config, training_args=TrainingArguments()) == "[PRONOUN]"

    def test_non_default_args_win_over_default_config(self):
        args = TrainingArguments(mask_placeholder="[PRONOUN]")
        assert resolve_mask_placeholder(config=TextPredictionConfig(), training_args=args) == "[PRONOUN]"

    def test_conflicting_non_defaults_raise(self):
        with pytest.raises(ValueError, match="Conflicting mask_placeholder"):
            resolve_mask_placeholder(
                config=TextPredictionConfig(mask_placeholder="[A]"),
                training_args=TrainingArguments(mask_placeholder="[B]"),
            )

    def test_defaults(self):
        assert resolve_mask_placeholder() == DEFAULT_DATASET_MASK_PLACEHOLDER


class TestTrainingDataset:
    def test_custom_placeholder_is_normalized_to_the_tokenizer_mask_token(self):
        """A dataset using ``[PRONOUN]`` must tokenize exactly like the same data using ``[MASK]``."""
        items = {}
        for placeholder in ("[MASK]", "[PRONOUN]"):
            dataset = TextTrainingDataset(
                data=_frame(f"{placeholder} said that {placeholder} was late"),
                tokenizer=MockTokenizer(),
                batch_size=1,
                is_decoder_only_model=False,
                mask_placeholder=placeholder,
            )
            items[placeholder] = dataset[0]
        custom, default = items["[PRONOUN]"], items["[MASK]"]
        assert custom["input_text"] == "[MASK] said that [MASK] was late"
        assert custom["text"] == "token_1 said that token_1 was late"
        for side in ("factual", "alternative"):
            for key in ("input_ids", "attention_mask", "labels"):
                assert custom[side][key].tolist() == default[side][key].tolist(), (side, key)
        assert (custom["factual"]["labels"] != -100).sum().item() == 2

    def test_custom_placeholder_for_decoder_only_cuts_the_prefix(self):
        dataset = TextTrainingDataset(
            data=_frame("token_5 token_6 [PRONOUN] token_7", factual="token_9", alternative="token_8"),
            tokenizer=MockTokenizer(),
            batch_size=1,
            is_decoder_only_model=True,
            mask_placeholder="[PRONOUN]",
        )
        item = dataset[0]
        assert item["input_text"] == "token_5 token_6 "
        assert item["template"] == "token_5 token_6 [PRONOUN] token_7"

    def test_template_without_the_placeholder_fails_with_an_actionable_message(self):
        with pytest.raises(ValueError, match="TextPredictionConfig.mask_placeholder"):
            TextTrainingDataset(
                data=_frame("The person [PRONOUN] runs", factual="token_9", alternative="token_8"),
                tokenizer=MockTokenizer(),
                batch_size=1,
                is_decoder_only_model=True,
                mask_placeholder="[MASK]",
            )

    def test_template_carrying_the_tokenizer_mask_token_is_still_accepted(self):
        tokenizer = MockTokenizer()
        assert tokenizer.mask_token
        validate_masked_templates_in_dataframe(
            _frame(f"{tokenizer.mask_token} said"), "[PRONOUN]", mask_token=tokenizer.mask_token
        )
        with pytest.raises(ValueError, match="mask_placeholder='\\[PRONOUN\\]'"):
            validate_masked_templates_in_dataframe(_frame("no slot here"), "[PRONOUN]")


class _Tok:
    """Minimal whitespace tokenizer, optionally owning an MLM mask special."""

    def __init__(self, mask_token=None):
        self.mask_token = mask_token

    def tokenize(self, text):
        return str(text).split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


TEXT = "the museum is still closed today but it reopens on monday for everyone"


class TestMaskedPairBuilder:
    """Masked templates carry the dataset placeholder, never a tokenizer special."""

    def test_mlm_branch_writes_the_dataset_placeholder(self):
        masked, target = create_masked_pair_from_text(
            TEXT, _Tok(mask_token="<mask>"), is_decoder_only_model=False, mask_token="<mask>"
        )
        assert DEFAULT_DATASET_MASK_PLACEHOLDER in masked
        assert "<mask>" not in masked, "tokenizer special leaked into the template"
        assert target in TEXT.split() and target not in masked.split()

    def test_mlm_output_passes_the_schema_validator(self):
        rows = []
        for _ in range(8):
            pair = create_masked_pair_from_text(
                TEXT, _Tok(mask_token="<mask>"), is_decoder_only_model=False, mask_token="<mask>"
            )
            if pair:
                rows.append(pair[0])
        assert rows
        validate_masked_templates_in_dataframe(pd.DataFrame({"masked": rows}), DEFAULT_DATASET_MASK_PLACEHOLDER)

    def test_decoder_only_branch_is_unchanged(self):
        masked, _ = create_masked_pair_from_text(TEXT, _Tok(), is_decoder_only_model=True)
        assert masked.endswith(DEFAULT_DATASET_MASK_PLACEHOLDER)

    @pytest.mark.parametrize("decoder_only", [True, False])
    def test_both_branches_honour_an_explicit_placeholder(self, decoder_only):
        pair = create_masked_pair_from_text(
            TEXT, _Tok(mask_token="<mask>"), is_decoder_only_model=decoder_only,
            mask_token="<mask>", mask_placeholder="[PRONOUN]",
        )
        assert pair is not None
        assert "[PRONOUN]" in pair[0]
        assert "<mask>" not in pair[0]


class TestTrainerWiring:
    @staticmethod
    def _df():
        return pd.DataFrame([
            {"masked": "[PRONOUN] is here", "split": "train", "label_class": "3SG", "label": "he"},
            {"masked": "[PRONOUN] are here", "split": "train", "label_class": "3PL", "label": "they"},
        ])

    def test_training_args_placeholder_reaches_the_training_dataset(self):
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=TextPredictionConfig(data=self._df(), target_classes=["3SG", "3PL"]),
            args=TrainingArguments(mask_placeholder="[PRONOUN]"),
        )
        training_data = trainer.create_training_data(MockTokenizer(), split="train", batch_size=1)
        item = training_data[0]
        assert training_data.mask_placeholder == "[PRONOUN]"
        assert item["input_text"].startswith("[MASK]")
        assert item["text"].startswith(("he", "they"))

    def test_config_placeholder_reaches_the_training_dataset(self):
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=TextPredictionConfig(
                data=self._df(), target_classes=["3SG", "3PL"], mask_placeholder="[PRONOUN]"
            ),
        )
        training_data = trainer.create_training_data(MockTokenizer(), split="train", batch_size=1)
        assert training_data.mask_placeholder == "[PRONOUN]"
        assert training_data[0]["input_text"].startswith("[MASK]")
