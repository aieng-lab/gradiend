"""One label-token rule for decoder-only items, versioned so old artifacts keep theirs.

The dataset used ``vocab[label]`` (``she``, the mid-word token) while
``create_inputs`` used ``tokenizer(" " + label)`` (``▁she``). The label sits on the
last prefix token and is predicted from the token before it, so the leading-space
variant is the natural one. ``label_token_protocol`` selects the rule per run:
``first_differing_token`` (new runs) or ``no_leading_space`` (the rule of everything
trained with gradiend 0.2.5 or earlier).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.clm_alignment import label_token_id, label_token_ids
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
from tests.testing_mocks import MockTokenizer


class SpaceVariantTokenizer(MockTokenizer):
    """Whitespace tokenizer that, like BPE/SentencePiece, has a distinct id for ' word'."""

    def __init__(self):
        super().__init__()
        self.vocab["she"] = 500          # mid-word variant
        self.vocab["▁she"] = 501    # leading-space variant

    def __call__(self, text, *args, **kwargs):
        if kwargs.get("add_special_tokens") is False and isinstance(text, str):
            ids = []
            for i, word in enumerate(text.split()):
                spaced = i > 0 or text.startswith(" ")
                ids.append(self.vocab["▁" + word] if spaced and "▁" + word in self.vocab else self.vocab.get(word, 0))
            return {"input_ids": ids}
        return super().__call__(text, *args, **kwargs)


def _dataset(protocol=None):
    data = pd.DataFrame({
        "masked": ["token_5 token_6 [MASK]"],
        "factual": ["she"],
        "alternative": ["token_8"],
        "factual_class": ["c1"],
        "alternative_class": ["c2"],
        "factual_id": [1],
        "alternative_id": [2],
        "label": ["positive"],
        "feature_class_id": [1],
        "feature_pole": ["pos"],
    })
    kwargs = {} if protocol is None else {"label_token_protocol": protocol}
    return TextTrainingDataset(
        data=data, tokenizer=SpaceVariantTokenizer(), batch_size=1, is_decoder_only_model=True, **kwargs
    )


def _labelled_id(dataset):
    item = dataset._create_item("token_5 token_6", "she")
    labels = item["labels"]
    return labels[labels != -100].item()


def test_helper_returns_the_leading_space_variant():
    tok = SpaceVariantTokenizer()
    assert label_token_ids(tok, "token_5 token_6 ", "she") == [tok.vocab["▁she"]]


def test_first_differing_token_protocol_labels_with_the_leading_space_token():
    assert _labelled_id(_dataset("first_differing_token")) == 501


def test_no_leading_space_protocol_keeps_the_old_vocab_label():
    assert _labelled_id(_dataset("no_leading_space")) == 500


def test_direct_dataset_construction_requires_the_protocol():
    """No silent default: a caller that forgets the rule must fail, not label under the wrong one."""
    with pytest.raises(TypeError, match="label_token_protocol"):
        _dataset(None)


def test_dataset_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="label_token_protocol"):
        _dataset("bogus")


def test_new_training_arguments_default_to_first_differing_token():
    assert TrainingArguments().label_token_protocol == "first_differing_token"


def test_serialized_arguments_without_the_key_load_as_no_leading_space():
    """A checkpoint written before the protocol existed must not be re-interpreted."""
    payload = TrainingArguments().to_dict()
    payload.pop("label_token_protocol")
    assert TrainingArguments.from_dict(payload).label_token_protocol == "no_leading_space"


@pytest.mark.parametrize("value", ["first_differing_token", "no_leading_space"])
def test_protocol_round_trips_through_to_dict(value):
    args = TrainingArguments(label_token_protocol=value)
    assert TrainingArguments.from_dict(args.to_dict()).label_token_protocol == value


def test_training_arguments_reject_unknown_protocol():
    with pytest.raises(ValueError, match="label_token_protocol"):
        TrainingArguments(label_token_protocol="bogus")


@pytest.mark.parametrize("value", ["first_differing_token", "no_leading_space"])
def test_trainer_hands_its_protocol_to_every_dataset(value):
    fake_trainer = SimpleNamespace(training_args=TrainingArguments(label_token_protocol=value))
    assert TextPredictionTrainer._label_token_protocol(fake_trainer) == value


def test_trainer_without_training_args_falls_back_to_no_leading_space():
    assert TextPredictionTrainer._label_token_protocol(SimpleNamespace(training_args=None)) == "no_leading_space"


def test_every_model_a_trainer_creates_is_stamped_with_its_protocol():
    """Single-row scorers read the protocol from the model; it must come from the trainer.

    Goes through the real creation path (``create_model_with_gradiend``), which
    ``get_model``, ``load_model`` and training all share.
    """

    class _Model:
        @classmethod
        def from_pretrained(cls, load_directory, **kwargs):
            return cls()

    class _Trainer(TextPredictionTrainer):
        def __init__(self, protocol):  # bypass the heavy constructor
            self._training_args = TrainingArguments(label_token_protocol=protocol)

        @property
        def model_with_gradiend_cls(self):
            return _Model

    for protocol in ("first_differing_token", "no_leading_space"):
        model = _Trainer(protocol).create_model_with_gradiend("ignored")
        assert model.label_token_protocol == protocol


def test_training_label_token_id_matches_the_dataset_for_both_protocols():
    tok = SpaceVariantTokenizer()
    assert label_token_id(tok, "token_5 token_6 ", "she", "first_differing_token") == 501
    assert label_token_id(tok, "token_5 token_6 ", "she", "no_leading_space") == 500
    with pytest.raises(ValueError, match="label_token_protocol"):
        label_token_id(tok, "token_5 token_6 ", "she", "bogus")


@pytest.mark.parametrize("old,new", [("canonical", "first_differing_token"), ("legacy", "no_leading_space")])
def test_settings_saved_with_the_old_names_still_load(old, new):
    payload = TrainingArguments().to_dict()
    payload["label_token_protocol"] = old
    assert TrainingArguments.from_dict(payload).label_token_protocol == new


@pytest.mark.parametrize("old", ["canonical", "legacy"])
def test_old_names_are_rejected_outside_from_dict(old):
    with pytest.raises(ValueError, match="label_token_protocol"):
        TrainingArguments(label_token_protocol=old)
    with pytest.raises(ValueError, match="label_token_protocol"):
        _dataset(old)


def _loaded_model(saved_training_args):
    """A model as ``from_pretrained`` returns it: the saved training.json sits on ``model.gradiend.kwargs``."""
    training = None if saved_training_args is None else {"training_args": saved_training_args}
    return SimpleNamespace(gradiend=SimpleNamespace(kwargs={} if training is None else {"training": training}))


def _configured(trainer_protocol, saved_training_args):
    class _Model:
        @classmethod
        def from_pretrained(cls, load_directory, **kwargs):
            return _loaded_model(saved_training_args)

    class _Trainer(TextPredictionTrainer):
        def __init__(self, protocol):  # bypass the heavy constructor
            self._training_args = TrainingArguments(label_token_protocol=protocol)

        @property
        def model_with_gradiend_cls(self):
            return _Model

    trainer = _Trainer(trainer_protocol)
    return trainer, trainer.create_model_with_gradiend("ignored")


@pytest.mark.parametrize(
    "saved,expected",
    [
        ({"label_token_protocol": "no_leading_space"}, "no_leading_space"),
        ({"label_token_protocol": "first_differing_token"}, "first_differing_token"),
        ({"label_token_protocol": "legacy"}, "no_leading_space"),  # saved under the old name
        ({"label_token_protocol": "canonical"}, "first_differing_token"),  # saved under the old name
        ({"max_steps": 5}, "no_leading_space"),  # a record from before the key existed
    ],
)
def test_a_loaded_model_keeps_the_rule_it_was_trained_with(saved, expected):
    """The recorded rule wins over the trainer's current arguments, which datasets and evaluation then follow."""
    other = "first_differing_token" if expected == "no_leading_space" else "no_leading_space"
    trainer, model = _configured(other, saved)
    assert model.label_token_protocol == expected
    assert trainer._label_token_protocol() == expected


def test_a_model_without_a_record_takes_the_trainers_rule():
    for protocol in ("first_differing_token", "no_leading_space"):
        trainer, model = _configured(protocol, None)
        assert model.label_token_protocol == protocol
        assert trainer._label_token_protocol() == protocol
