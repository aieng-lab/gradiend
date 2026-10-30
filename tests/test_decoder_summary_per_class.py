"""evaluate_decoder summarizes the class that was asked for, with feature factors derived from the model.

The mock trainer returns the result layout of the real evaluation: ``probs[T]`` is P(T) on the other class's rows and
``probs_factual[C]`` is P(C) on C's own rows. The classes get different numbers, so a summary that reads the wrong
class, or the wrong feature factor, fails. How the trainer scored (class tokens or per row) is the trainer's business
and is covered by the ``evaluate_base_model`` tests.
"""

import pytest

from gradiend.evaluator.decoder import DecoderEvaluator, derive_feature_factor_for_class
from tests.test_decoder_strengthen_weaken_dataset import TrainerForStrengthenWeakenTestGrid

# (base model, rewritten model): P(word of the other class) on a class's own rows, and P(own word) on those rows
_ALTERNATIVE = {"3SG": (0.10, 0.60), "3PL": (0.15, 0.70)}   # P(word of the other class) on this class's rows
_FACTUAL = {"3SG": (0.55, 0.20), "3PL": (0.50, 0.30)}       # P(own word) on this class's rows


class _Trainer(TrainerForStrengthenWeakenTestGrid):
    def evaluate_base_model(self, base_model, tokenizer, *, training_like_df=None, **kwargs):
        column = "label_class" if "label_class" in training_like_df.columns else "factual_id"
        datasets = sorted(training_like_df[column].dropna().astype(str).unique().tolist())
        self._evaluate_base_model_calls.append({"dataset_classes": datasets})
        self._call_count += 1
        stage = 0 if self._call_count == 1 else 1  # the first call scores the base model

        def other(class_id):
            return next(c for c in self.target_classes if c != class_id)

        probs_by_dataset = {
            ds: {ds: _FACTUAL[ds][stage], other(ds): _ALTERNATIVE[ds][stage]} for ds in datasets
        }
        return {
            "lms": {"lms": 0.99},
            "probs": {c: probs_by_dataset[other(c)][c] for c in self.target_classes if other(c) in probs_by_dataset},
            "probs_factual": {c: probs_by_dataset[c][c] for c in self.target_classes if c in probs_by_dataset},
            "probs_by_dataset": probs_by_dataset,
        }


def _evaluate(target_class, *, increase=True):
    return DecoderEvaluator().evaluate_decoder(
        _Trainer(), target_class=target_class, increase_target_probabilities=increase, lrs=[1e-2], plot=False,
    )


@pytest.mark.parametrize("target_class, expected", [("3SG", 0.70), ("3PL", 0.60)])
def test_strengthen_reads_the_requested_class_from_the_other_classes_rows(target_class, expected):
    # P(3SG word) is measured on the 3PL rows (alternative of 3PL = 0.70), and the other way round
    assert _evaluate(target_class)[target_class]["value"] == pytest.approx(expected)


@pytest.mark.parametrize("target_class, expected", [("3SG", 0.80), ("3PL", 0.70)])
def test_weaken_reads_one_minus_the_own_word_probability_on_the_own_rows(target_class, expected):
    assert _evaluate(target_class, increase=False)[f"{target_class}_weaken"]["value"] == pytest.approx(expected)


def test_strengthen_uses_the_feature_factor_of_the_requested_class():
    trainer = _Trainer()
    for class_id in ("3SG", "3PL"):
        expected = derive_feature_factor_for_class(trainer, trainer.get_model(), class_id)
        assert expected != 0
        assert _evaluate(class_id)[class_id]["feature_factor"] == expected


def test_without_a_target_class_every_class_is_summarized_under_its_own_name():
    result = DecoderEvaluator().evaluate_decoder(_Trainer(), lrs=[1e-2], plot=False)
    assert result["3SG"]["value"] == pytest.approx(0.70)
    assert result["3PL"]["value"] == pytest.approx(0.60)
    assert result["3SG"]["feature_factor"] == -result["3PL"]["feature_factor"] != 0
