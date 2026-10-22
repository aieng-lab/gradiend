"""Decoder grid cache validation and auto-invalidation helpers."""

import json

from gradiend.evaluator.decoder import (
    DecoderEvaluator,
    _decoder_cache_selection_context,
    _decoder_cache_selection_matches,
    _decoder_results_support_metrics,
)
from tests.test_decoder_strengthen_weaken_dataset import TrainerForStrengthenWeakenTest


def _grid_entry(probs, probs_factual, entry_id="c1"):
    return {
        "id": entry_id,
        "lms": {"lms": 0.99},
        "probs": probs,
        "probs_factual": probs_factual,
    }


def _context(**overrides):
    kwargs = dict(
        prob_on_other_class=True,
        increase_target_probabilities=True,
        metrics_for_summary=["3SG"],
        classes_to_eval=["3SG"],
    )
    kwargs.update(overrides)
    return _decoder_cache_selection_context(**kwargs)


def test_decoder_results_support_metrics_detects_missing_metric():
    results = {
        "base": {"id": "base", "lms": {"lms": 0.99}, "probs": {}, "probs_factual": {}},
        "c1": _grid_entry({"IO": 0.8, "neutral": 0.01}, {"IO": 0.3, "neutral": 0.99}),
    }
    ok, missing = _decoder_results_support_metrics(results, ["IO", "IO_weaken"])
    assert ok is True
    assert missing == []

    broken = {
        "base": {"id": "base", "lms": {"lms": 0.99}, "probs": {}, "probs_factual": {}},
        "c1": _grid_entry({"neutral": 0.01}, {"neutral": 0.99}),
    }
    ok, missing = _decoder_results_support_metrics(broken, ["IO", "neutral_weaken"])
    assert ok is False
    assert missing == ["IO"]


def test_decoder_cache_selection_matches_requires_context_when_present():
    expected = _context()
    # A cache written before the context existed carries none and stays reusable.
    assert _decoder_cache_selection_matches({}, expected) is True
    assert _decoder_cache_selection_matches({"selection_context": expected}, expected) is True
    for key, value in (
        ("prob_on_other_class", False),
        ("increase_target_probabilities", False),
        ("metrics_for_summary", ["3SG_weaken"]),
        ("classes_to_eval", ["3PL"]),
    ):
        wrong = dict(expected)
        wrong[key] = value
        assert _decoder_cache_selection_matches({"selection_context": wrong}, expected) is False, key


def test_selection_context_ignores_ordering_of_metrics_and_classes():
    assert _context(metrics_for_summary=["a", "b"], classes_to_eval=["x", "y"]) == _context(
        metrics_for_summary=["b", "a"], classes_to_eval=["y", "x"]
    )


def _evaluate(evaluator, trainer, *, strengthen=True):
    return evaluator.evaluate_decoder(
        trainer,
        target_class="3SG",
        increase_target_probabilities=strengthen,
        feature_factors=[-1.0] if strengthen else [1.0],
        lrs=[1e-2],
        plot=False,
        use_cache=True,
    )


def test_cache_is_reused_for_the_same_request_and_recomputed_for_the_other_direction(tmp_path):
    evaluator = DecoderEvaluator()
    trainer = TrainerForStrengthenWeakenTest()
    trainer.experiment_dir = str(tmp_path)

    first = _evaluate(evaluator, trainer)
    computed = len(trainer._evaluate_base_model_calls)
    assert computed >= 1

    again = _evaluate(evaluator, trainer)
    assert len(trainer._evaluate_base_model_calls) == computed, "same request must hit the cache"
    assert again["3SG"]["value"] == first["3SG"]["value"]

    cache_file = tmp_path / "decoder_grid_cache.json"
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["selection_context"]["increase_target_probabilities"] is True

    # Weakening reads other metrics (``3SG_weaken``): the strengthen grid must not be reused for it.
    weaken = _evaluate(evaluator, trainer, strengthen=False)
    assert len(trainer._evaluate_base_model_calls) > computed
    assert "3SG_weaken" in weaken
    refreshed = json.loads(cache_file.read_text(encoding="utf-8"))
    assert refreshed["selection_context"]["increase_target_probabilities"] is False


def test_stale_decoder_cache_missing_metric_recomputes(tmp_path):
    evaluator = DecoderEvaluator()
    trainer = TrainerForStrengthenWeakenTest()
    trainer.experiment_dir = str(tmp_path)

    # Matches the request structurally, but its rows carry no ``3SG`` metric (written by older code).
    cache_payload = {
        "part": "decoder",
        "split": "test",
        "max_size_training_like": 50,
        "max_size_neutral": 50,
        "feature_factors": [-1.0],
        "lrs": [0.01],
        "results": [
            {"id": "base", "lms": {"lms": 0.99}, "probs": {"neutral": 0.01}, "probs_factual": {"neutral": 0.99}},
            {
                "id": {"feature_factor": -1.0, "learning_rate": 0.01},
                "lms": {"lms": 0.98},
                "probs": {"neutral": 0.02},
                "probs_factual": {"neutral": 0.98},
            },
        ],
    }
    cache_file = tmp_path / "decoder_grid_cache.json"
    cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")

    result = _evaluate(evaluator, trainer)

    assert "3SG" in result
    assert result["3SG"]["value"] == 0.9
    assert len(trainer._evaluate_base_model_calls) >= 1
    refreshed = json.loads(cache_file.read_text(encoding="utf-8"))
    assert refreshed["selection_context"]["metrics_for_summary"] == ["3SG"]
