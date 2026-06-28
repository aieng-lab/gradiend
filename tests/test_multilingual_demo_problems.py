"""--problems filtering for multilingual_gradiend_demo."""

import pytest


@pytest.fixture(scope="module")
def demo():
    import experiments.multilingual_gradiend_demo as mod

    return mod


def test_parse_problems_default_is_all(demo):
    assert demo._parse_problems([]) is None


def test_parse_problems_single_problem(demo):
    assert demo._parse_problems(["gender_de"]) == frozenset({"gender_de"})


def test_parse_problems_comma_separated(demo):
    assert demo._parse_problems(["gender_de,pronoun"]) == frozenset({"gender_de", "pronoun"})


def test_parse_problems_repeated_args(demo):
    assert demo._parse_problems(["gender_de", "pronoun"]) == frozenset({"gender_de", "pronoun"})


def test_parse_problems_train_race_religion_alias(demo):
    assert demo._parse_problems(["train_race_religion"]) == frozenset({"race", "religion"})


def test_parse_problems_rejects_unknown(demo):
    with pytest.raises(ValueError, match="Unknown problem"):
        demo._parse_problems(["not_a_problem"])


def test_problem_selected_respects_filter(demo):
    selected = frozenset({"gender_de"})
    assert demo._problem_selected(selected, "gender_de") is True
    assert demo._problem_selected(selected, "pronoun") is False
    assert demo._problem_selected(None, "pronoun") is True


def test_demo_problem_keys_cover_learning_rate_problems(demo):
    for key in ("gender_de", "gender_en", "pronoun", "pronoun_merged", "sentiment"):
        assert key in demo.DEMO_PROBLEM_KEYS
