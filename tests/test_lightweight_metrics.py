import pytest

from gradiend.evaluator.encoder_metrics import get_correlation
from gradiend.util.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def test_accuracy_score_basic_and_empty():
    assert accuracy_score([1, 0, 1], [1, 0, 0]) == pytest.approx(2 / 3)
    assert accuracy_score([], []) == 0.0


def test_accuracy_score_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        accuracy_score([1], [1, 0])


def test_weighted_precision_recall_f1_with_zero_division_0():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "a", "a", "a"]  # class "b" never predicted

    # weighted precision = (prec_a*2 + prec_b*2) / 4 = (0.5*2 + 0*2)/4 = 0.25
    assert precision_score(y_true, y_pred, average="weighted", zero_division=0) == pytest.approx(0.25)
    # weighted recall = (rec_a*2 + rec_b*2) / 4 = (1*2 + 0*2)/4 = 0.5
    assert recall_score(y_true, y_pred, average="weighted", zero_division=0) == pytest.approx(0.5)
    # weighted f1 = (2/3*2 + 0*2) / 4 = 1/3
    assert f1_score(y_true, y_pred, average="weighted", zero_division=0) == pytest.approx(1 / 3)


def test_zero_division_argument_handling():
    y_true = ["a", "a"]
    y_pred = ["b", "b"]  # no predicted "a"

    p0 = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    p1 = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    pw = precision_score(y_true, y_pred, average="weighted", zero_division="warn")

    assert p0 == pytest.approx(0.0)
    assert p1 == pytest.approx(1.0)
    assert pw == pytest.approx(0.0)


def test_unsupported_average_raises():
    with pytest.raises(ValueError, match="average"):
        precision_score([1, 0], [1, 0], average="samples")


def test_numpy_correlation_matches_expected_perfect_linear():
    import pandas as pd

    df = pd.DataFrame(
        {
            "label": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "encoded": [-4.0, -2.0, 0.0, 2.0, 4.0],
        }
    )
    corr = get_correlation(df, encoded_col="encoded", label_col="label")
    assert corr == pytest.approx(1.0)


def test_numpy_correlation_constant_input_returns_zero():
    import pandas as pd

    df = pd.DataFrame({"label": [1.0, 1.0, 1.0], "encoded": [0.1, 0.2, 0.3]})
    corr = get_correlation(df, encoded_col="encoded", label_col="label")
    assert corr == 0.0
