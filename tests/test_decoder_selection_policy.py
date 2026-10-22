"""The default decoder selection must work for tuple ids and for dict ids (as read back from a cached grid)."""

from gradiend.evaluator.decoder import BaseContext, Candidate, LMSThresholdPolicy


def _candidates(make_id):
    return [Candidate(id=make_id(-1.0, lr), lms=0.5, metrics={"A": 0.1}) for lr in (0.1, 0.01)]


def test_fallback_picks_the_smallest_learning_rate_for_tuple_ids():
    # nobody reaches the LMS cutoff (0.99 * 1.0), so the fallback by smallest |lr| applies
    chosen = LMSThresholdPolicy(ratio=0.99).select("A", _candidates(lambda ff, lr: (ff, lr)), BaseContext(base_lms=1.0))
    assert chosen.id == (-1.0, 0.01)


def test_fallback_picks_the_smallest_learning_rate_for_dict_ids():
    ids = lambda ff, lr: {"feature_factor": ff, "learning_rate": lr}  # noqa: E731
    chosen = LMSThresholdPolicy(ratio=0.99).select("A", _candidates(ids), BaseContext(base_lms=1.0))
    assert chosen.id["learning_rate"] == 0.01
