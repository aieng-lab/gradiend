"""LR search: classification of probe runs and the bracket/bisect controller (pure logic, no GPU)."""

import json
import math

import numpy as np
import pytest

from gradiend.trainer.core.signal_checks import SignalNotDiverseError, assert_signal_diverse
from gradiend.trainer.core.lr_search import (
    LRSearch,
    LRSearchConfig,
    LRSearchResult,
    NoConvergentLearningRate,
    Probe,
    ProbeResult,
    RunState,
    _format_lr,
    classify_probe,
    nice_lr_grid,
    snap_to_grid,
)

STEPS = [0, 50, 100, 150, 200, 250]


def _result(scores, pos, neg, converged=False, losses=None):
    return ProbeResult(
        steps=STEPS[: len(scores)],
        scores=list(scores),
        converged=converged,
        mean_by_class=[{"1.0": p, "-1.0": n, "0.0": 0.0} for p, n in zip(pos, neg)],
        losses=losses or [0.1] * 10,
    )


COLLAPSE = ([0.5, 0.3, 0.05], [-0.95, -1.0, -1.0], [-1.0, -1.0, -1.0])
GOOD = ([0.1, 0.6, 0.95], [0.1, 0.5, 0.9], [-0.1, -0.5, -0.9])
FROZEN = ([0.56] * 4, [0.09] * 4, [-0.14] * 4)


class TestClassify:
    def test_converged_trusts_the_package_decision(self):
        assert classify_probe(_result(*GOOD, converged=True)) is RunState.CONVERGED

    def test_high_initial_correlation_is_not_convergence(self):
        # gemma-2 @1e-7: correlation 0.56 at init decays to 0.19, class means never separate
        scores = [0.56, 0.55, 0.54, 0.53, 0.50, 0.45, 0.40, 0.33, 0.26, 0.22, 0.19]
        pos = [0.096 - 0.01 * i for i in range(11)]
        neg = [-0.14 - 0.01 * i for i in range(11)]
        r = ProbeResult(
            steps=list(range(0, 550, 50)), scores=scores, converged=False,
            mean_by_class=[{"1.0": p, "-1.0": n} for p, n in zip(pos, neg)],
        )
        assert classify_probe(r) is RunState.DEGRADING

    def test_frozen_when_nothing_moves(self):
        assert classify_probe(_result(*FROZEN)) is RunState.FROZEN

    def test_collapse_both_poles_same_side(self):
        assert classify_probe(_result(*COLLAPSE)) is RunState.COLLAPSED

    def test_slow_when_still_improving(self):
        r = _result([0.0, 0.1, 0.2, 0.3], [0.0, 0.05, 0.1, 0.2], [0.0, -0.05, -0.1, -0.2])
        assert classify_probe(r) is RunState.SLOW

    def test_shrinking_pole_gap_is_drift_even_before_the_score_drops(self):
        r = _result([0.56, 0.56, 0.55], [0.3, 0.2, 0.1], [-0.3, -0.2, -0.1])
        assert classify_probe(r) is RunState.DEGRADING

    def test_nonfinite_loss_is_divergence(self):
        r = _result([0.5, 0.5], [0.1, 0.1], [-0.1, -0.1], losses=[0.1, float("nan")])
        assert classify_probe(r) is RunState.DIVERGED

    @staticmethod
    def _gemma3(scores, f, m):
        # gemma-3-4b-pt gender_en pair (slurm 4328305_4 / 4328306_1); F is the +1 pole
        return ProbeResult(
            steps=[16 * i for i in range(len(scores))], scores=list(scores), converged=False,
            mean_by_class=[{"1.0": a, "-1.0": b} for a, b in zip(f, m)],
        )

    def test_unmoved_run_that_starts_saturated_on_one_side_is_frozen_not_collapsed(self):
        # ACTIEND starts at F=-0.62, M=-0.56; at lr 1e-6..1e-12 nothing moves. Old rule: "collapsed"
        # -> LR down forever -> no_convergent_lr after 8 probes.
        assert classify_probe(self._gemma3([-0.3443, -0.3443], [-0.6218, -0.6218], [-0.5644, -0.5644])) is RunState.FROZEN
        assert classify_probe(self._gemma3([-0.3443, -0.3462], [-0.6218, -0.6205], [-0.5644, -0.5625])) is RunState.FROZEN

    def test_one_sided_run_whose_pole_gap_opens_is_not_collapsed(self):
        # ACTIEND lr 1e-5: corr -0.34 -> 0.74 -> 0.60, gap F-M -0.06 -> +0.12, both poles > 0.5
        r = self._gemma3([-0.3443, -0.5148, 0.7426, 0.6045], [-0.6218, -0.5153, 0.6349, 0.7137], [-0.5644, -0.3906, 0.4650, 0.5925])
        assert classify_probe(r) is RunState.SLOW

    def test_run_creeping_around_a_saturated_start_is_slow_not_collapsed(self):
        # ACTIEND lr 2e-6 (slurm 4328827): moved 0.05 in correlation, poles barely moved
        r = self._gemma3(
            [-0.3443, -0.3524, -0.3595, -0.3754, -0.3955],
            [-0.6218, -0.6160, -0.6212, -0.6286, -0.6365],
            [-0.5644, -0.5563, -0.5614, -0.5674, -0.5730],
        )
        assert classify_probe(r) is RunState.SLOW

    def test_full_lr_1e5_run_is_degrading(self):
        # ACTIEND lr 1e-5 run to the end: peak 0.74, then both poles drift to +0.8/+0.9
        r = self._gemma3(
            [-0.3443, -0.5148, 0.7426, 0.6045, 0.4271, 0.2455, 0.1886, 0.1905],
            [-0.6218, -0.5153, 0.6349, 0.7137, 0.7684, 0.8470, 0.8875, 0.8906],
            [-0.5644, -0.3906, 0.4650, 0.5925, 0.6593, 0.7593, 0.8044, 0.8037],
        )
        assert classify_probe(r) is RunState.DEGRADING

    def test_one_sided_run_whose_gap_does_not_open_is_still_collapsed(self):
        # AGIEND lr 0.5 / 1.0: both poles pushed to the neutral pole
        assert classify_probe(self._gemma3([-0.2486, -0.1087], [0.0028, 0.9593], [0.0033, 1.0])) is RunState.COLLAPSED
        r = self._gemma3([-0.2486, -0.4491, -0.4424, -0.4033], [0.0028, 0.3352, 0.3683, 0.5896], [0.0033, 1.0, 1.0, 0.9999])
        assert classify_probe(r) is RunState.COLLAPSED

    def test_direction(self):
        assert RunState.FROZEN.direction == +1 and RunState.SLOW.direction == +1
        assert RunState.COLLAPSED.direction == -1 and RunState.DEGRADING.direction == -1
        assert RunState.CONVERGED.direction == 0


class TestGrid:
    def test_coarse_grid_is_one_two_five(self):
        assert nice_lr_grid(1e-8, 1e-7) == pytest.approx([2e-8, 5e-8])

    def test_fine_grid_adds_intermediates(self):
        assert nice_lr_grid(1e-8, 1e-7, "fine") == pytest.approx([1.5e-8, 2e-8, 3e-8, 5e-8, 7e-8])

    def test_open_interval(self):
        assert nice_lr_grid(2e-8, 5e-8) == []

    def test_snap(self):
        assert snap_to_grid(1.1e-6) == pytest.approx(1e-6)
        assert snap_to_grid(3.4e-7) in (pytest.approx(2e-7), pytest.approx(5e-7))

    def test_bad_resolution(self):
        with pytest.raises(ValueError):
            nice_lr_grid(1e-8, 1e-7, "nope")


def _window_run(good_lo, good_hi):
    """LR below the window is frozen, above it collapses, inside it converges."""
    calls = []

    def run(lr):
        calls.append(lr)
        if lr < good_lo:
            return _result(*FROZEN)
        if lr > good_hi:
            return _result(*COLLAPSE)
        return _result(*GOOD, converged=True)

    return run, calls


class TestSearch:
    def test_finds_a_convergent_lr_expanding_upward(self):
        run, calls = _window_run(3e-7, 3e-6)
        res = LRSearch(run, 1e-9).search()
        assert res.status == "converged"
        assert 3e-7 <= res.best_lr <= 3e-6
        assert calls[:3] == pytest.approx([1e-9, 1e-8, 1e-7])

    def test_finds_a_convergent_lr_expanding_downward(self):
        run, _ = _window_run(3e-7, 3e-6)
        res = LRSearch(run, 1e-3).search()
        assert res.status == "converged" and 3e-7 <= res.best_lr <= 3e-6

    def test_bisects_a_narrow_window_between_the_decades(self):
        run, _ = _window_run(4e-7, 6e-7)  # only 5e-7 converges on the coarse grid
        res = LRSearch(run, 1e-6).search()
        assert res.status == "converged" and res.best_lr == pytest.approx(5e-7)

    def test_no_convergent_lr_closes_the_bracket_with_a_verdict(self):
        # frozen below 1e-8, drifting from 1e-8 up: nothing converges anywhere
        def run(lr):
            if lr < 1e-8:
                return _result(*FROZEN)
            return _result([0.56, 0.4, 0.1], [0.05, 0.02, 0.0], [-0.2, -0.1, -0.05])

        res = LRSearch(run, 1e-6, max_probes=20).search()
        assert res.status == "no_convergent_lr"
        low, high = res.bracket
        assert high / low <= 2.0 + 1e-9
        assert res.best_lr is None
        assert len(res.probes) < 20  # it stopped by itself, well under the budget

    def test_never_probes_the_same_lr_twice(self):
        run, calls = _window_run(1e-5, 2e-5)
        LRSearch(run, 1e-9, max_probes=20).search()
        assert len(calls) == len(set(calls))

    def test_budget_exhaustion_is_reported_not_disguised(self):
        run, _ = _window_run(1e30, 1e31)  # frozen everywhere, bounds far away
        res = LRSearch(run, 1e-9, max_probes=3).search()
        assert res.status == "budget_exhausted" and len(res.probes) == 3

    def test_lr_bounds_stop_the_expansion(self):
        run, _ = _window_run(1e30, 1e31)
        res = LRSearch(run, 1e-3, lr_bounds=(1e-12, 1e-1), max_probes=20).search()
        assert res.status == "no_convergent_lr"
        assert max(p.lr for p in res.probes) <= 1e-1 * (1 + 1e-9)

    def test_failed_confirmation_demotes_the_probe_and_continues(self):
        run, _ = _window_run(1e-7, 1e-5)

        def confirm(lr):  # the short probe converged, the full budget collapses above 1e-6
            if lr > 1e-6:
                return _result(*COLLAPSE)
            return _result(*GOOD, converged=True)

        res = LRSearch(run, 1e-4, confirm=confirm).search()
        assert res.status == "converged" and res.best_lr <= 1e-6
        demoted = [p for p in res.probes if p.confirmed is False]
        assert demoted and all(p.state is RunState.COLLAPSED for p in demoted)

    def test_validation(self):
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 0.0)
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 1e-6, factor=1.0)


class TestSignalDiversity:
    def test_identical_rows_of_a_class_are_rejected(self):
        x = np.vstack([np.ones((5, 4)), -np.ones((5, 4))])
        y = [1] * 5 + [-1] * 5
        with pytest.raises(SignalNotDiverseError, match="identical"):
            assert_signal_diverse(x, y, where="unit")

    def test_varied_rows_pass(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 4))
        assert_signal_diverse(x, [1] * 5 + [-1] * 5)


class _FakeTrainer:
    """Mimics Trainer.train(): afterwards it points at the checkpoint it wrote and holds the model."""

    def __init__(self, tmp_path):
        self._base_model_arg = "base-model"
        self._model_arg = "base-model"
        self._model_instance = None
        self._training_args = type("A", (), {"max_steps": 500, "eval_steps": 50})()
        self.started_from = []
        self.tmp_path = tmp_path

    def train(self, **kw):
        self.started_from.append((self._model_arg, self._model_instance))
        out = kw["experiment_dir"] + "/model"
        import os

        os.makedirs(out, exist_ok=True)
        good = kw["learning_rate"] >= 1e-6
        payload = {
            "training_stats": {
                "scores": {"0": 0.1, "50": 0.9 if good else 0.1},
                "mean_by_class": {"0": {"1.0": 0.1, "-1.0": -0.1},
                                  "50": {"1.0": 0.9, "-1.0": -0.9} if good else {"1.0": 0.1, "-1.0": -0.1}},
            },
            "losses": [0.1],
            "convergence_info": {"converged": good},
        }
        with open(out + "/training.json", "w") as fh:
            json.dump(payload, fh)
        with open(out + "/model.safetensors", "wb") as fh:
            fh.write(b"x" * 1024)  # stands in for a checkpoint
        self._model_arg = out  # what the real Trainer does
        self._model_instance = object()  # the trained model stays resident
        return self


class _SeedDependentTrainer(_FakeTrainer):
    """Only seed 1 converges; every other seed drifts both classes to the same pole (a collapse)."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self._training_args = type("A", (), {"max_steps": 500, "eval_steps": 50, "seed": 0})()
        self.seeds = []

    def train(self, **kw):
        self.seeds.append(kw.get("seed", 0))
        good = kw.get("seed", 0) == 1
        out = kw["experiment_dir"] + "/model"
        import os

        os.makedirs(out, exist_ok=True)
        payload = {
            "training_stats": {
                "scores": {"0": 0.2, "50": 0.95 if good else 0.0},
                "mean_by_class": {"0": {"1.0": 0.05, "-1.0": 0.04},
                                  "50": {"1.0": 0.9, "-1.0": -0.9} if good else {"1.0": -0.8, "-1.0": -0.7}},
            },
            "losses": [0.1],
            "convergence_info": {"converged": good},
        }
        with open(out + "/training.json", "w") as fh:
            json.dump(payload, fh)
        self._model_arg, self._model_instance = out, object()
        return self


def test_a_collapsed_probe_is_retried_with_the_next_seed(tmp_path):
    from gradiend.trainer.core.lr_search import LRSearchConfig, RunState, tune_learning_rate

    single = _SeedDependentTrainer(tmp_path)
    one = tune_learning_rate(single, 1e-6, experiment_dir=str(tmp_path / "a"),
                             config=LRSearchConfig(probe_steps=100, confirm=False, max_probes=1))
    assert one.probes[0].state is RunState.COLLAPSED and single.seeds == [0]

    multi = _SeedDependentTrainer(tmp_path)
    three = tune_learning_rate(multi, 1e-6, experiment_dir=str(tmp_path / "b"),
                               config=LRSearchConfig(probe_steps=100, confirm=False, max_probes=1, probe_seeds=3))
    assert three.probes[0].state is RunState.CONVERGED
    # the first probe: seed 0 collapsed, seed 1 converged, so the third seed never runs;
    # whatever probes follow (window centring) start again at seed 0
    assert multi.seeds[:2] == [0, 1] and 2 not in multi.seeds


def test_probe_seeds_must_be_positive():
    import pytest

    from gradiend.trainer.core.lr_search import LRSearchConfig

    with pytest.raises(ValueError, match="probe_seeds"):
        LRSearchConfig(probe_seeds=0)


def test_every_probe_starts_from_the_base_model_and_the_trainer_is_left_untouched(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    trainer = _FakeTrainer(tmp_path)
    res = tune_learning_rate(
        trainer, 1e-8, experiment_dir=str(tmp_path / "exp"),
        config=LRSearchConfig(probe_steps=100, confirm=False, max_probes=6),
    )
    assert res.status == "converged" and res.best_lr >= 1e-6
    assert len(trainer.started_from) >= 2
    assert all(arg == "base-model" and inst is None for arg, inst in trainer.started_from), (
        "a later probe continued from the previous probe's checkpoint"
    )
    assert trainer._model_arg == "base-model" and trainer._model_instance is None
    assert (tmp_path / "exp" / "lr_search.json").exists()


def test_confirmation_reruns_at_the_full_budget(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    seen = []
    trainer = _FakeTrainer(tmp_path)
    original = trainer.train

    def spy(**kw):
        seen.append(kw.get("max_steps"))
        return original(**kw)

    trainer.train = spy
    tune_learning_rate(
        trainer, 1e-6, experiment_dir=str(tmp_path / "exp"),
        config=LRSearchConfig(probe_steps=100, confirm=True, center_window=False),
    )
    assert seen == [100, 500]


def test_adapter_installs_the_guard_always_and_a_fresh_stop_callback_unless_disabled(tmp_path):
    from gradiend.trainer.core.lr_search import ProbeStopCallback, tune_learning_rate
    from gradiend.trainer.core.signal_checks import SignalDiversityCallback

    seen = []
    trainer = _FakeTrainer(tmp_path)
    original = trainer.train
    trainer.train = lambda **kw: (seen.append(kw.get("callbacks")), original(**kw))[1]
    tune_learning_rate(trainer, 1e-6, experiment_dir=str(tmp_path / "a"),
                       config=LRSearchConfig(probe_steps=100, confirm=False, center_window=False))
    assert [type(c) for c in seen[0]] == [SignalDiversityCallback, ProbeStopCallback]
    seen.clear()
    tune_learning_rate(trainer, 1e-6, experiment_dir=str(tmp_path / "b"),
                       config=LRSearchConfig(probe_steps=100, confirm=False, center_window=False, early_stop=False))
    assert [[type(c) for c in cbs] for cbs in seen] == [[SignalDiversityCallback]]


class TestWindowCentering:
    def test_returns_the_centre_of_the_convergent_window_not_its_first_hit(self):
        run, calls = _window_run(1e-6, 1e-4)  # convergent: 1e-6 .. 1e-4 (grid: 1,2,5 x)
        res = LRSearch(run, 1e-6).search()  # first hit is the lower edge
        assert res.status == "converged"
        assert res.window == pytest.approx((1e-6, 1e-4)) or res.window[0] >= 1e-6
        centre = (res.window[0] * res.window[1]) ** 0.5
        assert res.best_lr not in (res.window[0], res.window[1])
        assert abs(math.log(res.best_lr / centre)) < math.log(2.6)

    def test_window_walk_stops_at_the_first_non_convergent_neighbour(self):
        run, calls = _window_run(4e-7, 6e-7)  # only 5e-7 converges
        res = LRSearch(run, 5e-7).search()
        assert res.best_lr == pytest.approx(5e-7) and res.window == pytest.approx((5e-7, 5e-7))
        assert sorted(calls) == pytest.approx([2e-7, 5e-7, 1e-6])  # exactly the two neighbours were checked

    def test_window_probes_have_their_own_budget(self):
        run, calls = _window_run(1e-9, 1e9)  # everything converges
        res = LRSearch(run, 1e-6, max_probes=1, max_window_probes=3).search()
        assert res.status == "converged" and len(calls) == 1 + 3

    def test_both_sides_are_explored_even_when_the_window_is_wider_than_the_budget(self):
        run, calls = _window_run(1e-9, 1e9)  # everything converges: the walk is only limited by its budget
        res = LRSearch(run, 1e-6, max_window_probes=4).search()
        assert sorted(c for c in calls if c != 1e-6) == pytest.approx([2e-7, 5e-7, 2e-6, 5e-6])  # two per side
        assert res.window == pytest.approx((2e-7, 5e-6))

    def test_disabled(self):
        run, calls = _window_run(1e-9, 1e9)
        res = LRSearch(run, 1e-6, center_window=False).search()
        assert res.best_lr == pytest.approx(1e-6) and len(calls) == 1

    def test_ties_go_to_the_lower_lr(self):
        run, _ = _window_run(1e-6, 2e-6)  # two convergent grid points, equal distance to the centre
        res = LRSearch(run, 1e-6).search()
        assert res.best_lr == pytest.approx(1e-6)

    def test_reported_window_probes_are_never_repeated(self):
        run, calls = _window_run(1e-7, 1e-5)
        LRSearch(run, 1e-9).search()
        assert len(calls) == len(set(round(c, 15) for c in calls))


class TestGridResolution:
    def test_bracket_closes_on_the_grid_not_on_a_ratio(self):
        # frozen below 3e-8, collapsed from 3e-8 up; fine grid has interior points down to ratio 1.33
        def run(lr):
            return _result(*FROZEN) if lr < 3e-8 else _result(*COLLAPSE)

        res = LRSearch(run, 1e-6, resolution="fine", max_probes=30).search()
        assert res.status == "no_convergent_lr"
        low, high = res.bracket
        assert nice_lr_grid(low, high, "fine") == []  # nothing untried left inside
        assert high / low < 1.7  # finer than the old ratio-2 stop

    def test_coarse_grid_closes_at_two_to_two_and_a_half(self):
        def run(lr):
            return _result(*FROZEN) if lr < 3e-8 else _result(*COLLAPSE)

        res = LRSearch(run, 1e-6, max_probes=30).search()
        low, high = res.bracket
        assert nice_lr_grid(low, high) == [] and high / low <= 2.5 + 1e-9

    def test_optional_min_ratio_still_stops_earlier(self):
        def run(lr):
            return _result(*FROZEN) if lr < 3e-8 else _result(*COLLAPSE)

        res = LRSearch(run, 1e-6, resolution="fine", min_ratio=2.0, max_probes=30).search()
        low, high = res.bracket
        assert high / low <= 2.0 + 1e-9

    def test_expansion_by_division_does_not_reprobe_grid_values(self):
        # 1e-6 / 10 is not bit-identical to the grid literal 1e-7; it must still count as tried
        run, calls = _window_run(1e30, 1e31)
        LRSearch(run, 1e-3, lr_bounds=(1e-12, 1e-2), max_probes=30).search()
        assert len(calls) == len(set(round(c, 15) for c in calls))

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 1e-6, resolution="nope")
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 1e-6, min_ratio=1.0)


def test_grid_neighbor():
    from gradiend.trainer.core.lr_search import grid_neighbor

    assert grid_neighbor(1e-7, +1) == pytest.approx(2e-7)
    assert grid_neighbor(5e-7, +1) == pytest.approx(1e-6)
    assert grid_neighbor(1e-7, -1) == pytest.approx(5e-8)
    assert grid_neighbor(1e-7, +1, "fine") == pytest.approx(1.5e-7)
    assert grid_neighbor(2e-7, -1, "fine") == pytest.approx(1.5e-7)
    with pytest.raises(ValueError):
        grid_neighbor(1e-7, 0)


class TestProbeStopCallback:
    @staticmethod
    def _stats(scores, pos, neg):
        return {
            "scores": {i * 50: s for i, s in enumerate(scores)},
            "mean_by_class": {i * 50: {1.0: p, -1.0: n} for i, (p, n) in enumerate(zip(pos, neg))},
        }

    def _step(self, cb, stats, loss=0.1, step=0):
        control = {"should_stop": False}
        cb.on_step_end(step=step, loss=loss, model=None, config={}, control=control, training_stats=stats)
        return control["should_stop"]

    def test_stops_on_collapse_but_not_on_drift_or_slow_progress(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        cb = ProbeStopCallback()
        assert not self._step(cb, self._stats([0.5], [0.1], [-0.1]))  # below min_evals
        assert self._step(cb, self._stats([0.5, 0.1], [0.1, -0.9], [-0.1, -1.0]))
        assert cb.stopped_state is RunState.COLLAPSED

        cb = ProbeStopCallback()
        self._step(cb, self._stats([0.56], [0.1], [-0.14]))
        assert not self._step(cb, self._stats([0.56, 0.2], [0.1, 0.02], [-0.14, -0.2]))  # drift: not decisive

    def test_only_judges_when_a_new_evaluation_exists(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        calls = []

        def classify(result):
            calls.append(len(result.scores))
            return RunState.SLOW

        cb = ProbeStopCallback(classify=classify)
        stats = self._stats([0.1, 0.2], [0.0, 0.1], [0.0, -0.1])
        self._step(cb, stats)
        self._step(cb, stats)  # same evaluations again (a step without evaluation)
        assert calls == [2]

    def test_nonfinite_loss_stops_immediately(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        cb = ProbeStopCallback()
        assert self._step(cb, {}, loss=float("nan"), step=7)
        assert cb.stopped_state is RunState.DIVERGED and cb.stopped_step == 7

    def test_configurable_stop_states(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        cb = ProbeStopCallback(stop_states=(RunState.DEGRADING,))
        self._step(cb, self._stats([0.56], [0.1], [-0.14]))
        assert self._step(cb, self._stats([0.56, 0.2], [0.1, 0.02], [-0.14, -0.2]))

    def test_accepts_integer_and_string_step_keys(self):
        stats = {"scores": {"0": 0.1, "50": 0.2}, "mean_by_class": {"0": {"1.0": 0.0}, "50": {"1.0": 0.1}}}
        r = ProbeResult.from_training_stats(stats)
        assert r.steps == [0, 50] and r.mean_by_class[1] == {"1.0": 0.1}


class TestTrainingLoopHonoursShouldStop:
    def test_step_loop_stops_mid_epoch_and_still_writes_the_results(self, tmp_path):
        import torch
        from torch.utils.data import DataLoader

        from gradiend.trainer.core.arguments import TrainingArguments
        from gradiend.trainer.core.callbacks import TrainingCallback
        from gradiend.trainer.core.dataset import GradientTrainingDataset
        from gradiend.trainer.core.stats import load_training_stats
        from gradiend.trainer.core.training import train
        from tests.test_training_loop import MockModelWithGradiend, MockTrainingData

        data = MockTrainingData([{"factual": torch.randn(10), "alternative": torch.randn(10), "label": 1.0}] * 50)
        dataset = GradientTrainingDataset(
            training_data=data, gradient_creator=lambda inputs: torch.randn(100), source="factual", target="diff",
        )
        steps = []

        class StopAtThree(TrainingCallback):
            def on_step_end(self, step, loss, model, config, **kwargs):
                steps.append(step)
                if step >= 3:
                    kwargs["control"]["should_stop"] = True

        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=40, eval_steps=100, train_batch_size=1,
            num_train_epochs=1, convergent_score_threshold=None,
        )
        out = train(
            model_with_gradiend=MockModelWithGradiend(), data=DataLoader(dataset, batch_size=1),
            training_args=args, callbacks=[StopAtThree()],
        )
        assert isinstance(out, str)
        assert max(steps) == 3, f"training continued past the stop request: {steps}"
        assert load_training_stats(str(tmp_path))["training_stats"]["global_step"] == 3


def test_probe_result_reads_training_json(tmp_path):
    payload = {
        "training_stats": {
            "scores": {"0": 0.5, "50": 0.6},
            "mean_by_class": {"0": {"1.0": 0.1, "-1.0": -0.1}, "50": {"1.0": 0.5, "-1.0": -0.5}},
        },
        "losses": [0.1, 0.2],
        "convergence_info": {"converged": True, "threshold": 0.5, "convergence_metric": "correlation"},
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(payload))
    r = ProbeResult.from_training_json(str(path))
    assert r.steps == [0, 50] and r.scores == [0.5, 0.6] and r.converged
    assert r.mean_by_class[1]["1.0"] == 0.5 and r.metric == "correlation"
    assert classify_probe(r) is RunState.CONVERGED


class TestSignalDiversityCallback:
    @staticmethod
    def _eval(std, n=20):
        return {"std_by_class": {1.0: std, -1.0: std, 0.0: 0.3}, "n_by_class": {1.0: n, -1.0: n, 0.0: n}}

    def test_identical_inputs_raise_at_the_first_evaluation(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback()
        with pytest.raises(SignalNotDiverseError, match="identical"):
            cb.on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=self._eval(0.0))

    def test_spread_encodings_pass_and_the_check_is_then_inert(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback()
        cb.on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=self._eval(0.05))
        assert cb.checked
        # a later, saturated evaluation (collapse) is the LR search's business, not a signal bug
        cb.on_step_end(step=50, loss=0.1, model=None, config={}, eval_result=self._eval(0.0))

    def test_one_diverse_class_is_enough(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        result = {"std_by_class": {1.0: 0.0, -1.0: 0.2}, "n_by_class": {1.0: 20, -1.0: 20}}
        SignalDiversityCallback().on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=result)

    def test_tiny_classes_and_neutral_rows_are_not_evidence(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback(min_examples=4)
        result = {"std_by_class": {1.0: 0.0, -1.0: 0.0, 0.0: 0.0}, "n_by_class": {1.0: 2, -1.0: 3, 0.0: 50}}
        cb.on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=result)  # nothing judged: no error

    def test_waits_for_an_evaluation_that_carries_the_statistic(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback()
        cb.on_step_end(step=0, loss=0.1, model=None, config={})  # no eval_result
        cb.on_step_end(step=1, loss=0.1, model=None, config={}, eval_result={"correlation": 0.1})
        assert not cb.checked


class TestDiversityCallbackInTheRealLoop:
    @staticmethod
    def _run(tmp_path, std):
        import torch
        from torch.utils.data import DataLoader

        from gradiend.trainer.core.arguments import TrainingArguments
        from gradiend.trainer.core.dataset import GradientTrainingDataset
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback
        from gradiend.trainer.core.training import train
        from tests.test_training_loop import MockModelWithGradiend, MockTrainingData

        data = MockTrainingData([{"factual": torch.randn(10), "alternative": torch.randn(10), "label": 1.0}] * 20)
        dataset = GradientTrainingDataset(
            training_data=data, gradient_creator=lambda inputs: torch.randn(100), source="factual", target="diff",
        )

        def evaluate_fn(config=None, training_stats=None, **kw):
            return {
                "correlation": 0.5,
                "mean_by_class": {1.0: 0.1, -1.0: -0.1},
                "std_by_class": {1.0: std, -1.0: std},
                "n_by_class": {1.0: 10, -1.0: 10},
            }

        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=5, eval_steps=100, train_batch_size=1, num_train_epochs=1,
            evaluate_fn=evaluate_fn, do_eval=True, convergent_score_threshold=None,
        )
        return train(
            model_with_gradiend=MockModelWithGradiend(), data=DataLoader(dataset, batch_size=1),
            training_args=args, callbacks=[SignalDiversityCallback()],
        )

    def test_constant_signal_aborts_the_run(self, tmp_path):
        with pytest.raises(SignalNotDiverseError):
            self._run(tmp_path, std=0.0)

    def test_diverse_signal_trains_normally(self, tmp_path):
        assert isinstance(self._run(tmp_path, std=0.2), str)


def test_probe_checkpoints_are_deleted_and_only_training_json_stays(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    tune_learning_rate(
        _FakeTrainer(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"),
        config=LRSearchConfig(probe_steps=100, confirm=False, center_window=False),
    )
    files = [f for _r, _d, fs in __import__("os").walk(tmp_path / "exp" / "lr_search") for f in fs]
    assert files and set(files) == {"training.json"}
    assert (tmp_path / "exp" / "lr_search.json").exists()


def test_probe_checkpoints_can_be_kept(tmp_path):
    import os

    from gradiend.trainer.core.lr_search import tune_learning_rate

    tune_learning_rate(
        _FakeTrainer(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"),
        config=LRSearchConfig(probe_steps=100, confirm=False, center_window=False, keep_probe_artifacts=True),
    )
    files = {f for _r, _d, fs in os.walk(tmp_path / "exp" / "lr_search") for f in fs}
    assert "model.safetensors" in files


def test_no_convergent_learning_rate_carries_the_trace():
    from gradiend.trainer.core.lr_search import LRSearchResult, NoConvergentLearningRate

    result = LRSearchResult("no_convergent_lr", None, [], bracket=(1e-8, 2e-8))
    err = NoConvergentLearningRate("boom", result)
    assert isinstance(err, RuntimeError) and err.result is result and "boom" in str(err)


def test_search_context_is_stored_next_to_the_result(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    tune_learning_rate(
        _FakeTrainer(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"),
        config=LRSearchConfig(probe_steps=100, confirm=False, center_window=False),
        metadata={"backend": "gradiend", "probe_steps": 100},
    )
    saved = json.loads((tmp_path / "exp" / "lr_search.json").read_text())
    assert saved["backend"] == "gradiend" and saved["probe_steps"] == 100 and saved["initial_lr"] == 1e-6
    assert saved["status"] == "converged" and saved["best_lr"] >= 1e-6


class TestRefineAndFullLengthSlow:
    """gemma-3-27b GRADIEND gender_en (slurm 4328877): 1e-6 slow but still rising at the end of the
    150-step probe, 2e-6 degrading, coarse grid empty in between -> no_convergent_lr after 3 probes."""

    SLOW_RISING = ([0.13, 0.19, 0.28, 0.36], [0.0, 0.002, 0.004, 0.005], [0.0, -0.005, -0.01, -0.012])
    # A genuinely degrading run: the encoder moves (means ~0.2..0.9) and its correlation then collapses.
    # (The real gemma-3-27b GRADIEND trace at 2e-6 -- means ~0.005, correlation sign flip -- was labelled
    # "degrading" by the old classifier although nothing had moved; its convergent window is 5e-6..5e-5.)
    DEGRADING = ([0.13, 0.06, -0.04, -0.12], [0.0, 0.2, 0.5, 0.8], [0.0, 0.3, 0.6, 0.9])

    def test_coarse_bracket_without_convergence_is_refined_on_the_fine_grid(self):
        calls = []

        def run(lr):
            calls.append(lr)
            if abs(lr - 1.5e-6) < 1e-12:
                return _result(*GOOD, converged=True)
            return _result(*(self.SLOW_RISING if lr < 1.5e-6 else self.DEGRADING))

        res = LRSearch(run, 1e-6, confirm=lambda lr: _result(*GOOD, converged=True)).search()
        assert res.status == "converged" and res.best_lr == pytest.approx(1.5e-6)

    def test_still_improving_slow_probe_gets_a_full_length_run(self):
        full_runs = []

        def run(lr):
            return _result(*(self.SLOW_RISING if lr <= 1e-6 else self.DEGRADING))

        def confirm(lr):
            full_runs.append(lr)
            return _result(*GOOD, converged=True)

        res = LRSearch(run, 1e-6, confirm=confirm).search()
        assert full_runs == [pytest.approx(1e-6)]
        assert res.status == "converged" and res.best_lr == pytest.approx(1e-6)
        assert "full budget" in next(p for p in res.probes if p.lr == pytest.approx(1e-6)).note

    def test_slow_probe_that_already_peaked_is_not_rerun(self):
        peaked = ([0.13, 0.23, 0.18, 0.14], [0.0, 0.01, 0.005, 0.004], [0.0, -0.01, -0.006, -0.005])
        full_runs = []

        def run(lr):
            return _result(*(peaked if lr <= 1e-6 else self.DEGRADING))

        res = LRSearch(run, 1e-6, confirm=lambda lr: full_runs.append(lr) or _result(*GOOD, converged=True)).search()
        assert full_runs == [] and res.status == "no_convergent_lr"

    def test_failed_full_length_run_still_reports_no_convergent_lr(self):
        def run(lr):
            return _result(*(self.SLOW_RISING if lr <= 1e-6 else self.DEGRADING))

        res = LRSearch(run, 1e-6, confirm=lambda lr: _result(*self.SLOW_RISING)).search()
        assert res.status == "no_convergent_lr"
        assert any("full budget: slow" in p.note for p in res.probes)


def test_expansion_tries_the_bound_instead_of_stopping_short_of_it():
    # gemma-3-270m AGIEND one-pole: slow at 0.5, x10 = 5.0 is above the 1.0 bound
    calls = []

    def run(lr):
        calls.append(lr)
        return _result(*FROZEN)

    res = LRSearch(run, 0.5, lr_bounds=(1e-12, 1.0)).search()
    assert calls == [pytest.approx(0.5), pytest.approx(1.0)]
    assert res.status == "no_convergent_lr"
    calls.clear()
    LRSearch(lambda lr: calls.append(lr) or _result(*COLLAPSE), 2e-12, lr_bounds=(1e-12, 1.0)).search()
    assert calls == [pytest.approx(2e-12), pytest.approx(1e-12)]


def test_sign_flip_of_an_untrained_encoder_is_frozen_not_degrading():
    # gemma-3-4b AGIEND one-pole M, lr 7e-4: class means ~0.003 throughout, correlation +0.1657 -> -0.1655
    r = ProbeResult(
        steps=[0, 25, 75, 125, 150], scores=[0.1657, -0.1655, -0.1656, -0.1656, -0.1655], converged=False,
        mean_by_class=[{"1.0": a, "-1.0": b, "0.0": n} for a, b, n in
                       [(0.0032, 0.0028, 0.0029), (0.0009, 0.0012, 0.0012), (0.0020, 0.0023, 0.0023),
                        (0.0020, 0.0023, 0.0023), (0.0020, 0.0023, 0.0023)]],
    )
    assert classify_probe(r) is RunState.FROZEN


def test_encoder_that_actually_moved_is_still_classified_by_its_correlation():
    r = ProbeResult(
        steps=[0, 25, 75], scores=[0.14, 0.82, 0.31], converged=False,
        mean_by_class=[{"1.0": a, "-1.0": b} for a, b in [(0.015, 0.011), (0.977, -0.918), (0.97, -0.99)]],
    )
    assert classify_probe(r) is RunState.DEGRADING


class TestProbeLogging:
    @staticmethod
    def _logging_fake(tmp_path):
        import logging

        fake = _FakeTrainer(tmp_path)
        original = fake.train

        def train(**kw):
            logging.getLogger("gradiend.trainer.core.training").info("probe step")
            logging.getLogger("gradiend.trainer.core.training").warning("probe did not converge")
            return original(**kw)

        fake.train = train
        return fake

    @pytest.mark.parametrize("verbose", [False, True])
    def test_probe_logs_are_quiet_unless_verbose(self, tmp_path, caplog, verbose):
        import logging

        from gradiend.trainer.core.lr_search import tune_learning_rate

        before = logging.getLogger("gradiend").level
        with caplog.at_level(logging.INFO, logger="gradiend"):
            tune_learning_rate(
                self._logging_fake(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"),
                config=LRSearchConfig(probe_steps=100, confirm=False, center_window=False, verbose=verbose),
            )
        probe_logs = [r for r in caplog.records if r.name == "gradiend.trainer.core.training"]
        assert bool(probe_logs) is verbose
        assert any(r.name.endswith("lr_search") for r in caplog.records)  # the search still reports each probe
        assert logging.getLogger("gradiend").level == before  # restored

    def test_logging_is_restored_when_a_probe_fails(self, tmp_path):
        import logging

        from gradiend.trainer.core.lr_search import tune_learning_rate

        fake = _FakeTrainer(tmp_path)
        fake.train = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        before = logging.getLogger("gradiend").level
        with pytest.raises(RuntimeError, match="boom"):
            tune_learning_rate(fake, 1e-6, experiment_dir=str(tmp_path / "exp"), config=LRSearchConfig(probe_steps=100))
        assert logging.getLogger("gradiend").level == before


class TestLRSearchConfig:
    def test_round_trips_through_training_arguments(self):
        from gradiend.trainer.core.arguments import TrainingArguments

        args = TrainingArguments(learning_rate="tune", lr_search_config=LRSearchConfig(probe_steps=40, lr_bounds=(1e-9, 1e-2)))
        restored = TrainingArguments.from_dict(json.loads(json.dumps(args.to_dict())))
        assert restored.learning_rate == "tune"
        assert restored.lr_search_config == LRSearchConfig(probe_steps=40, lr_bounds=(1e-9, 1e-2))

    @pytest.mark.parametrize("kwargs", [dict(resolution="medium"), dict(probe_steps=0)])
    def test_invalid_options_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            LRSearchConfig(**kwargs)

    def test_search_options_are_not_silently_forwarded_to_the_probes(self, tmp_path):
        from gradiend.trainer.core.lr_search import tune_learning_rate

        with pytest.raises(TypeError, match="LRSearchConfig"):
            tune_learning_rate(_FakeTrainer(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"), probe_steps=100)

    def test_learning_rate_must_be_a_number_or_tune(self):
        from gradiend.trainer.core.arguments import TrainingArguments

        with pytest.raises(TypeError):
            TrainingArguments(learning_rate="auto")
        with pytest.raises(TypeError):
            TrainingArguments(lr_search_config={"probe_steps": 10})

    def test_search_result_round_trips(self, tmp_path):
        res = LRSearch(lambda lr: _result(*GOOD, converged=lr >= 1e-6), 1e-8).search()
        assert LRSearchResult.from_dict(json.loads(json.dumps(res.to_dict()))).to_dict() == res.to_dict()


def _tunable_fake(tmp_path, **config):
    """The fake trainer with the real tune_learning_rate() / learning_rate="tune" logic of Trainer bound to it."""
    from gradiend.trainer.core.arguments import TrainingArguments
    from gradiend.trainer.trainer import Trainer

    class TunableFake(_FakeTrainer):
        run_id = None
        tune_learning_rate = Trainer.tune_learning_rate
        _tune_learning_rate = Trainer._tune_learning_rate
        _resolve_learning_rate = Trainer._resolve_learning_rate
        _resolve_experiment_dir = Trainer._resolve_experiment_dir

    fake = TunableFake(tmp_path)
    cfg = dict(probe_steps=100, confirm=False, max_probes=6, center_window=False)
    cfg.update(config)
    fake._training_args = TrainingArguments(
        experiment_dir=str(tmp_path / "exp"), learning_rate="tune", max_steps=500, eval_steps=50, use_cache=True,
        lr_search_config=LRSearchConfig(**cfg),
    )
    fake._tuned_learning_rate = None
    fake.lr_search_result = None
    fake.trains = 0
    original = fake.train

    def counting_train(**kw):
        fake.trains += 1
        return original(**kw)

    fake.train = counting_train
    return fake


class TestTrainerIntegration:
    def test_tune_learning_rate_keeps_the_result_for_train(self, tmp_path):
        fake = _tunable_fake(tmp_path)
        res = fake.tune_learning_rate(1e-8)
        assert res.status == "converged" and fake._tuned_learning_rate == res.best_lr >= 1e-6
        assert fake.lr_search_result is res
        assert (tmp_path / "exp" / "lr_search.json").exists()

    def test_config_fields_can_be_overridden_per_call(self, tmp_path):
        seen = []
        fake = _tunable_fake(tmp_path)
        original = fake.train
        fake.train = lambda **kw: (seen.append(kw["max_steps"]), original(**kw))[1]
        fake.tune_learning_rate(1e-6, probe_steps=60)
        assert set(seen) == {60}
        with pytest.raises(TypeError):
            fake.tune_learning_rate(1e-6, not_an_option=1)

    def test_a_failed_search_raises_with_the_trace(self, tmp_path):
        fake = _tunable_fake(tmp_path, lr_bounds=(1e-12, 1e-8))
        with pytest.raises(NoConvergentLearningRate) as err:
            fake.tune_learning_rate(1e-9)
        assert err.value.result.status == "no_convergent_lr" and fake._tuned_learning_rate is None

    def test_a_finished_search_is_reused_unless_forced_or_changed(self, tmp_path):
        fake = _tunable_fake(tmp_path)
        fake.tune_learning_rate(1e-8)
        trains = fake.trains
        again = fake.tune_learning_rate(1e-8)
        assert fake.trains == trains and again.status == "converged"
        fake.tune_learning_rate(1e-8, force=True)
        assert fake.trains > trains
        trains = fake.trains
        fake.tune_learning_rate(1e-8, probe_steps=80)  # different settings: the stored search no longer applies
        assert fake.trains > trains

    def test_use_cache_false_always_searches(self, tmp_path):
        fake = _tunable_fake(tmp_path)
        fake._training_args.use_cache = False
        fake.tune_learning_rate(1e-8)
        trains = fake.trains
        fake.tune_learning_rate(1e-8)
        assert fake.trains > trains

    def test_search_needs_evaluation(self, tmp_path):
        fake = _tunable_fake(tmp_path)
        fake._training_args.do_eval = False
        with pytest.raises(ValueError, match="needs evaluation"):
            fake.tune_learning_rate(1e-8)

    def test_train_resolves_tune_once_and_marks_the_arguments(self, tmp_path):
        from gradiend.trainer.core.arguments import TrainingArguments

        fake = _tunable_fake(tmp_path)
        args = TrainingArguments.from_dict(fake._training_args.to_dict())
        fake._resolve_learning_rate(args, str(tmp_path / "exp"), {})
        assert isinstance(args.learning_rate, float) and args.learning_rate >= 1e-6
        assert args.metadata["learning_rate_tuned"] is True
        trains = fake.trains
        again = TrainingArguments.from_dict(fake._training_args.to_dict())
        fake._resolve_learning_rate(again, str(tmp_path / "exp"), {})  # already tuned: no second search
        assert fake.trains == trains and again.learning_rate == args.learning_rate

    def test_an_explicit_learning_rate_wins(self, tmp_path):
        from gradiend.trainer.core.arguments import TrainingArguments

        fake = _tunable_fake(tmp_path)
        fake.tune_learning_rate(1e-8)
        trains = fake.trains
        args = TrainingArguments(learning_rate=3e-5)
        fake._resolve_learning_rate(args, None, {"learning_rate": 3e-5})
        assert args.learning_rate == 3e-5 and fake.trains == trains

    def test_a_previous_tune_call_is_used_for_a_numeric_learning_rate(self, tmp_path):
        from gradiend.trainer.core.arguments import TrainingArguments

        fake = _tunable_fake(tmp_path)
        res = fake.tune_learning_rate(1e-8)
        args = TrainingArguments(learning_rate=1e-5)
        fake._resolve_learning_rate(args, None, {})
        assert args.learning_rate == res.best_lr


def test_format_lr():
    assert [_format_lr(v) for v in (2e-5, 1.5e-6, 1e-4, 0.1, 10.0)] == ["2e-5", "1.5e-6", "1e-4", "1e-1", "1e1"]


def _example_script():
    """The simulated search of the documentation figure (scripts/plot_lr_search_example.py)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_lr_search_example.py"
    spec = importlib.util.spec_from_file_location("plot_lr_search_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simulated_search_finds_the_window_between_slow_and_overshoot():
    result = _example_script().simulated_search()
    states = {probe.lr: probe.state for probe in result.probes}
    assert states[1e-7] is RunState.FROZEN and states[2e-6] is RunState.SLOW and states[5e-5] is RunState.COLLAPSED
    assert [probe.lr for probe in result.probes if probe.state is RunState.CONVERGED] == [1e-5, 2e-5, 5e-6]
    assert result.status == "converged" and result.window == (5e-6, 2e-5) and result.best_lr == 1e-5


def test_probes_record_what_the_plot_needs():
    result = _example_script().simulated_search()
    converged = next(probe for probe in result.probes if probe.lr == 1e-5)
    slow = next(probe for probe in result.probes if probe.lr == 1e-6)
    assert converged.metric == "correlation" and converged.threshold == 0.5 and converged.class_mean_threshold == 0.5
    assert converged.min_class_mean >= 0.5 > slow.min_class_mean  # why the slow probe does not count: its class means are too small
    restored = LRSearchResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert [p.min_class_mean for p in restored.probes] == [p.min_class_mean for p in result.probes]


def test_searches_stored_without_the_evidence_still_load():
    old = {"status": "converged", "best_lr": 1e-5, "probes": [{"lr": 1e-5, "state": "converged", "final_score": 0.9, "best_score": 0.9}]}
    probe = LRSearchResult.from_dict(old).probes[0]
    assert probe.min_class_mean is None and probe.threshold is None


def test_result_plot_shows_scores_class_means_thresholds_and_choice(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    result = _example_script().simulated_search()
    output = tmp_path / "docs" / "search.png"
    fig = result.plot(output=str(output), show=False)
    top, bottom = fig.axes
    assert output.is_file() and top.get_xscale() == "log"
    legend = {text.get_text() for text in top.get_legend().get_texts()}
    assert {"convergent range", "frozen (too low)", "slow (too low)", "converged", "collapsed (too high)",
            "re-run for the full max_steps"} <= legend
    top_texts = [text.get_text() for text in top.texts]
    assert [text for text in top_texts if text.isdigit()] == [str(i) for i in range(1, 8)]  # probe order
    assert "convergence threshold" in top_texts and "class-mean threshold" in [text.get_text() for text in bottom.texts]
    assert "chosen 1e-5" in [text.get_text() for text in bottom.texts] and "chosen 1e-5" in top.get_title()
    assert "correlation" in top.get_ylabel() and "class mean" in bottom.get_ylabel()


def test_result_plot_without_class_means_or_a_choice():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    result = LRSearchResult("no_convergent_lr", None, [Probe(1e-6, RunState.FROZEN, 0.0, 0.05), Probe(1e-3, RunState.DIVERGED)])
    fig = result.plot(show=False)
    assert len(fig.axes) == 1 and "no_convergent_lr" in fig.axes[0].get_title()
    assert not any(text.get_text().startswith("chosen") for text in fig.axes[0].texts)
