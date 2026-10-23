"""
Automatic learning-rate search for GRADIEND-style training.

A short probe run is *classified* (not just scored) and the classification decides the next
learning rate: raise it when the run is too slow to learn, lower it when it overshoots or
degrades, stop when it converges. Once one LR is "too low" and a larger one is "too high" the
bracket is bisected in log space on a small grid of readable values (1/2/5 x 10^k by default)
until it is closed. If the bracket closes without a convergent run the search returns
``no_convergent_lr`` with the full trace instead of continuing forever: that verdict means no
learning rate can fix the run (e.g. the reconstruction target is not representable by the
latent), so the cause has to be found elsewhere.

The bracket closes when no untried grid value remains strictly inside it, so the grid alone
defines the finest spacing tried (``resolution="coarse"``: neighbours 2-2.5x apart, ``"fine"``:
1.33-1.67x). When a run converges the search additionally walks outward along the grid to find the
edges of the convergent window and returns the LR closest (in log space) to its centre, so the
result does not sit on the edge of the stable range.

The search logic (:class:`LRSearch`, :func:`classify_probe`) is pure and takes any callable
``run(lr) -> ProbeResult``; :func:`tune_learning_rate` is the adapter that drives a real
:class:`~gradiend.trainer.trainer.Trainer`. :class:`ProbeStopCallback` ends a probe as soon as it
is decisively collapsed or diverged instead of training the remaining steps.

Notes:

    * Probes are short. GRADIEND is schedule-sensitive (equal ``lr * steps`` does not give an

      equal optimisation), so pass ``confirm=True`` to re-run a converged probe at the
      configured full budget before accepting it.

    * The classification uses the package's own convergence decision

      (``convergence_info['converged']``, which includes the class-mean check) as ground truth for
      "converged". A run can have a high correlation at initialisation without being converged.
"""

from __future__ import annotations

import contextlib
import enum
import gc
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from gradiend.trainer.core.callbacks import TrainingCallback
from gradiend.trainer.core.signal_checks import SignalDiversityCallback
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "RunState",
    "ProbeResult",
    "Probe",
    "LRSearchResult",
    "LRSearchConfig",
    "NoConvergentLearningRate",
    "LRSearch",
    "classify_probe",
    "nice_lr_grid",
    "snap_to_grid",
    "grid_neighbor",
    "ProbeStopCallback",
    "tune_learning_rate",
]

COARSE_GRID: Tuple[float, ...] = (1.0, 2.0, 5.0)
FINE_GRID: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)


def _format_lr(value: float) -> str:
    """Compact learning-rate label: ``2e-5``, ``1.5e-5``."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


class RunState(str, enum.Enum):
    """What a probe run did. Decides the direction of the next learning rate."""

    CONVERGED = "converged"
    FROZEN = "frozen"  # nothing moved: LR too low
    SLOW = "slow"  # improving but not converged in the budget: LR too low
    DEGRADING = "degrading"  # improved/started well, then got worse: LR too high (slow drift)
    COLLAPSED = "collapsed"  # both poles on the same side / saturated: LR too high
    DIVERGED = "diverged"  # non-finite loss: LR too high

    @property
    def direction(self) -> int:
        """+1: raise the LR, -1: lower it, 0: stop."""
        if self in (RunState.FROZEN, RunState.SLOW):
            return +1
        if self in (RunState.DEGRADING, RunState.COLLAPSED, RunState.DIVERGED):
            return -1
        return 0


@dataclass
class ProbeResult:
    """The evidence one probe run leaves behind (all histories are aligned to ``steps``)."""

    steps: List[int]
    scores: List[float]  # convergence-metric history (correlation, or min_auc_n_o for one-pole)
    converged: bool
    threshold: Optional[float] = None
    metric: Optional[str] = None
    mean_by_class: List[Dict[str, float]] = field(default_factory=list)  # per eval, {"-1.0": m, "1.0": m}
    losses: List[float] = field(default_factory=list)
    class_mean_threshold: Optional[float] = None  # each class mean must reach this magnitude to converge

    @classmethod
    def from_training_stats(
        cls,
        stats: Mapping[str, Any],
        *,
        converged: bool = False,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
        losses: Optional[Sequence[float]] = None,
        class_mean_threshold: Optional[float] = None,
    ) -> "ProbeResult":
        """Build from a ``training_stats`` mapping (live in memory or loaded from ``training.json``).

        Step keys are ints while training runs and strings once written to JSON; both are accepted.
        """
        scores_by_step: Dict[int, float] = {int(k): float(v) for k, v in (stats.get("scores") or {}).items()}
        means_by_step: Dict[int, Mapping[Any, float]] = {
            int(k): v for k, v in (stats.get("mean_by_class") or {}).items()
        }
        steps = sorted(scores_by_step)
        return cls(
            steps=steps,
            scores=[scores_by_step[s] for s in steps],
            converged=converged,
            threshold=threshold,
            metric=metric,
            mean_by_class=[
                {str(k): float(v) for k, v in (means_by_step.get(s) or {}).items()} for s in steps
            ],
            losses=[float(x) for x in (losses or [])],
            class_mean_threshold=class_mean_threshold,
        )

    @classmethod
    def from_training_json(cls, path: str) -> "ProbeResult":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        info = data.get("convergence_info") or {}
        return cls.from_training_stats(
            data.get("training_stats") or {},
            converged=bool(info.get("converged", False)),
            threshold=info.get("threshold"),
            metric=info.get("convergence_metric"),
            losses=data.get("losses"),
            class_mean_threshold=info.get("convergent_mean_by_class_threshold"),
        )


def _signed_pole_means(means: Mapping[str, float]) -> Tuple[Optional[float], Optional[float]]:
    """(mean of the +1 class, mean of the -1 class) or None where absent."""
    pos = neg = None
    for key, val in means.items():
        try:
            label = float(key)
        except (TypeError, ValueError):
            continue
        if label > 0:
            pos = float(val)
        elif label < 0:
            neg = float(val)
    return pos, neg


def _probe_evidence(result: ProbeResult) -> Dict[str, Any]:
    """Metric, thresholds and the weaker class mean at the best evaluation of a probe (what the plot shows)."""
    weaker = None
    if result.scores and result.mean_by_class:
        best = max(range(len(result.scores)), key=result.scores.__getitem__)
        pos, neg = _signed_pole_means(result.mean_by_class[min(best, len(result.mean_by_class) - 1)])
        if pos is not None and neg is not None:
            weaker = min(abs(pos), abs(neg))
    return {"metric": result.metric, "threshold": result.threshold, "min_class_mean": weaker,
            "class_mean_threshold": result.class_mean_threshold}


def classify_probe(
    result: ProbeResult,
    *,
    frozen_range: float = 0.05,
    degrade_margin: float = 0.15,
    collapse_abs_mean: float = 0.5,
) -> RunState:
    """Classify a probe run.

    Args:
        frozen_range: total movement (max - min) of the score history, and of the class-mean gap,
            below which nothing is considered to have moved.
        degrade_margin: how far the final score must sit below the best score to count as degrading.
        collapse_abs_mean: both poles on the same side, each with ``|mean|`` above this, is a collapse.
    """
    if result.losses and not all(math.isfinite(x) for x in result.losses):
        return RunState.DIVERGED
    if result.scores and not all(math.isfinite(x) for x in result.scores):
        return RunState.DIVERGED
    if result.converged:
        return RunState.CONVERGED

    scores = result.scores
    score_range = (max(scores) - min(scores)) if len(scores) >= 2 else 0.0
    gaps = []
    for means in result.mean_by_class:
        pos, neg = _signed_pole_means(means)
        if pos is not None and neg is not None:
            gaps.append(pos - neg)
    gap_range = (max(gaps) - min(gaps)) if len(gaps) >= 2 else 0.0

    # Frozen is judged BEFORE collapse: an untrained encoder can already sit with both poles
    # saturated on one side (gemma-3-4b ACTIEND starts at F=-0.62, M=-0.56). A run that never moved
    # from there is "LR too low", not "LR too high"; calling it collapsed sends the search down forever.
    if score_range < frozen_range and gap_range < frozen_range:
        return RunState.FROZEN

    # Untrained encoder: every class mean stays tiny for the whole probe, so the correlation is computed
    # on ~1e-3 values and can flip sign without anything having been learned (gemma-3-4b AGIEND one-pole
    # M: outputs ~0.003 at every LR from 1e-4 to 1e-3, correlation +0.1657 -> -0.1655 at 7e-4, which was
    # read as "degrading" and sent the search DOWN). Nothing has moved, so the LR is too low.
    amplitude = 0.0
    for means in result.mean_by_class:
        for value in means.values():
            amplitude = max(amplitude, abs(value))
    # ... AND the pole gap never opens: a run whose class means are small but whose gap grows steadily
    # (gemma-3-27b GRADIEND at 1e-6: gap 0 -> 0.017, correlation 0.13 -> 0.36) is genuine slow progress.
    max_gap = max((abs(g) for g in gaps), default=None)
    if max_gap is not None and amplitude < frozen_range and max_gap < 0.005:
        return RunState.FROZEN

    # Collapse: the +1 and -1 classes end on the same side, saturated, AND the pole gap did not open
    # in the right direction relative to the start (a run separating the poles while still on one
    # side of zero is learning, not collapsing).
    # A run that STARTED saturated on one side (untrained gemma-3-4b ACTIEND) only counts as collapsed
    # if training actually moved the poles there; creeping a few hundredths around its initial state
    # (lr 2e-6: F -0.62 -> -0.64, M -0.56 -> -0.57, corr -0.34 -> -0.40) is "too low", not "too high".
    # Moved = a pole shifted, or the score fell, by at least ``degrade_margin``.
    def _is_collapsed(p, n):
        return p * n > 0 and min(abs(p), abs(n)) >= collapse_abs_mean

    if result.mean_by_class:
        pos, neg = _signed_pole_means(result.mean_by_class[-1])
        pos0, neg0 = _signed_pole_means(result.mean_by_class[0])
        if pos is not None and neg is not None:
            gap_opened = len(gaps) >= 2 and gaps[-1] > gaps[0] + frozen_range
            started_collapsed = pos0 is not None and neg0 is not None and _is_collapsed(pos0, neg0)
            pole_move = (
                max(abs(pos - pos0), abs(neg - neg0)) if pos0 is not None and neg0 is not None else float("inf")
            )
            score_drop = (max(scores) - scores[-1]) if scores else 0.0
            moved_into_it = (
                not started_collapsed or pole_move >= degrade_margin or score_drop >= degrade_margin
            )
            if _is_collapsed(pos, neg) and not gap_opened and moved_into_it:
                return RunState.COLLAPSED

    if len(scores) >= 2 and scores[-1] < max(scores) - degrade_margin:
        return RunState.DEGRADING
    # Moved, did not degrade, did not converge: still learning when the budget ended. A run whose
    # class-pole gap is *shrinking* is drifting to a common pole even if the score has not dropped yet.
    if len(gaps) >= 2 and gaps[-1] < gaps[0] - frozen_range:
        return RunState.DEGRADING
    return RunState.SLOW


def nice_lr_grid(lo: float, hi: float, resolution: str = "coarse") -> List[float]:
    """Readable learning rates ``m * 10^k`` strictly inside ``(lo, hi)``, ascending."""
    if resolution not in ("coarse", "fine"):
        raise ValueError(f"resolution must be 'coarse' or 'fine', got {resolution!r}")
    mantissas = COARSE_GRID if resolution == "coarse" else FINE_GRID
    out: List[float] = []
    k_lo = math.floor(math.log10(lo)) - 1
    k_hi = math.ceil(math.log10(hi)) + 1
    for k in range(k_lo, k_hi + 1):
        for m in mantissas:
            value = float(f"{m}e{k}")
            if lo * (1 + 1e-9) < value < hi * (1 - 1e-9):
                out.append(value)
    return sorted(set(out))


def snap_to_grid(value: float, resolution: str = "coarse") -> float:
    """Nearest readable learning rate in log space."""
    candidates = nice_lr_grid(value / 10.0, value * 10.0, resolution)
    return min(candidates, key=lambda c: abs(math.log(c) - math.log(value)))


def grid_neighbor(lr: float, direction: int, resolution: str = "coarse") -> float:
    """The adjacent readable learning rate above (``direction > 0``) or below (``< 0``) ``lr``."""
    if direction > 0:
        return nice_lr_grid(lr, lr * 10.5, resolution)[0]
    if direction < 0:
        return nice_lr_grid(lr / 10.5, lr, resolution)[-1]
    raise ValueError("direction must be non-zero")


class ProbeStopCallback(TrainingCallback):
    """End a probe as soon as its fate is decided.

    After every new evaluation the run so far is classified with the same rules as the final
    verdict; if the state is one of ``stop_states`` the training loop is told to stop
    (``control['should_stop']``) and the epoch-end hooks still write the checkpoint and
    ``training.json``. Only *decisive* states stop by default: a collapse (both poles saturated on
    one side) and a non-finite loss cannot recover, whereas a slow or drifting run might. A
    non-finite loss stops immediately, without waiting for an evaluation.

    Args:
        stop_states: states that end the run.
        min_evals: number of evaluations (including the initial one) required before judging.
    """

    def __init__(
        self,
        stop_states: Sequence[RunState] = (RunState.COLLAPSED, RunState.DIVERGED),
        *,
        min_evals: int = 2,
        classify: Callable[[ProbeResult], RunState] = classify_probe,
    ) -> None:
        self.stop_states = frozenset(stop_states)
        self.min_evals = int(min_evals)
        self.classify = classify
        self.stopped_state: Optional[RunState] = None
        self.stopped_step: Optional[int] = None
        self._seen_evals = 0

    def _stop(self, control: Dict[str, Any], state: RunState, step: int) -> None:
        self.stopped_state, self.stopped_step = state, int(step)
        control["should_stop"] = True
        logger.info("LR probe stopped early at step %s: %s", step, state.value)

    def on_step_end(self, step, loss, model, config, **kwargs):
        control = kwargs.get("control")
        if control is None or self.stopped_state is not None:
            return None
        if loss is not None and not math.isfinite(float(loss)):
            self._stop(control, RunState.DIVERGED, step)
            return None
        stats = kwargs.get("training_stats") or {}
        n_evals = len(stats.get("scores") or {})
        if n_evals == self._seen_evals:
            return None  # no evaluation happened at this step
        self._seen_evals = n_evals
        if n_evals < self.min_evals:
            return None
        state = self.classify(ProbeResult.from_training_stats(stats))
        if state in self.stop_states:
            self._stop(control, state, step)
        return None


@dataclass
class Probe:
    lr: float
    state: RunState
    final_score: Optional[float] = None
    best_score: Optional[float] = None
    confirmed: Optional[bool] = None
    steps_run: Optional[int] = None  # last evaluated step (shows early stops)
    note: str = ""
    metric: Optional[str] = None  # convergence metric of ``best_score`` (e.g. "correlation")
    threshold: Optional[float] = None  # ``best_score`` must reach this to converge
    min_class_mean: Optional[float] = None  # weaker class mean (magnitude) at the best evaluation
    class_mean_threshold: Optional[float] = None  # ... and this must reach ``class_mean_threshold``

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class LRSearchResult:
    status: str  # "converged" | "no_convergent_lr" | "budget_exhausted"
    best_lr: Optional[float]
    probes: List[Probe]
    bracket: Optional[Tuple[float, float]] = None  # (highest "too low", lowest "too high")
    window: Optional[Tuple[float, float]] = None  # (lowest, highest) convergent LR probed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "best_lr": self.best_lr,
            "bracket": list(self.bracket) if self.bracket else None,
            "window": list(self.window) if self.window else None,
            "probes": [p.to_dict() for p in self.probes],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LRSearchResult":
        """Inverse of :meth:`to_dict` (e.g. for a stored ``lr_search.json``)."""
        probes = [
            Probe(
                lr=p["lr"], state=RunState(p["state"]), final_score=p.get("final_score"), best_score=p.get("best_score"),
                confirmed=p.get("confirmed"), steps_run=p.get("steps_run"), note=p.get("note", ""),
                metric=p.get("metric"), threshold=p.get("threshold"), min_class_mean=p.get("min_class_mean"),
                class_mean_threshold=p.get("class_mean_threshold"),
            )
            for p in d.get("probes", [])
        ]
        bracket, window = d.get("bracket"), d.get("window")
        return cls(d["status"], d.get("best_lr"), probes, tuple(bracket) if bracket else None, tuple(window) if window else None)

    def plot(self, *, output: Optional[str] = None, show: bool = True, title: Optional[str] = None) -> Any:
        """Plot what the search did.

        Every probe (a short training run) is one dot at its learning rate, numbered in the order it ran and coloured by
        its state: blue too low, green converged, orange/red too high. The top panel shows the convergence score
        (the correlation of the encoded value with the class) the probe reached; the bottom panel the weaker of the two
        class means. A probe converges only if both reach their threshold (dotted lines). The green band is the
        convergent range, the dashed line the chosen learning rate, and a ring marks a probe that was re-run for the full
        ``max_steps``. Needs matplotlib.

        Args:
            output: Optional path to save the figure to.
            show: Whether to call ``plt.show()``.
            title: Optional title (default: the search status and the chosen learning rate).

        Returns:
            The matplotlib figure.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter, LogLocator
        except ImportError as exc:  # pragma: no cover - matplotlib is a dependency of the plotting extras
            raise ImportError("LRSearchResult.plot needs matplotlib: pip install matplotlib") from exc

        colors = {
            RunState.FROZEN: ("#9ecae1", "frozen (too low)"),
            RunState.SLOW: ("#1f77b4", "slow (too low)"),
            RunState.CONVERGED: ("#2ca02c", "converged"),
            RunState.DEGRADING: ("#fdae6b", "degrading (too high)"),
            RunState.COLLAPSED: ("#e6550d", "collapsed (too high)"),
            RunState.DIVERGED: ("#a50f15", "diverged (too high)"),
        }

        def first(values):
            return next((v for v in values if v is not None), None)

        metric = first(p.metric for p in self.probes) or "score"
        threshold = first(p.threshold for p in self.probes)
        mean_threshold = first(p.class_mean_threshold for p in self.probes)
        with_means = any(p.min_class_mean is not None for p in self.probes)

        scores = [p.best_score for p in self.probes if p.best_score is not None]
        score_floor = min([0.5 if threshold is None else threshold - 0.1, *scores]) - 0.1
        panels = [("best " + metric + " in the probe", lambda p: p.best_score, score_floor, 1.18, threshold,
                   "convergence threshold")]
        if with_means:
            panels.append(("weaker class mean |m|\nat that evaluation", lambda p: p.min_class_mean, 0.0, 1.1, mean_threshold,
                           "class-mean threshold"))
        fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False, figsize=(8.8, 2.9 * len(panels) + 0.5),
                                 gridspec_kw={"height_ratios": [3, 2][: len(panels)], "hspace": 0.12})
        for row, (label, value, floor, top, line, line_label) in enumerate(panels):
            ax = axes[row][0]
            if self.window:
                ax.axvspan(*self.window, color="#2ca02c", alpha=0.12, label="convergent range", zorder=0)
            if line is not None:
                ax.axhline(line, color="0.45", ls=":", lw=1.2, zorder=1)
                ax.annotate(line_label, (0.005, line), xycoords=("axes fraction", "data"), textcoords="offset points",
                            xytext=(3, 3), fontsize=8, color="0.35")
            points = [(p, floor if value(p) is None else value(p)) for p in self.probes]
            for state, (color, state_label) in colors.items():
                xs = [(p.lr, y) for p, y in points if p.state == state and (value(p) is not None or row == 0)]
                if xs:
                    ax.scatter(*zip(*xs), s=70, color=color, edgecolor="white", label=state_label, zorder=3)
            if row == 0:
                confirmed = [(p.lr, y) for p, y in points if p.confirmed]
                if confirmed:
                    ax.scatter(*zip(*confirmed), s=190, facecolors="none", edgecolors="black", linewidths=1.2,
                               label="re-run for the full max_steps", zorder=2)
                for order, (p, y) in enumerate(points, start=1):
                    ax.annotate(str(order), (p.lr, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
            if self.best_lr is not None:
                ax.axvline(self.best_lr, color="black", ls="--", lw=1, zorder=0)
                if row == len(panels) - 1:
                    ax.annotate(f"chosen {_format_lr(self.best_lr)}", (self.best_lr, 0.03), xycoords=("data", "axes fraction"),
                                textcoords="offset points", xytext=(4, 0), ha="left", va="bottom", fontsize=9)
            ax.set_ylim(floor, top)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        ax = axes[-1][0]
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))  # the 1/2/5 grid of the search
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_lr(value)))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("learning rate")
        axes[0][0].set_title(title or (f"Learning-rate search: {self.status}"
                                       + (f", chosen {_format_lr(self.best_lr)}" if self.best_lr else "")))
        axes[0][0].legend(loc="center left", bbox_to_anchor=(1.01, 0.0 if len(panels) > 1 else 0.5), fontsize=8, frameon=False)
        if output:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            fig.savefig(output, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return fig


class NoConvergentLearningRate(RuntimeError):
    """The search ended without a convergent learning rate; ``result`` carries the full trace."""

    def __init__(self, message: str, result: "LRSearchResult") -> None:
        super().__init__(message)
        self.result = result


class LRSearch:
    """Bracket-and-bisect controller over a probe function.

    Args:
        run: ``run(lr) -> ProbeResult``; must be deterministic enough that repeating an LR is not needed.
        initial_lr: first guess (use a readable value such as ``1e-6``).
        factor: expansion step while no bracket exists (10 by default).
        min_ratio: optional extra stop: the bracket is also closed once ``high / low <= min_ratio``.
            ``None`` (default) closes it only when no untried grid value lies strictly inside, so the
            grid (``resolution``) alone sets the finest spacing.
        max_probes: budget for the bracketing phase. On exhaustion the status is ``budget_exhausted``.
        resolution: ``"coarse"`` (1/2/5) or ``"fine"`` (1/1.5/2/3/5/7) grid for bisection candidates and
            window edges.
        lr_bounds: never propose an LR outside ``(lo, hi)``.
        confirm: optional ``confirm(lr) -> ProbeResult`` re-run (e.g. at the full budget) of a converged
            probe. A failed confirmation is classified like any probe and the search continues.
        center_window: once a probe converges, walk outward along the grid until the neighbours stop
            converging and return the convergent LR closest (in log space) to the window's centre,
            instead of the first hit, which may sit on the edge of the stable range.
        max_window_probes: extra probes allowed for that walk (separate from ``max_probes``), split
            evenly between the two sides (upward gets the odd one); the worst case is therefore
            ``max_probes + max_window_probes`` probes.
        refine_fine: when a ``"coarse"`` search closes its bracket without a convergent LR, continue
            bisecting inside that bracket on the ``"fine"`` grid before giving up (gemma-3-27b GRADIEND:
            bracket 1e-6 slow / 2e-6 degrading had no coarse value in between).
        max_refine_probes: probe budget for that fine refinement.
        confirm_slow: before reporting ``no_convergent_lr``, re-run the highest-LR ``slow`` probe that
            was still at its best score when the probe ended through ``confirm`` (the full budget):
            a short probe cannot tell "too low" from "converges, but needs the full run".
    """

    def __init__(
        self,
        run: Callable[[float], ProbeResult],
        initial_lr: float,
        *,
        factor: float = 10.0,
        min_ratio: Optional[float] = None,
        max_probes: int = 12,
        resolution: str = "coarse",
        lr_bounds: Tuple[float, float] = (1e-12, 10.0),
        confirm: Optional[Callable[[float], ProbeResult]] = None,
        classify: Callable[[ProbeResult], RunState] = classify_probe,
        center_window: bool = True,
        max_window_probes: int = 4,
        refine_fine: bool = True,
        max_refine_probes: int = 3,
        confirm_slow: bool = True,
    ) -> None:
        if initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        if factor <= 1:
            raise ValueError("factor must be > 1")
        if resolution not in ("coarse", "fine"):
            raise ValueError(f"resolution must be 'coarse' or 'fine', got {resolution!r}")
        if min_ratio is not None and min_ratio <= 1:
            raise ValueError("min_ratio must be > 1 (or None)")
        self.run = run
        self.initial_lr = self._norm(initial_lr)
        self.factor = float(factor)
        self.min_ratio = None if min_ratio is None else float(min_ratio)
        self.max_probes = int(max_probes)
        self.resolution = resolution
        self.lr_bounds = lr_bounds
        self.confirm = confirm
        self.classify = classify
        self.center_window = bool(center_window)
        self.max_window_probes = int(max_window_probes)
        self.refine_fine = bool(refine_fine)
        self.max_refine_probes = int(max_refine_probes)
        self.confirm_slow = bool(confirm_slow)
        self.probes: List[Probe] = []

    @staticmethod
    def _norm(lr: float) -> float:
        """Canonical float for an LR, so 1e-6 / 10 and the grid value 1e-7 compare equal."""
        return float(f"{lr:.9g}")

    # ------------------------------------------------------------------ bracket
    def _tried(self) -> Dict[float, Probe]:
        return {self._norm(p.lr): p for p in self.probes}

    def _bracket(self) -> Tuple[Optional[float], Optional[float]]:
        """(highest 'too low' LR below the lowest 'too high' LR, lowest 'too high' LR)."""
        highs = [p.lr for p in self.probes if p.state.direction < 0]
        high = min(highs) if highs else None
        lows = [p.lr for p in self.probes if p.state.direction > 0 and (high is None or p.lr < high)]
        low = max(lows) if lows else None
        return low, high

    def _next_lr(self) -> Optional[float]:
        low, high = self._bracket()
        tried = self._tried()
        lo_b, hi_b = self.lr_bounds
        if low is None and high is None:
            return self.initial_lr
        # Expansion clamps to the bound instead of stopping short of it: gemma-3-270m AGIEND one-pole
        # was "slow" at 0.5, the x10 step (5.0) lies above the 1.0 bound, and the search used to end
        # after that single probe without ever trying 1.0.
        if high is None:  # only "too low" so far: expand upward
            nxt = self._norm(min(low * self.factor, hi_b))
            return nxt if nxt > low and nxt not in tried else None
        if low is None:  # only "too high" so far: expand downward
            nxt = self._norm(max(high / self.factor, lo_b))
            return nxt if nxt < high and nxt not in tried else None
        # bracket exists: bisect on the readable grid
        if self.min_ratio is not None and high / low <= self.min_ratio * (1 + 1e-9):
            return None
        inside = [c for c in nice_lr_grid(low, high, self.resolution) if self._norm(c) not in tried]
        if not inside:
            return None
        target = math.sqrt(low * high)
        return min(inside, key=lambda c: abs(math.log(c) - math.log(target)))

    # -------------------------------------------------------------------- probe
    def _probe(self, lr: float) -> Probe:
        result = self.run(lr)
        state = self.classify(result)
        best = max(result.scores) if result.scores else None
        final = result.scores[-1] if result.scores else None
        probe = Probe(
            lr=lr, state=state, final_score=final, best_score=best,
            steps_run=result.steps[-1] if result.steps else None, **_probe_evidence(result),
        )
        if state is RunState.CONVERGED and self.confirm is not None:
            confirmed = self.confirm(lr)
            cstate = self.classify(confirmed)
            probe.confirmed = cstate is RunState.CONVERGED
            if not probe.confirmed:
                probe.state = cstate
                probe.note = f"probe converged but the confirmation run was {cstate.value}"
                probe.final_score = confirmed.scores[-1] if confirmed.scores else None
                probe.best_score = max(confirmed.scores) if confirmed.scores else None
                for name, value in _probe_evidence(confirmed).items():
                    setattr(probe, name, value)
        logger.info(
            "lr_search: lr=%g -> %s (final=%s best=%s)%s",
            lr, probe.state.value, probe.final_score, probe.best_score,
            f" [{probe.note}]" if probe.note else "",
        )
        return probe

    # ------------------------------------------------------------------- window
    def _walk_window(self, first_lr: float, direction: int, budget: int) -> Tuple[List[float], int]:
        """Follow the grid from ``first_lr`` while neighbours keep converging; return (LRs, budget left)."""
        found: List[float] = []
        edge = first_lr
        lo_b, hi_b = self.lr_bounds
        while True:
            nxt = self._norm(grid_neighbor(edge, direction, self.resolution))
            if not lo_b <= nxt <= hi_b:
                break
            known = self._tried().get(nxt)
            if known is None:
                if budget <= 0:
                    break
                known = self._probe(nxt)
                self.probes.append(known)
                budget -= 1
            if known.state is not RunState.CONVERGED:
                break
            found.append(nxt)
            edge = nxt
        return found, budget

    def _converged_result(self, first_lr: float) -> LRSearchResult:
        convergent = [first_lr]
        if self.center_window:
            # each side gets its own budget: a walk that spends everything upward would return the centre of
            # a truncated window (found on Qwen-2B: 1e-6 .. 2e-5 probed only upward)
            budgets = {+1: (self.max_window_probes + 1) // 2, -1: self.max_window_probes // 2}
            for direction in (+1, -1):
                found, _ = self._walk_window(first_lr, direction, budgets[direction])
                convergent.extend(found)
        lo, hi = min(convergent), max(convergent)
        centre = 0.5 * (math.log(lo) + math.log(hi))
        # ties go to the lower LR: the upper edge of the window is where runs collapse
        best = min(convergent, key=lambda c: (round(abs(math.log(c) - centre), 9), c))
        return LRSearchResult("converged", best, list(self.probes), window=(lo, hi))

    def _still_improving_slow_probe(self) -> Optional[Probe]:
        """Highest-LR ``slow`` probe below the bracket's high end whose score ended at its best."""
        _, high = self._bracket()
        candidates = [
            p for p in self.probes
            if p.state is RunState.SLOW and p.confirmed is None
            and (high is None or p.lr < high)
            and p.final_score is not None and p.best_score is not None
            and p.final_score >= p.best_score - 0.02
        ]
        return max(candidates, key=lambda p: p.lr) if candidates else None

    def search(self) -> LRSearchResult:
        while len(self.probes) < self.max_probes:
            lr = self._next_lr()
            if lr is None:
                break
            probe = self._probe(lr)
            self.probes.append(probe)
            if probe.state is RunState.CONVERGED:
                return self._converged_result(probe.lr)
        bracket_closed = self._next_lr() is None
        if bracket_closed and self.refine_fine and self.resolution == "coarse":
            low, high = self._bracket()
            if low is not None and high is not None:
                self.resolution = "fine"
                logger.info("lr_search: coarse bracket (%g, %g) closed without convergence; refining on the fine grid", low, high)
                for _ in range(self.max_refine_probes):
                    lr = self._next_lr()
                    if lr is None:
                        break
                    probe = self._probe(lr)
                    self.probes.append(probe)
                    if probe.state is RunState.CONVERGED:
                        return self._converged_result(probe.lr)
        if self._next_lr() is None and self.confirm_slow and self.confirm is not None:
            slow = self._still_improving_slow_probe()
            if slow is not None:
                logger.info("lr_search: lr=%g was still improving when its probe ended; running it at the full budget", slow.lr)
                confirmed = self.confirm(slow.lr)
                cstate = self.classify(confirmed)
                slow.confirmed = cstate is RunState.CONVERGED
                slow.note = f"slow probe re-run at the full budget: {cstate.value}"
                slow.final_score = confirmed.scores[-1] if confirmed.scores else slow.final_score
                slow.best_score = max(confirmed.scores) if confirmed.scores else slow.best_score
                logger.info("lr_search: lr=%g at the full budget -> %s", slow.lr, cstate.value)
                if slow.confirmed:
                    slow.state = RunState.CONVERGED
                    return LRSearchResult("converged", slow.lr, list(self.probes), window=(slow.lr, slow.lr))
        low, high = self._bracket()
        bracket = (low, high) if low is not None and high is not None else None
        exhausted = len(self.probes) >= self.max_probes and self._next_lr() is not None
        status = "budget_exhausted" if exhausted else "no_convergent_lr"
        return LRSearchResult(status, None, list(self.probes), bracket)


# --------------------------------------------------------------------- trainer adapter
@dataclass
class LRSearchConfig:
    """Options of the learning-rate search.

    Set it as ``TrainingArguments(learning_rate="tune", lr_search_config=LRSearchConfig(...))`` or pass single fields
    to ``trainer.tune_learning_rate(...)``. ``factor``, ``min_ratio``, ``max_probes``, ``resolution``, ``lr_bounds``,
    ``center_window`` and ``max_window_probes`` are those of :class:`LRSearch`.
    """

    probe_steps: Optional[int] = 150
    """Steps of each short probe run. ``None`` trains every probe for the full ``max_steps``."""

    probe_eval_steps: Optional[int] = None
    """Evaluation interval inside a probe (default: ``probe_steps // 6``)."""

    confirm: bool = True
    """Re-run a converged probe at the full ``max_steps`` before accepting it (one extra full run)."""

    early_stop: bool = True
    """End a probe as soon as it is decisively collapsed or diverged."""

    max_probes: int = 12
    resolution: str = "coarse"
    factor: float = 10.0
    min_ratio: Optional[float] = None
    lr_bounds: Tuple[float, float] = (1e-12, 10.0)
    center_window: bool = True
    max_window_probes: int = 4

    probe_seeds: int = 1
    """Seeds a probe may try. Some regimes (the GRADIEND pair) drift to a common pole for one seed and converge for
    another, so a single-seed probe can report ``collapsed`` or ``degrading`` at a learning rate that multi-seed
    training (``max_seeds`` > 1) converges at. With ``probe_seeds`` > 1 a probe that ends ``collapsed`` or ``degrading``
    is repeated with the next seeds (``seed + 1``, ...) and the best run is kept (converged first, then the highest
    score). ``frozen``, ``slow`` and ``diverged`` probes are not repeated: another seed does not change their verdict."""

    keep_probe_artifacts: bool = False
    """Keep the probe checkpoints (default: only each probe's ``training.json``)."""

    verbose: bool = False
    """Show the full training log of every probe. By default the search logs one line per probe."""

    def __post_init__(self) -> None:
        self.lr_bounds = (float(self.lr_bounds[0]), float(self.lr_bounds[1]))
        if self.resolution not in ("coarse", "fine"):
            raise ValueError(f"resolution must be 'coarse' or 'fine', got {self.resolution!r}")
        if self.probe_steps is not None and self.probe_steps < 1:
            raise ValueError(f"probe_steps must be >= 1 or None, got {self.probe_steps}")
        if self.probe_seeds < 1:
            raise ValueError(f"probe_seeds must be >= 1, got {self.probe_seeds}")


def tune_learning_rate(
    trainer: Any,
    initial_lr: float,
    *,
    experiment_dir: str,
    config: Optional[LRSearchConfig] = None,
    precheck: Optional[Callable[[], None]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **train_overrides: Any,
) -> LRSearchResult:
    """Search the learning rate of ``trainer`` and write ``lr_search.json`` into ``experiment_dir``.

    Most users call :meth:`Trainer.tune_learning_rate() <gradiend.trainer.trainer.Trainer.tune_learning_rate>` or set
    ``TrainingArguments(learning_rate="tune")`` instead; this is the function behind both.

    Each probe trains one seed into ``<experiment_dir>/lr_search/<probe|confirm>_<lr>`` (no cache
    reuse), always from the trainer's original base model, with ``max_steps=config.probe_steps``. With
    ``config.confirm`` a converged probe is re-run at the trainer's configured ``max_steps`` before it
    is accepted. With ``config.early_stop`` a probe ends as soon as it is decisively collapsed or
    diverged (:class:`ProbeStopCallback`). ``precheck`` runs once before the first probe (use
    :func:`~gradiend.trainer.core.signal_checks.assert_signal_diverse` on stacked inputs) for extra checks.
    Probe checkpoints are deleted after each probe (only ``training.json`` is kept) unless
    ``config.keep_probe_artifacts``. ``metadata`` (e.g. the first guess and probe length) is stored next to the
    result in ``lr_search.json`` so a later run can tell whether a finished search still applies. The search options
    are those of :class:`LRSearchConfig` (``probe_seeds`` > 1 repeats a ``collapsed`` or ``degrading`` probe with
    the next seeds); ``train_overrides`` are passed to every probe's ``trainer.train``.
    """
    cfg = config or LRSearchConfig()
    misplaced = sorted(set(train_overrides) & {f.name for f in fields(LRSearchConfig)})
    if misplaced:
        raise TypeError(f"tune_learning_rate got search option(s) {misplaced}: pass them as config=LRSearchConfig(...)")
    os.makedirs(experiment_dir, exist_ok=True)
    if precheck is not None:
        precheck()

    base_args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
    full_steps = getattr(base_args, "max_steps", None)
    full_eval = getattr(base_args, "eval_steps", None)

    base_seed = int(getattr(base_args, "seed", 0) or 0)
    seed_dependent = (RunState.COLLAPSED, RunState.DEGRADING)

    def _one_seed(lr: float, steps: Optional[int], eval_steps: Optional[int], tag: str, k: int) -> ProbeResult:
        probe_dir = os.path.join(experiment_dir, "lr_search", f"{tag}_{lr:.3g}" + (f"_seed{k}" if k else ""))
        overrides: Dict[str, Any] = dict(train_overrides)
        overrides.update(
            learning_rate=lr, experiment_dir=probe_dir, max_seeds=1, use_cache=False,
        )
        if k:
            overrides["seed"] = base_seed + k
        if steps is not None:
            overrides["max_steps"] = int(steps)
        if eval_steps is not None:
            overrides["eval_steps"] = int(eval_steps)
        # the diversity guard is free (it reads statistics every evaluation computes) and stays on
        overrides["callbacks"] = [SignalDiversityCallback()] + ([ProbeStopCallback()] if cfg.early_stop else [])
        with _isolated_probe(trainer), _quiet_logs(not cfg.verbose):
            trainer.train(**overrides)
            result = _read_probe(probe_dir)
        if not cfg.keep_probe_artifacts:
            _prune_probe_dir(probe_dir)
        return result

    def _train_and_read(lr: float, steps: Optional[int], eval_steps: Optional[int], tag: str) -> ProbeResult:
        best: Optional[ProbeResult] = None
        for k in range(cfg.probe_seeds):
            result = _one_seed(lr, steps, eval_steps, tag, k)
            if best is None or _better_probe(result, best):
                best = result
            if classify_probe(best) not in seed_dependent:
                break  # converged, frozen, slow or diverged: another seed cannot change that verdict
            if k + 1 < cfg.probe_seeds:
                logger.info("lr_search: lr=%g seed %d ended %s; trying the next seed", lr, k,
                            classify_probe(result).value)
        assert best is not None
        return best

    eval_steps = cfg.probe_eval_steps
    if eval_steps is None and cfg.probe_steps is not None:
        eval_steps = max(1, int(cfg.probe_steps) // 6)

    def run(lr: float) -> ProbeResult:
        return _train_and_read(lr, cfg.probe_steps, eval_steps, "probe")

    confirm_fn = None
    if cfg.confirm and cfg.probe_steps is not None and cfg.probe_steps != full_steps:
        def confirm_fn(lr: float) -> ProbeResult:  # noqa: E306
            return _train_and_read(lr, full_steps, full_eval, "confirm")

    search = LRSearch(
        run, initial_lr, factor=cfg.factor, min_ratio=cfg.min_ratio, max_probes=cfg.max_probes,
        resolution=cfg.resolution, lr_bounds=cfg.lr_bounds, confirm=confirm_fn,
        center_window=cfg.center_window, max_window_probes=cfg.max_window_probes,
    )
    result = search.search()
    payload = {**dict(metadata or {}), "initial_lr": float(initial_lr), **result.to_dict()}
    with open(os.path.join(experiment_dir, "lr_search.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return result


def _better_probe(a: ProbeResult, b: ProbeResult) -> bool:
    """``a`` beats ``b``: converged first, then the higher best score."""
    if a.converged != b.converged:
        return bool(a.converged)
    score_a = max(a.scores) if a.scores else float("-inf")
    score_b = max(b.scores) if b.scores else float("-inf")
    return score_a > score_b


@contextlib.contextmanager
def _quiet_logs(enabled: bool) -> Iterator[None]:
    """Log only errors from the package while a probe trains (the search itself logs one line per probe).

    Warnings such as "model did not converge" are expected for probes with a bad learning rate.
    """
    package_logger = logging.getLogger("gradiend")
    previous = package_logger.level
    if enabled:
        package_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        package_logger.setLevel(previous)


@contextlib.contextmanager
def _isolated_probe(trainer: Any) -> Iterator[None]:
    """Make one probe an independent run.

    ``Trainer.train()`` leaves the trainer pointing at the checkpoint it just wrote
    (``_model_arg = out_path``) with that model resident on the GPU. Calling ``train()`` again
    would therefore continue from the previous probe's weights instead of the base model (and
    keep both models in memory). Each probe starts from the original base model and, afterwards,
    the trainer is handed back as it was found with the probe's model released.
    """
    if not hasattr(trainer, "_base_model_arg"):
        raise AttributeError("trainer has no _base_model_arg; cannot start each LR probe from the base model")
    saved_arg = trainer._model_arg
    trainer._model_arg = trainer._base_model_arg
    trainer._model_instance = None
    try:
        yield
    finally:
        trainer._model_arg = saved_arg
        trainer._model_instance = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch is a hard dependency; never mask the probe's own error
            pass


def _prune_probe_dir(probe_dir: str) -> None:
    """Keep only ``training.json`` of a probe: its checkpoint is as large as a real model."""
    for root, _dirs, files in os.walk(probe_dir):
        for name in files:
            if name != "training.json":
                try:
                    os.remove(os.path.join(root, name))
                except OSError:  # pragma: no cover - best effort, never masks the probe result
                    pass
    for root, dirs, files in os.walk(probe_dir, topdown=False):
        if root != probe_dir and not os.listdir(root):
            try:
                os.rmdir(root)
            except OSError:  # pragma: no cover
                pass


def _read_probe(probe_dir: str) -> ProbeResult:
    for root, _dirs, files in os.walk(probe_dir):
        if "training.json" in files:
            return ProbeResult.from_training_json(os.path.join(root, "training.json"))
    raise FileNotFoundError(f"no training.json under {probe_dir}")
