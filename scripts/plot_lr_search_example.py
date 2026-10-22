"""Draw the learning-rate search figure of the documentation from a simulated training response.

No model is trained: ``simulated_probe`` invents what a probe at a given learning rate would report (nothing moves,
slow progress, convergence, an overshoot), and the real ``LRSearch`` controller searches over it starting from ``1e-7``.

    python scripts/plot_lr_search_example.py docs/img/lr_search_example.png
"""
import math
import sys
from typing import Optional, Sequence

from gradiend.trainer.core.lr_search import LRSearch, LRSearchResult, ProbeResult

PROBE_STEPS = range(0, 151, 25)  # seven evaluations, like the confirmation run over twice as many steps
CONFIRM_STEPS = range(0, 301, 50)
THRESHOLD = 0.5  # correlation and each class mean must reach it, as in the default TrainingArguments
# a too large learning rate: the classes separate for a moment (correlation 0.42), then both means end on the same side
OVERSHOOT = {"corr": [0.0, 0.42, 0.30, 0.0, -0.40, -0.60, -0.65],
             "pos": [0.0, 0.50, 0.60, 0.75, 0.85, 0.90, 0.90],
             "neg": [0.0, -0.20, 0.20, 0.55, 0.75, 0.85, 0.88]}


def simulated_probe(lr: float, steps: Sequence[int] = PROBE_STEPS) -> ProbeResult:
    """Made-up evaluations of a short training run: too small a learning rate barely moves, a moderate one converges
    (the classes get opposite means and a high correlation), a too large one overshoots."""
    steps = list(steps)
    scores, means = [], []
    for at, step in enumerate(steps):
        if lr >= 4e-5:
            corr, pos, neg = OVERSHOOT["corr"][at], OVERSHOOT["pos"][at], OVERSHOOT["neg"][at]
        else:
            progress = 1 - math.exp(-(lr / 1e-5) * step / 60)  # 0 -> 1: how far the encoder has separated the classes
            corr, pos, neg = 0.95 * progress, 0.92 * progress, -0.92 * progress
        scores.append(corr)
        means.append({"1.0": pos, "-1.0": neg})
    best = max(range(len(scores)), key=scores.__getitem__)
    weaker = min(abs(means[best]["1.0"]), abs(means[best]["-1.0"]))
    return ProbeResult(
        steps=steps, scores=scores, converged=scores[best] >= THRESHOLD and weaker >= THRESHOLD, threshold=THRESHOLD,
        metric="correlation", mean_by_class=means, class_mean_threshold=THRESHOLD,
    )


def simulated_search() -> LRSearchResult:
    return LRSearch(simulated_probe, initial_lr=1e-7, confirm=lambda lr: simulated_probe(lr, CONFIRM_STEPS)).search()


def main(output: Optional[str] = None) -> None:
    result = simulated_search()
    for probe in result.probes:
        print(f"{probe.lr:g}: {probe.state.value}")
    result.plot(output=output, show=output is None)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
