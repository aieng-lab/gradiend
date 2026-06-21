"""Logging summaries for text-prediction data creation."""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Optional, Tuple, Union

from gradiend.util.logging import get_logger

logger = get_logger(__name__)

SUMMARY_LOG_CLASS_THRESHOLD = 10
SUMMARY_LOG_EXAMPLES = 5


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return ordered[idx]


def _limited_dict(items: List[Tuple[str, Union[int, float, str]]], max_items: int = SUMMARY_LOG_EXAMPLES) -> Dict[str, Union[int, float, str]]:
    return dict(items[:max_items])


def _format_rate(value: float) -> str:
    return f"{100 * value:.2f}%"


def _log_training_filter_summary(
    stats_per_group: Dict[str, int],
    match_rates: Dict[str, float],
    *,
    total_target: Optional[int],
    total_so_far: int,
    max_size_per_class: Optional[int],
    n_processed: int,
) -> None:
    n_classes = len(stats_per_group)
    if not stats_per_group:
        return

    if n_classes <= SUMMARY_LOG_CLASS_THRESHOLD:
        logger.info("Training filter stats (instances per group): %s", stats_per_group)
        if match_rates:
            rates_vals = list(match_rates.values())
            if len(set(rates_vals)) == 1:
                logger.info(
                    "Match rate: %.2f%% of sentences scanned matched each class (single stream, shared denominator)",
                    100 * rates_vals[0],
                )
            else:
                logger.info(
                    "Match rate per class (fraction of sentences scanned): %s",
                    {k: _format_rate(v) for k, v in match_rates.items()},
                )
        return

    counts = list(stats_per_group.values())
    logger.info(
        "Training filter stats: classes=%s, total_matches=%s, sentences_scanned=%s, "
        "count min/median/p90/max=%s/%s/%s/%s",
        n_classes,
        total_so_far,
        n_processed,
        min(counts),
        median(counts),
        _percentile([float(v) for v in counts], 0.9),
        max(counts),
    )
    if total_target is not None and max_size_per_class is not None:
        completed = sum(1 for n in counts if n >= max_size_per_class)
        empty = sum(1 for n in counts if n == 0)
        partial = n_classes - completed - empty
        logger.info(
            "Class completion: completed=%s, partial=%s, empty=%s, target_per_class=%s",
            completed,
            partial,
            empty,
            max_size_per_class,
        )

    sorted_counts = sorted(stats_per_group.items(), key=lambda item: (item[1], item[0]))
    logger.info(
        "Count examples: lowest=%s, highest=%s, representative=%s",
        _limited_dict(sorted_counts),
        _limited_dict(list(reversed(sorted_counts))),
        _limited_dict(list(stats_per_group.items())),
    )

    if match_rates:
        rate_values = list(match_rates.values())
        sorted_rates = sorted(match_rates.items(), key=lambda item: (item[1], item[0]))
        logger.info(
            "Match rate summary: min/median/p90/max=%s/%s/%s/%s; lowest examples=%s",
            _format_rate(min(rate_values)),
            _format_rate(median(rate_values)),
            _format_rate(_percentile(rate_values, 0.9)),
            _format_rate(max(rate_values)),
            {k: _format_rate(v) for k, v in sorted_rates[:SUMMARY_LOG_EXAMPLES]},
        )
    logger.debug("Training filter stats (instances per group): %s", stats_per_group)

__all__ = ["SUMMARY_LOG_CLASS_THRESHOLD", "SUMMARY_LOG_EXAMPLES", "_log_training_filter_summary"]
