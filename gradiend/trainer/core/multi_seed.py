"""
Multi-seed evaluation view for Trainer.

All multi-seed analysis goes through ``trainer.multi_seed()`` → ``MultiSeedTrainerView``.
Single-seed Trainer methods are unchanged.
"""

from __future__ import annotations

import gc
import math
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from gradiend.util.logging import get_logger

logger = get_logger(__name__)

VALID_SELECTIONS = frozenset({"best", "all_convergent", "all_tried"})
VALID_AGGREGATES = frozenset({"mean", "median", "min", "max"})
VALID_DISPERSION = frozenset({"none", "std", "range", "minmax"})

MULTI_SEED_CAPABLE_METHODS: set = {
    "evaluate",
    "evaluate_encoder",
    "evaluate_decoder",
    "plot_training_convergence",
    "plot_encoder_distributions",
    "plot_encoder_scatter",
    "plot_probability_shifts",
}

PLOT_METHODS = frozenset({
    "plot_training_convergence",
    "plot_encoder_distributions",
    "plot_encoder_scatter",
    "plot_probability_shifts",
})

# Keys omitted from top-level aggregate (kept in seeds.per_seed when requested).
EVAL_SKIP_TOP_LEVEL_KEYS = frozenset({
    "encoder_df",
    "training_rows",
    "grid",
    "plot_paths",
    "plot_path",
    "seeds",
})

# Scalar metrics aggregated with seed_aggregate / dispersion.
EVAL_SCALAR_KEYS = frozenset({
    "correlation",
    "n_samples",
    "accuracy",
    "positive_mean",
    "negative_mean",
})

SEED_AGGREGATOR_REGISTRY: Dict[str, Callable[[Sequence[Any]], Any]] = {}


def register_seed_aggregator(key: str, fn: Callable[[Sequence[Any]], Any]) -> None:
    """Register a custom aggregator for a result key.

    Args:
        key: Result key handled by the aggregator.
        fn: Callable receiving the per-seed values for ``key``.
    """
    SEED_AGGREGATOR_REGISTRY[str(key)] = fn


def _validate_aggregate_dispersion(aggregate: str, dispersion: str) -> None:
    if aggregate not in VALID_AGGREGATES:
        raise ValueError(f"aggregate must be one of {sorted(VALID_AGGREGATES)}, got {aggregate!r}")
    if dispersion not in VALID_DISPERSION:
        raise ValueError(f"dispersion must be one of {sorted(VALID_DISPERSION)}, got {dispersion!r}")
    if dispersion == "range" and aggregate in {"min", "max"}:
        raise ValueError("dispersion='range' does not make sense with aggregate='min' or 'max'")


def _aggregate_numeric_values(
    values: Sequence[float],
    *,
    aggregate: str,
    dispersion: str,
) -> Dict[str, Any]:
    scores = [float(v) for v in values]
    if not scores:
        raise ValueError("Cannot aggregate an empty score list")
    ordered = sorted(scores)
    n = len(ordered)
    if aggregate == "mean":
        agg = float(sum(ordered) / n)
    elif aggregate == "median":
        mid = n // 2
        agg = float(ordered[mid] if n % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid]))
    elif aggregate == "min":
        agg = float(ordered[0])
    elif aggregate == "max":
        agg = float(ordered[-1])
    else:
        raise ValueError(f"Unsupported aggregate {aggregate!r}")
    result: Dict[str, Any] = {
        "mean": agg if aggregate == "mean" else float(sum(ordered) / n),
        "aggregate": agg,
        "n": n,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }
    if dispersion == "std":
        mean = float(sum(ordered) / n)
        result["std"] = float(math.sqrt(sum((v - mean) ** 2 for v in ordered) / n))
    elif dispersion == "range":
        result["range_half_width"] = float((ordered[-1] - ordered[0]) / 2.0)
    elif dispersion == "minmax":
        result["minmax"] = [float(ordered[0]), float(ordered[-1])]
    return result


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _aggregate_nested_numeric(
    values: Sequence[Any],
    *,
    aggregate: str,
    dispersion: str,
) -> Any:
    if not values:
        return None
    first = values[0]
    if _is_numeric_scalar(first):
        stats = _aggregate_numeric_values([float(v) for v in values if _is_numeric_scalar(v)], aggregate=aggregate, dispersion=dispersion)
        return stats["aggregate"]
    if isinstance(first, dict):
        keys = set()
        for item in values:
            if isinstance(item, dict):
                keys.update(item.keys())
        out: Dict[Any, Any] = {}
        for key in keys:
            child = [item.get(key) for item in values if isinstance(item, dict) and key in item]
            if child:
                out[key] = _aggregate_nested_numeric(child, aggregate=aggregate, dispersion=dispersion)
        return out
    return first


def _collect_scalar_stats(
    results: Sequence[Dict[str, Any]],
    key: str,
    *,
    aggregate: str,
    dispersion: str,
) -> Optional[Dict[str, Any]]:
    values: List[float] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        val = result.get(key)
        if _is_numeric_scalar(val):
            values.append(float(val))
        elif isinstance(val, dict) and "value" in val and _is_numeric_scalar(val["value"]):
            values.append(float(val["value"]))
    if not values:
        return None
    return _aggregate_numeric_values(values, aggregate=aggregate, dispersion=dispersion)


def aggregate_eval_results(
    results: Sequence[Dict[str, Any]],
    seed_values: Sequence[int],
    *,
    selection: str = "all_convergent",
    aggregate: str,
    dispersion: str,
    return_per_seed: bool = False,
) -> Dict[str, Any]:
    """Merge per-seed evaluation dicts into an aggregate result.

    Args:
        results: Per-seed evaluation result dictionaries.
        seed_values: Seed values corresponding to ``results``.
        selection: Seed selection mode used to produce the inputs.
        aggregate: Scalar aggregation mode, e.g. ``"mean"``.
        dispersion: Dispersion statistic stored under ``seeds.stats``.
        return_per_seed: If True, include full per-seed payloads.
    """
    if not results:
        raise ValueError("aggregate_eval_results requires at least one result")
    _validate_aggregate_dispersion(aggregate, dispersion)

    stats: Dict[str, Any] = {}
    for key in set().union(*(r.keys() for r in results if isinstance(r, dict))):
        if key in EVAL_SKIP_TOP_LEVEL_KEYS:
            continue
        if key in SEED_AGGREGATOR_REGISTRY:
            stats[key] = SEED_AGGREGATOR_REGISTRY[key]([r.get(key) for r in results])
            continue
        key_stats = _collect_scalar_stats(results, key, aggregate=aggregate, dispersion=dispersion)
        if key_stats is not None:
            stats[key] = key_stats

    merged: Dict[str, Any] = {}
    for key in set().union(*(r.keys() for r in results if isinstance(r, dict))):
        if key in EVAL_SKIP_TOP_LEVEL_KEYS:
            continue
        if key in stats:
            stat = stats[key]
            if isinstance(stat, dict) and "aggregate" in stat:
                merged[key] = stat["aggregate"]
            else:
                merged[key] = stat
            continue
        values = [r.get(key) for r in results if isinstance(r, dict) and key in r]
        if not values:
            continue
        first = values[0]
        if _is_numeric_scalar(first):
            merged[key] = _aggregate_nested_numeric(values, aggregate=aggregate, dispersion=dispersion)
        elif isinstance(first, dict):
            if all(isinstance(v, dict) and "value" in v for v in values):
                merged[key] = dict(first)
                val_stats = _collect_scalar_stats(
                    [{key: v} for v in values],
                    key,
                    aggregate=aggregate,
                    dispersion=dispersion,
                )
                if val_stats is not None:
                    merged[key]["value"] = val_stats["aggregate"]
            else:
                merged[key] = _aggregate_nested_numeric(values, aggregate=aggregate, dispersion=dispersion)
        else:
            merged[key] = first

    seeds_block: Dict[str, Any] = {
        "n": len(results),
        "values": list(seed_values),
        "selection": selection,
        "aggregate": aggregate,
        "dispersion": dispersion,
        "stats": {k: v for k, v in stats.items() if isinstance(v, dict)},
    }
    if return_per_seed:
        seeds_block["per_seed"] = {
            int(seed): results[i] for i, seed in enumerate(seed_values)
        }
    merged["seeds"] = seeds_block
    return merged


def _aggregate_plot_results(
    results: Sequence[Any],
    seed_values: Sequence[int],
    *,
    selection: str,
    aggregate: str,
    dispersion: str,
    return_per_seed: bool,
) -> Dict[str, Any]:
    paths: List[str] = []
    per_seed: Dict[int, Any] = {}
    for idx, seed in enumerate(seed_values):
        item = results[idx]
        per_seed[int(seed)] = item
        if isinstance(item, str) and item:
            paths.append(item)
        elif isinstance(item, dict):
            p = item.get("path") or item.get("plot_path")
            if isinstance(p, str) and p:
                paths.append(p)
            pp = item.get("plot_paths")
            if isinstance(pp, list):
                paths.extend(str(x) for x in pp if x)

    out: Dict[str, Any] = {
        "paths": paths,
        "seeds": {
            "n": len(seed_values),
            "values": list(seed_values),
            "selection": selection,
            "aggregate": aggregate,
            "dispersion": dispersion,
        },
    }
    if paths:
        out["path"] = paths[0]
    if return_per_seed:
        out["seeds"]["per_seed"] = per_seed
    return out


def resolve_default_seed_selection(trainer: Any, seed_selection: Optional[str] = None) -> str:
    """Resolve default seed selection for suite/comparison calls.

    Args:
        trainer: Trainer whose training args may request seed-stability analysis.
        seed_selection: Optional explicit selection override.
    """
    if seed_selection is not None:
        selected = str(seed_selection).strip().lower()
        if selected not in VALID_SELECTIONS:
            raise ValueError(f"seed_selection must be one of {sorted(VALID_SELECTIONS)}, got {seed_selection!r}")
        return selected
    args = getattr(trainer, "_training_args", None) or getattr(trainer, "training_args", None)
    if args is not None and getattr(args, "analyze_seed_stability", False):
        return "all_convergent"
    return "best"


def load_seed_model_group(
    trainer: Any,
    *,
    selection: str = "all_convergent",
    shared_base_model: Any = None,
    shared_tokenizer: Any = None,
) -> Tuple[List[Any], Any, Any]:
    """Load selected seed checkpoints for comparison.

    Args:
        trainer: Trainer used to load seed checkpoints.
        selection: Which saved seed runs to load.
        shared_base_model: Optional shared base model reused across checkpoints.
        shared_tokenizer: Optional shared tokenizer reused across checkpoints.

    Returns:
        Tuple ``(models, shared_base_model, shared_tokenizer)``.
    """
    entries = resolve_seed_run_entries(trainer, selection)
    models: List[Any] = []
    for _, seed_path in entries:
        load_kwargs: Dict[str, Any] = {"use_cache": False}
        if shared_base_model is not None:
            load_kwargs["base_model"] = shared_base_model
        if shared_tokenizer is not None:
            load_kwargs["tokenizer"] = shared_tokenizer
        model = trainer.load_model(seed_path, **load_kwargs)
        if shared_base_model is None and getattr(model, "base_model", None) is not None:
            shared_base_model = model.base_model
        if shared_tokenizer is None and getattr(model, "tokenizer", None) is not None:
            shared_tokenizer = model.tokenizer
        models.append(model)
    return models, shared_base_model, shared_tokenizer


def resolve_seed_run_entries(trainer: Any, selection: str) -> List[Tuple[int, str]]:
    """Return ``(seed_value, output_dir)`` pairs for multi-seed analysis.

    Args:
        trainer: Trainer exposing a seed report and model path.
        selection: ``"best"``, ``"all_convergent"``, or ``"all_tried"``.
    """
    selection = str(selection).strip().lower()
    if selection not in VALID_SELECTIONS:
        raise ValueError(f"selection must be one of {sorted(VALID_SELECTIONS)}, got {selection!r}")

    if selection == "best":
        report = trainer.get_seed_report() if hasattr(trainer, "get_seed_report") else None
        best_seed = report.get("best_seed") if isinstance(report, dict) else None
        path = str(trainer.model_path)
        seed_val = int(best_seed) if isinstance(best_seed, int) else 0
        return [(seed_val, path)]

    if hasattr(trainer, "get_seed_report"):
        report = trainer.get_seed_report()
    else:
        report = None
    if not report:
        return [(0, str(trainer.model_path))]

    runs = report.get("runs", [])
    if not isinstance(runs, list):
        return [(0, str(trainer.model_path))]

    entries: List[Tuple[int, str]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        output_dir = run.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir:
            continue
        if selection == "all_convergent" and not bool(run.get("converged")):
            continue
        if not os.path.isdir(output_dir):
            continue
        seed_raw = run.get("seed")
        seed_val = int(seed_raw) if isinstance(seed_raw, int) else len(entries)
        entries.append((seed_val, output_dir))

    if not entries:
        return [(0, str(trainer.model_path))]
    return entries


@contextmanager
def _seed_execution_context(trainer: Any, seed_path: str, seed_model: Any):
    """Temporarily point trainer caches and get_model() at one seed checkpoint."""
    args = getattr(trainer, "_training_args", None)
    prev_exp = getattr(args, "experiment_dir", None) if args is not None else None
    prev_instance = getattr(trainer, "_model_instance", None)
    try:
        if args is not None:
            args.experiment_dir = seed_path
        trainer._model_instance = seed_model
        yield
    finally:
        if args is not None:
            args.experiment_dir = prev_exp
        trainer._model_instance = prev_instance


class MultiSeedTrainerView:
    """
    Multi-seed evaluation/plotting view for a Trainer.

    Obtain via ``trainer.multi_seed(...)``. Exposes the same eval/plot methods as the
    underlying trainer, running each across convergent seed checkpoints and aggregating.
    """

    def __init__(
        self,
        trainer: Any,
        *,
        selection: str = "all_convergent",
        aggregate: str = "mean",
        dispersion: str = "std",
        return_per_seed: bool = False,
    ) -> None:
        _validate_aggregate_dispersion(aggregate, dispersion)
        self._trainer = trainer
        self.selection = str(selection).strip().lower()
        self.aggregate = aggregate
        self.dispersion = dispersion
        self.return_per_seed = bool(return_per_seed)
        self._entries = resolve_seed_run_entries(trainer, self.selection)
        self._shared_base_model: Any = None
        self._shared_tokenizer: Any = None

    @property
    def trainer(self) -> Any:
        return self._trainer

    def seed_paths(self) -> List[str]:
        return [path for _, path in self._entries]

    def seed_values(self) -> List[int]:
        return [seed for seed, _ in self._entries]

    def _load_seed_model(self, seed_path: str) -> Any:
        load_kwargs: Dict[str, Any] = {"use_cache": False}
        if self._shared_base_model is not None:
            load_kwargs["base_model"] = self._shared_base_model
        if self._shared_tokenizer is not None:
            load_kwargs["tokenizer"] = self._shared_tokenizer
        model = self._trainer.load_model(seed_path, **load_kwargs)
        if self._shared_base_model is None and getattr(model, "base_model", None) is not None:
            self._shared_base_model = model.base_model
        if self._shared_tokenizer is None and getattr(model, "tokenizer", None) is not None:
            self._shared_tokenizer = model.tokenizer
        return model

    def _cleanup_model(self, model: Any) -> None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_for_seeds(self, method_name: str, fn: Callable[..., Any], /, **kwargs: Any) -> Any:
        if len(self._entries) == 1:
            seed_val, seed_path = self._entries[0]
            model = self._load_seed_model(seed_path)
            try:
                with _seed_execution_context(self._trainer, seed_path, model):
                    single = fn(**kwargs)
            finally:
                self._cleanup_model(model)
            if method_name in PLOT_METHODS:
                plot_payload = _aggregate_plot_results(
                    [single],
                    [seed_val],
                    selection=self.selection,
                    aggregate=self.aggregate,
                    dispersion=self.dispersion,
                    return_per_seed=self.return_per_seed,
                )
                if not self.return_per_seed:
                    plot_payload["seeds"].pop("per_seed", None)
                return plot_payload
            if isinstance(single, dict):
                payload = dict(single)
                per_seed_payload = dict(payload)
                payload["seeds"] = {
                    "n": 1,
                    "values": [seed_val],
                    "selection": self.selection,
                    "aggregate": self.aggregate,
                    "dispersion": self.dispersion,
                    "stats": _build_stats_from_single(payload, self.aggregate, self.dispersion),
                }
                if self.return_per_seed:
                    payload["seeds"]["per_seed"] = {seed_val: per_seed_payload}
                return payload
            return single

        results: List[Any] = []
        seed_values: List[int] = []
        for seed_val, seed_path in self._entries:
            model = self._load_seed_model(seed_path)
            try:
                with _seed_execution_context(self._trainer, seed_path, model):
                    results.append(fn(**kwargs))
                seed_values.append(int(seed_val))
            finally:
                self._cleanup_model(model)

        if method_name in PLOT_METHODS:
            payload = _aggregate_plot_results(
                results,
                seed_values,
                selection=self.selection,
                aggregate=self.aggregate,
                dispersion=self.dispersion,
                return_per_seed=self.return_per_seed,
            )
            if not self.return_per_seed:
                payload["seeds"].pop("per_seed", None)
            return payload

        dict_results = [r for r in results if isinstance(r, dict)]
        if len(dict_results) != len(results):
            return {
                "results": results,
                "seeds": {
                    "n": len(results),
                    "values": seed_values,
                    "selection": self.selection,
                    "aggregate": self.aggregate,
                    "dispersion": self.dispersion,
                    "per_seed": dict(zip(seed_values, results)) if self.return_per_seed else None,
                },
            }

        if method_name == "evaluate":
            enc_results = [r.get("encoder", {}) for r in dict_results if isinstance(r.get("encoder"), dict)]
            dec_results = [r.get("decoder", {}) for r in dict_results if isinstance(r.get("decoder"), dict)]
            merged: Dict[str, Any] = {}
            if enc_results:
                merged["encoder"] = aggregate_eval_results(
                    enc_results,
                    seed_values,
                    selection=self.selection,
                    aggregate=self.aggregate,
                    dispersion=self.dispersion,
                    return_per_seed=self.return_per_seed,
                )
            if dec_results:
                merged["decoder"] = aggregate_eval_results(
                    dec_results,
                    seed_values,
                    selection=self.selection,
                    aggregate=self.aggregate,
                    dispersion=self.dispersion,
                    return_per_seed=self.return_per_seed,
                )
            return merged

        merged_eval = aggregate_eval_results(
            dict_results,
            seed_values,
            selection=self.selection,
            aggregate=self.aggregate,
            dispersion=self.dispersion,
            return_per_seed=self.return_per_seed,
        )
        if not self.return_per_seed and "per_seed" in merged_eval.get("seeds", {}):
            merged_eval["seeds"].pop("per_seed", None)
        return merged_eval

    def _bind_method(self, name: str) -> Callable[..., Any]:
        fn = getattr(self._trainer, name)

        def _caller(**kwargs: Any) -> Any:
            return self._run_for_seeds(name, fn, **kwargs)

        _caller.__name__ = name
        _caller.__doc__ = fn.__doc__
        return _caller

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        """Run ``trainer.evaluate`` for each selected seed and aggregate results.

        Args:
            **kwargs: Forwarded to the underlying trainer method.
        """
        return self._bind_method("evaluate")(**kwargs)

    def evaluate_encoder(self, **kwargs: Any) -> Dict[str, Any]:
        """Run encoder evaluation for each selected seed and aggregate metrics.

        Args:
            **kwargs: Forwarded to ``trainer.evaluate_encoder``.
        """
        return self._bind_method("evaluate_encoder")(**kwargs)

    def evaluate_decoder(self, **kwargs: Any) -> Dict[str, Any]:
        """Run decoder evaluation for each selected seed and aggregate metrics.

        Args:
            **kwargs: Forwarded to ``trainer.evaluate_decoder``.
        """
        return self._bind_method("evaluate_decoder")(**kwargs)

    def plot_training_convergence(self, **kwargs: Any) -> Dict[str, Any]:
        """Create training-convergence plots for each selected seed.

        Args:
            **kwargs: Forwarded to ``trainer.plot_training_convergence``.
        """
        return self._bind_method("plot_training_convergence")(**kwargs)

    def plot_encoder_distributions(self, **kwargs: Any) -> Dict[str, Any]:
        """Create encoder-distribution plots for each selected seed.

        Args:
            **kwargs: Forwarded to ``trainer.plot_encoder_distributions``.
        """
        return self._bind_method("plot_encoder_distributions")(**kwargs)

    def plot_encoder_scatter(self, **kwargs: Any) -> Dict[str, Any]:
        """Create interactive encoder-scatter plots for each selected seed.

        Args:
            **kwargs: Forwarded to ``trainer.plot_encoder_scatter``.
        """
        return self._bind_method("plot_encoder_scatter")(**kwargs)

    def plot_probability_shifts(self, **kwargs: Any) -> Dict[str, Any]:
        """Create decoder probability-shift plots for each selected seed.

        Args:
            **kwargs: Forwarded to ``trainer.plot_probability_shifts``.
        """
        return self._bind_method("plot_probability_shifts")(**kwargs)

    def seed_models(self) -> Iterator[Any]:
        """Lazy-load seed checkpoints (shared base model when possible)."""
        for _seed_val, seed_path in self._entries:
            model = self._load_seed_model(seed_path)
            try:
                yield model
            finally:
                self._cleanup_model(model)

    def load_models(self) -> List[Any]:
        """Load all seed checkpoints into memory (caller should release when done)."""
        models, _, _ = load_seed_model_group(
            self._trainer,
            selection=self.selection,
            shared_base_model=self._shared_base_model,
            shared_tokenizer=self._shared_tokenizer,
        )
        if models and self._shared_base_model is None:
            first = models[0]
            if getattr(first, "base_model", None) is not None:
                self._shared_base_model = first.base_model
            if getattr(first, "tokenizer", None) is not None:
                self._shared_tokenizer = first.tokenizer
        return models

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in MULTI_SEED_CAPABLE_METHODS:
            return self._bind_method(name)
        return getattr(self._trainer, name)


def _build_stats_from_single(result: Dict[str, Any], aggregate: str, dispersion: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for key, val in result.items():
        if key in EVAL_SKIP_TOP_LEVEL_KEYS:
            continue
        if _is_numeric_scalar(val):
            stats[key] = _aggregate_numeric_values([float(val)], aggregate=aggregate, dispersion=dispersion)
        elif isinstance(val, dict) and _is_numeric_scalar(val.get("value")):
            stats[key] = _aggregate_numeric_values([float(val["value"])], aggregate=aggregate, dispersion=dispersion)
    return stats


def multi_seed_capable(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator marking a Trainer method as supported by MultiSeedTrainerView."""
    MULTI_SEED_CAPABLE_METHODS.add(method.__name__)
    return method
