"""
Seed-selection policy for trainer analysis and inter-model comparison.

Comparison and visualization code should call helpers here instead of
duplicating seed loops or reading ``analyze_seed_stability`` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from gradiend.trainer.core.multi_seed import (
    MultiSeedTrainerView,
    load_seed_model_group,
    resolve_default_seed_selection,
    resolve_dispersion_for_trainers,
    resolve_seed_run_entries,
    resolve_seed_selection_for_trainers,
)
from gradiend.trainer.core.seed_models import SeedModelGroup

ModelOrGroup = Union[Any, SeedModelGroup, List[Any], Tuple[Any, ...]]


def is_multi_seed_view(trainer: Any) -> bool:
    return isinstance(trainer, MultiSeedTrainerView)


def unwrap_trainer(trainer: Any) -> Any:
    """Return the underlying :class:`Trainer` for a view or pass-through."""
    if is_multi_seed_view(trainer):
        return trainer.trainer
    return trainer


def _training_args(trainer: Any) -> Any:
    base = unwrap_trainer(trainer)
    return getattr(base, "_training_args", None) or getattr(base, "training_args", None)


def wants_multi_seed_analysis(trainer: Any) -> bool:
    """True when analysis should aggregate across multiple seed checkpoints."""
    if is_multi_seed_view(trainer):
        return len(trainer.seed_values()) > 1 or trainer.selection != "best"
    return resolve_default_seed_selection(unwrap_trainer(trainer), None) != "best"


def default_multi_seed_kwargs(trainer: Any) -> Dict[str, Any]:
    args = _training_args(trainer)
    stability = bool(getattr(args, "analyze_seed_stability", False)) if args is not None else False
    return {
        "selection": "all_convergent",
        "aggregate": "mean",
        "dispersion": "std" if stability else "none",
    }


def enter_analysis_mode(
    trainer: Any,
    *,
    selection: Optional[str] = None,
    aggregate: Optional[str] = None,
    dispersion: Optional[str] = None,
    return_per_seed: bool = False,
    force: bool = False,
) -> Any:
    """Return a multi-seed view when stability analysis is enabled.

    When ``analyze_seed_stability`` is false and ``force`` is false, returns
    ``trainer`` unchanged. Already-wrapped views are returned as-is unless
    ``force`` requests a fresh wrap of the inner trainer.
    """
    if is_multi_seed_view(trainer) and not force:
        return trainer
    base = unwrap_trainer(trainer)
    if not force and not getattr(_training_args(base), "analyze_seed_stability", False):
        return trainer
    defaults = default_multi_seed_kwargs(base)
    if selection is not None:
        defaults["selection"] = selection
    if aggregate is not None:
        defaults["aggregate"] = aggregate
    if dispersion is not None:
        defaults["dispersion"] = dispersion
    defaults["return_per_seed"] = return_per_seed
    return base.multi_seed(**defaults)


def enter_analysis_mode_for_trainers(trainers: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap each trainer that requests seed-stability analysis."""
    return {str(key): enter_analysis_mode(trainer) for key, trainer in trainers.items()}


def analysis_seed_entries(
    trainer: Any,
    *,
    seed_selection: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """Resolved ``(seed, checkpoint_dir)`` pairs for comparison encoding."""
    if is_multi_seed_view(trainer):
        selection = seed_selection or trainer.selection
        if selection == trainer.selection:
            return list(trainer._entries)
    base = unwrap_trainer(trainer)
    selection = resolve_default_seed_selection(base, seed_selection)
    return resolve_seed_run_entries(base, selection)


def evaluate_encoder_for_comparison(
    trainer: Any,
    /,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run (possibly multi-seed) encoder evaluation for comparison workflows."""
    view = enter_analysis_mode(trainer) if not is_multi_seed_view(trainer) else trainer
    if is_multi_seed_view(view):
        return view.evaluate_encoder(**kwargs)
    return unwrap_trainer(view).evaluate_encoder(**kwargs)


def models_for_comparison(
    trainer: Any,
    *,
    seed_selection: Optional[str] = None,
    gradiend_only: bool = False,
    use_cache: bool = True,
    shared_base_model: Any = None,
    shared_tokenizer: Any = None,
) -> Tuple[ModelOrGroup, Any, Any]:
    """Load one model or a seed group for weight-space comparison.

    Returns:
        ``(model_or_group, shared_base_model, shared_tokenizer)``
    """
    base = unwrap_trainer(trainer)
    if is_multi_seed_view(trainer):
        selection = seed_selection or trainer.selection
        aggregate = trainer.aggregate
        dispersion = trainer.dispersion
        seed_values = trainer.seed_values()
    else:
        selection = resolve_default_seed_selection(base, seed_selection)
        args = _training_args(base)
        stability = bool(getattr(args, "analyze_seed_stability", False)) if args is not None else False
        aggregate = "mean"
        dispersion = "std" if stability else "none"
        seed_values = [seed for seed, _ in resolve_seed_run_entries(base, selection)]

    if selection == "best" or len(seed_values) <= 1:
        load_kwargs: Dict[str, Any] = {"use_cache": use_cache}
        if shared_base_model is not None:
            load_kwargs["base_model"] = shared_base_model
        if shared_tokenizer is not None:
            load_kwargs["tokenizer"] = shared_tokenizer
        if gradiend_only:
            from gradiend.trainer.suite.definitions import _load_gradiend_only_model

            path = str(base.model_path)
            model = _load_gradiend_only_model(path, device="cpu")
        else:
            model = base.get_model(**load_kwargs)
        if shared_base_model is None and getattr(model, "base_model", None) is not None:
            shared_base_model = model.base_model
        if shared_tokenizer is None and getattr(model, "tokenizer", None) is not None:
            shared_tokenizer = model.tokenizer
        return model, shared_base_model, shared_tokenizer

    if gradiend_only:
        from gradiend.trainer.suite.definitions import _load_gradiend_only_model

        models = [
            _load_gradiend_only_model(path, device="cpu")
            for _, path in resolve_seed_run_entries(base, selection)
        ]
        group: ModelOrGroup = SeedModelGroup(
            models,
            selection=selection,
            aggregate=aggregate,
            dispersion=dispersion,
            seed_values=seed_values,
        )
        return group, shared_base_model, shared_tokenizer

    models, shared_base_model, shared_tokenizer = load_seed_model_group(
        base,
        selection=selection,
        shared_base_model=shared_base_model,
        shared_tokenizer=shared_tokenizer,
    )
    group = SeedModelGroup(
        models,
        selection=selection,
        aggregate=aggregate,
        dispersion=dispersion,
        seed_values=seed_values,
    )
    return group, shared_base_model, shared_tokenizer


def comparison_seed_metadata(
    trainers: Dict[str, Any],
    *,
    seed_selection: Optional[str] = None,
    seed_aggregate: str = "mean",
    dispersion: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared seed metadata for matrix builders and heatmaps."""
    resolved_selection = resolve_seed_selection_for_trainers(trainers, seed_selection)
    resolved_dispersion = resolve_dispersion_for_trainers(trainers, dispersion)
    return {
        "seed_selection": resolved_selection,
        "seed_aggregate": seed_aggregate,
        "dispersion": resolved_dispersion,
        "multi_seed": resolved_selection != "best",
    }
