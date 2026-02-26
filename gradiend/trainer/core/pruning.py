"""
Pre-prune and post-prune configs and helpers.

Pre-prune: gradient-mean over n stratified samples, then prune (before training).
Post-prune: weight-based prune after training (same prune(), importance from get_weight_importance(part)).
"""

import random
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from tqdm import tqdm

from gradiend.util.logging import get_logger
from gradiend.trainer.core.config import (
    factual_computation_required_keywords,
    alternative_computation_required_keywords,
)

logger = get_logger(__name__)


def _validate_topk(topk: Optional[Union[int, float]], param_name: str = "topk") -> None:
    """Validate topk: int >= 0 or float in (0, 1.0]. Raises TypeError/ValueError."""
    if topk is None:
        return
    if isinstance(topk, bool):
        raise TypeError(f"{param_name} must be int or float, not bool")
    if isinstance(topk, int):
        if topk < 0:
            raise ValueError(f"{param_name} as int must be >= 0, got {topk}")
        return
    if isinstance(topk, float):
        if not (0.0 < topk <= 1.0):
            raise ValueError(f"{param_name} as float must be in (0, 1.0], got {topk}")
        return
    raise TypeError(f"{param_name} must be int, float, or None, got {type(topk).__name__}")


def _validate_threshold(threshold: Optional[float], param_name: str = "threshold") -> None:
    """Validate threshold: non-negative float when provided."""
    if threshold is None:
        return
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"{param_name} must be float or None, got {type(threshold).__name__}")
    if threshold < 0:
        raise ValueError(f"{param_name} must be >= 0, got {threshold}")


# ---------------------------------------------------------------------------
# PostPruneConfig (weight-based prune after training)
# ---------------------------------------------------------------------------

@dataclass
class PostPruneConfig:
    """
    Config for post-training prune: select dimensions by weight-based importance (part), then prune.

    When passed as training_args.post_prune_config, train() runs post_prune() automatically after
    training and saves the pruned model. You can also call post_prune() manually when not using this config.
    """

    topk: Optional[Union[int, float]] = None
    """Same as prune(): int (absolute, top-k dims) or float in (0,1] (relative). topk=1.0 (float) means no pruning. One of topk, threshold, or mask required."""

    threshold: Optional[float] = None
    """Same as prune(): keep dims with importance >= threshold."""

    mask: Optional[torch.Tensor] = None
    """Optional bool mask of shape (input_dim,). Not serialized in to_dict."""

    part: str = "decoder-weight"
    """Importance source: 'encoder-weight' | 'decoder-weight' | 'decoder-bias' | 'decoder-sum'."""

    inplace: bool = True
    """If True, prune the model in place."""

    return_mask: bool = False
    """If True, also return the combined mask from prune."""

    def __post_init__(self) -> None:
        if self.topk is None and self.threshold is None and self.mask is None:
            raise ValueError("PostPruneConfig: at least one of topk, threshold, or mask must be set.")
        _validate_topk(self.topk, "topk")
        _validate_threshold(self.threshold, "threshold")
        if not isinstance(self.part, str):
            raise TypeError(f"part must be str, got {type(self.part).__name__}")
        if self.part not in ("encoder-weight", "decoder-weight", "decoder-bias", "decoder-sum"):
            raise ValueError(
                "part must be encoder-weight, decoder-weight, decoder-bias, or decoder-sum; "
                f"got {self.part!r}"
            )
        if not isinstance(self.inplace, bool):
            raise TypeError(f"inplace must be bool, got {type(self.inplace).__name__}")
        if not isinstance(self.return_mask, bool):
            raise TypeError(f"return_mask must be bool, got {type(self.return_mask).__name__}")

    def __str__(self) -> str:
        return f"PostPruneConfig(topk={self.topk!r}, threshold={self.threshold!r}, part={self.part!r}, inplace={self.inplace})"


def post_prune(
    model_with_gradiend: Any,
    config: PostPruneConfig,
) -> Any:
    """
    Post-prune: apply weight-based prune to the trained model using the given config.

    Call after train(). Uses get_weight_importance(part) for importance; no gradient mean.

    Args:
        model_with_gradiend: Model with .gradiend and .prune_gradiend().
        config: PostPruneConfig (topk, threshold, mask, part, etc.).

    Returns:
        model (or copy if not inplace), or (model, combined_mask) if config.return_mask.
    """
    kwargs = {
        "topk": config.topk,
        "threshold": config.threshold,
        "mask": config.mask,
        "part": config.part,
        "inplace": config.inplace,
        "return_mask": config.return_mask,
    }
    return model_with_gradiend.prune_gradiend(**kwargs)


# ---------------------------------------------------------------------------
# PrePruneConfig (gradient-mean then prune before training)
# ---------------------------------------------------------------------------


@dataclass
class PrePruneConfig:
    """
    Config for pre-prune: gradient mean over n samples, then prune by topk or threshold.

    By default use the dataset passed to pre_prune(); set dataset to use different data for this step.
    """

    n_samples: int
    """Total number of samples to use for the gradient mean."""

    topk: Optional[Union[int, float]] = None
    """Same as prune(): int (absolute, top-k dims) or float in (0,1] (relative). topk=1.0 (float) means no pruning. One of topk or threshold required."""

    threshold: Optional[float] = None
    """Same as prune(): keep dims with importance >= threshold."""

    source: str = "factual"
    """Which gradient to average: 'factual' | 'alternative' | 'diff'."""

    batch_size: int = 8
    """Batch size for gradient computation (samples per chunk; one gradient per sample)."""

    feature_class_key: str = "feature_class_id"
    """Key in item dict for stratification (must have exactly two target classes)."""

    target_feature_class_ids: Optional[List[Any]] = None
    """Class IDs to stratify over (neutral/identity are ignored). If None and pre_prune(..., definition=trainer) is used, the trainer's target feature class IDs are used automatically."""

    dataset: Optional[Any] = None
    """Optional override: use this dataset instead of the one passed to pre_prune()."""

    use_cached_gradients: bool = False
    """If True, can use same cache key as training. Default False."""

    def __post_init__(self) -> None:
        if self.topk is None and self.threshold is None:
            raise ValueError("PrePruneConfig: at least one of topk or threshold must be set.")
        if not isinstance(self.n_samples, int):
            raise TypeError(f"n_samples must be int, got {type(self.n_samples).__name__}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        _validate_topk(self.topk, "topk")
        _validate_threshold(self.threshold, "threshold")
        if not isinstance(self.source, str):
            raise TypeError(f"source must be str, got {type(self.source).__name__}")
        if self.source not in ("factual", "alternative", "diff"):
            raise ValueError(f"source must be one of factual, alternative, diff; got {self.source!r}")
        if not isinstance(self.batch_size, int):
            raise TypeError(f"batch_size must be int, got {type(self.batch_size).__name__}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not isinstance(self.feature_class_key, str):
            raise TypeError(f"feature_class_key must be str, got {type(self.feature_class_key).__name__}")
        if not isinstance(self.use_cached_gradients, bool):
            raise TypeError(f"use_cached_gradients must be bool, got {type(self.use_cached_gradients).__name__}")

    def __str__(self) -> str:
        return f"PrePruneConfig(n_samples={self.n_samples}, topk={self.topk!r}, threshold={self.threshold!r}, source={self.source!r})"


def _stratified_indices(
    dataset: Any,
    n_samples: int,
    feature_class_key: str,
    target_feature_class_ids: Optional[List[Any]],
) -> List[int]:
    """
    Return list of indices: n_samples // n_classes per feature class (with replacement if needed).
    
    Samples are stratified evenly across all specified feature classes. If n_samples is not
    evenly divisible by the number of classes, the remainder is distributed to the first classes.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty.")

    by_fc: dict = {}
    for idx in range(n):
        item = dataset[idx]
        fc = item.get(feature_class_key)
        if fc is None:
            raise KeyError(
                f"Pre-prune requires dataset items to have key {feature_class_key!r}. "
                f"Add it to your dataset __getitem__ (e.g. TextTrainingDataset)."
            )
        by_fc.setdefault(fc, []).append(idx)

    if target_feature_class_ids is not None:
        class_ids = list(target_feature_class_ids)
    else:
        class_ids = list(by_fc.keys())

    if len(class_ids) == 0:
        raise ValueError("No feature classes found for stratification.")

    n_classes = len(class_ids)
    per_class = n_samples // n_classes
    remainder = n_samples % n_classes
    
    out: List[int] = []
    for i, cid in enumerate(class_ids):
        indices = by_fc.get(cid, [])
        if not indices:
            raise ValueError(f"No samples for feature class {cid!r}.")
        
        # Distribute remainder to first classes
        samples_needed = per_class + (1 if i < remainder else 0)
        
        if len(indices) >= samples_needed:
            out.extend(random.sample(indices, samples_needed))
        else:
            out.extend(random.choices(indices, k=samples_needed))
    random.shuffle(out)
    return out


def _gradient_to_vector(grad: Any, gradiend) -> torch.Tensor:
    """Turn gradient (dict or tensor) into 1D CPU float tensor in GRADIEND input space."""
    if isinstance(grad, dict):
        vec = gradiend.flatten_gradient_dict(grad)
    elif torch.is_tensor(grad):
        vec = grad.flatten()
    else:
        raise TypeError(f"Gradient must be dict or tensor, got {type(grad)}")
    return vec.detach().float().cpu()


def pre_prune(
    model_with_gradiend: Any,
    dataset: Any,
    config: PrePruneConfig,
    *,
    definition: Optional[Any] = None,
    inplace: bool = True,
    return_mask: bool = False,
) -> Any:
    """
    Pre-prune: compute running mean of gradients over n stratified samples, then prune the model.

    Uses the same prune() as post-training; importance comes from abs(mean gradient) instead of weights.
    Call this before train(); typically pass the same dataset you will use for training.

    When the dataset has more than two feature classes (e.g. target pair + identity/neutral),
    pass definition=trainer so stratification uses only the target pair. The trainer supplies
    the target feature class IDs via get_target_feature_class_ids().

    Args:
        model_with_gradiend: Model with .gradiend and .gradient_creator (e.g. ModelWithGradiend).
        dataset: Dataset with __len__ and __getitem__ returning dict with 'factual', 'alternative',
            and config.feature_class_key (e.g. 'feature_class_id'). Ignored if config.dataset is set.
        config: PrePruneConfig (n_samples, topk or threshold, source, etc.).
        definition: Optional trainer/definition with get_target_feature_class_ids(). When given and
            the dataset has more feature classes than desired, target feature class IDs are used automatically.
        inplace: If True, prune the model in place; else return a copy.
        return_mask: If True, also return the combined mask from prune.

    Returns:
        model (or copy if not inplace), or (model, combined_mask) if return_mask.
    """
    data = config.dataset if config.dataset is not None else dataset
    if data is None:
        raise ValueError("No dataset: pass dataset to pre_prune() or set config.dataset.")

    gradiend = model_with_gradiend.gradiend
    gradient_creator = getattr(model_with_gradiend, "gradient_creator", None) or getattr(
        model_with_gradiend, "_gradient_creator", model_with_gradiend
    )
    if gradient_creator is None:
        raise ValueError("model_with_gradiend must have gradient_creator for pre_prune.")

    source = config.source
    requires_factual = source in factual_computation_required_keywords
    requires_alternative = source in alternative_computation_required_keywords

    target_ids = config.target_feature_class_ids
    if target_ids is None and definition is not None:
        get_ids = getattr(definition, "get_target_feature_class_ids", None)
        if callable(get_ids):
            target_ids = get_ids()

    indices = _stratified_indices(
        data,
        config.n_samples,
        config.feature_class_key,
        target_ids,
    )

    running_sum: Optional[torch.Tensor] = None
    count = 0

    _tqdm_kw = dict(
        total=len(indices),
        desc="Pre-prune",
        unit="datapoint",
        leave=True,
        ncols=80,
        dynamic_ncols=False,
        ascii=True,
        mininterval=0.5,
        position=0,
        disable=not sys.stderr.isatty(),
    )
    for idx in tqdm(indices, **_tqdm_kw):
        item = data[idx]
        factual_in = item["factual"]
        alternative_in = item["alternative"]

        fact_g = None
        if requires_factual:
            fact_g = gradient_creator(factual_in, target_device=torch.device("cpu"))
        alt_g = None
        if requires_alternative:
            alt_g = gradient_creator(alternative_in, target_device=torch.device("cpu"))

        if source == "factual":
            g = fact_g
        elif source == "alternative":
            g = alt_g
        else:
            if isinstance(fact_g, dict):
                g = {k: fact_g[k] - alt_g[k] for k in fact_g}
            else:
                g = fact_g - alt_g

        vec = _gradient_to_vector(g, gradiend)
        if running_sum is None:
            running_sum = torch.zeros_like(vec)
        running_sum.add_(vec)
        count += 1

    if running_sum is None or count == 0:
        raise RuntimeError("No gradients accumulated in pre_prune.")

    mean = running_sum / count
    importance = mean.abs()

    logger.debug(
        "Pre-prune: computed mean over %s samples, importance shape %s, topk=%s, threshold=%s",
        count, importance.shape, config.topk, config.threshold,
    )

    kwargs = {
        "importance": importance,
        "topk": config.topk,
        "threshold": config.threshold,
        "inplace": inplace,
        "return_mask": return_mask,
    }
    return model_with_gradiend.prune_gradiend(**kwargs)
