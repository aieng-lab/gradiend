"""
Pre-prune and post-prune configs and helpers.

Pre-prune: gradient-mean over n stratified samples, then prune (before training).
Post-prune: weight-based prune after training (same prune(), importance from get_weight_importance(part)).
"""

import json
import os
import random
import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from gradiend.util.tqdm_utils import gradiend_tqdm

from gradiend.util import format_count
from gradiend.util.logging import get_logger
from gradiend.util.paths import has_saved_pre_prune_cache, should_use_cached
from gradiend.trainer.core.config import (
    factual_computation_required_keywords,
    alternative_computation_required_keywords,
)

logger = get_logger(__name__)


def _available_system_memory_bytes() -> Optional[int]:
    """Best-effort available RAM query; returns None when unsupported."""
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
                return int(pages) * int(page_size)
        except (OSError, ValueError):
            pass
    return None


def _format_bytes(n_bytes: Optional[int]) -> str:
    if n_bytes is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n_bytes} B"


def _estimate_pre_prune_cpu_peak_bytes(input_dim: int, source: str) -> int:
    """
    Estimate current pre-prune CPU vector peak.

    This implementation materializes full unpruned vectors before applying top-k.
    Gradients are typically bf16/fp16 on CPU, while _gradient_to_vector promotes
    the working vector and running sum to float32.
    """
    source = source or "diff"
    if source == "diff":
        bytes_per_dim = 14  # factual bf16 + alternative bf16 + diff bf16 + vec fp32 + running_sum fp32
    elif source in ("factual", "alternative"):
        bytes_per_dim = 10  # one gradient bf16 + vec fp32 + running_sum fp32
    else:
        bytes_per_dim = 14
    return int(input_dim) * bytes_per_dim


def _guard_pre_prune_memory(gradiend: Any, source: str) -> None:
    if os.environ.get("GRADIEND_ALLOW_UNSAFE_PREPRUNE") == "1":
        return

    input_dim = int(getattr(gradiend, "input_dim", 0) or 0)
    if input_dim <= 0:
        return

    estimated = _estimate_pre_prune_cpu_peak_bytes(input_dim, source)
    available = _available_system_memory_bytes()
    logger.info(
        "Pre-prune memory guard: input_dim=%s, source=%s, estimated current CPU vector peak=%s, available RAM=%s.",
        input_dim,
        source,
        _format_bytes(estimated),
        _format_bytes(available),
    )
    if available is not None and estimated > int(available * 0.75):
        raise RuntimeError(
            "Pre-prune refused before starting because the current implementation would materialize full "
            f"unpruned gradient vectors. Estimated CPU vector peak is {_format_bytes(estimated)} with "
            f"{_format_bytes(available)} available RAM. Lower topk does not reduce this current peak because "
            "topk is applied only after full-gradient accumulation. Use a streaming/distributed pre-prune "
            "implementation for this model size, or set GRADIEND_ALLOW_UNSAFE_PREPRUNE=1 to bypass this guard."
        )


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

    topk: Optional[Union[int, float]] = 0.01
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

    n_samples: int = 8
    """Total number of samples to use for the gradient mean."""

    topk: Optional[Union[int, float]] = 0.1
    """Same as prune(): int (absolute, top-k dims) or float in (0,1] (relative). topk=1.0 (float) means no pruning. One of topk or threshold required."""

    threshold: Optional[float] = None
    """Same as prune(): keep dims with importance >= threshold."""

    source: str = "alternative"
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

    seed: Optional[int] = None
    """Optional RNG seed for stratified sample selection. When set, smaller ``n_samples`` values
    draw a subset of the indices used for larger ``n_samples`` (per class, after shuffling)."""

    use_streaming: Optional[bool] = None
    """Pre-prune implementation selection. ``None`` (default): auto — use streaming when dict
    inputs and a base forward model are available. ``False``: classic v0.1.0-style loop via
    ``gradient_creator`` (fp32 CPU accumulation). ``True``: force streaming hooks. Overridden by
    env ``GRADIEND_PREPRUNE_STREAMING=0|1|classic|streaming`` when set."""

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
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError(f"seed must be int or None, got {type(self.seed).__name__}")
        if self.use_streaming is not None and not isinstance(self.use_streaming, bool):
            raise TypeError(
                f"use_streaming must be bool or None, got {type(self.use_streaming).__name__}"
            )

    def __str__(self) -> str:
        streaming = "auto" if self.use_streaming is None else str(self.use_streaming)
        return (
            f"PrePruneConfig(n_samples={self.n_samples}, topk={self.topk!r}, "
            f"threshold={self.threshold!r}, source={self.source!r}, seed={self.seed!r}, "
            f"use_streaming={streaming})"
        )


def _is_noop_pre_prune(config: PrePruneConfig) -> bool:
    """True when pre-prune would keep every dimension (topk=1.0 float, no threshold)."""
    if config.threshold is not None:
        return False
    topk = config.topk
    if topk is None:
        return False
    return isinstance(topk, float) and topk >= 1.0


def _stratified_indices(
    dataset: Any,
    n_samples: int,
    feature_class_key: str,
    target_feature_class_ids: Optional[List[Any]],
    seed: Optional[int] = None,
) -> List[int]:
    """
    Return list of indices: n_samples // n_classes per feature class (with replacement if needed).

    Samples are stratified evenly across all specified feature classes. If n_samples is not
    evenly divisible by the number of classes, the remainder is distributed to the first classes.

    When ``seed`` is set, each class pool is shuffled once; smaller ``n_samples`` values take
    prefixes of those pools so indices for k1 are a subset of indices for k2 when k1 < k2.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty.")

    by_fc: dict = {}
    raw_data = getattr(dataset, "data", None)
    if raw_data is not None and hasattr(raw_data, "columns") and feature_class_key in raw_data.columns:
        feature_classes = raw_data[feature_class_key].tolist()
        for idx, fc in enumerate(feature_classes):
            if fc is None:
                raise KeyError(
                    f"Pre-prune requires dataset items to have key {feature_class_key!r}. "
                    f"Column {feature_class_key!r} contains None at row {idx}."
                )
            by_fc.setdefault(fc, []).append(idx)
    else:
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
        # Only stratify over classes that actually have data in the dataset
        class_ids = [cid for cid in target_feature_class_ids if cid in by_fc]
    else:
        class_ids = list(by_fc.keys())

    if len(class_ids) == 0:
        raise ValueError("No feature classes found for stratification.")

    n_classes = len(class_ids)
    per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    rng = random.Random(seed) if seed is not None else random

    class_pools: Dict[Any, List[int]] = {}
    for cid in class_ids:
        pool = list(by_fc.get(cid, []))
        if not pool:
            raise ValueError(f"No samples for feature class {cid!r}.")
        shuffled = pool[:]
        rng.shuffle(shuffled)
        class_pools[cid] = shuffled

    out: List[int] = []
    for i, cid in enumerate(class_ids):
        samples_needed = per_class + (1 if i < remainder else 0)
        if samples_needed == 0:
            continue
        pool = class_pools[cid]
        if len(pool) >= samples_needed:
            out.extend(pool[:samples_needed])
        else:
            out.extend(pool)
            out.extend(rng.choices(pool, k=samples_needed - len(pool)))
    rng.shuffle(out)
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


def _normalize_param_name(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def _base_forward_model(model_with_gradiend: Any) -> Any:
    get_base = getattr(model_with_gradiend, "_get_base_forward_model", None)
    if callable(get_base):
        return get_base()
    return getattr(model_with_gradiend, "base_model", None)


def _place_inputs_for_base_forward(model_with_gradiend: Any, inputs: Any) -> Any:
    place = getattr(model_with_gradiend, "_place_inputs_for_base_forward", None)
    if callable(place):
        inputs = place(inputs)
    if isinstance(inputs, dict):
        return {
            k: (
                v.unsqueeze(0)
                if torch.is_tensor(v) and v.ndim == 1
                else (v.squeeze(dim=1) if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] == 1 else v)
            )
            for k, v in inputs.items()
        }
    return inputs


def _zero_base_grad(model_with_gradiend: Any) -> None:
    zero = getattr(model_with_gradiend, "_zero_base_grad", None)
    if callable(zero):
        zero(set_to_none=True)
        return
    base = _base_forward_model(model_with_gradiend)
    if base is not None and hasattr(base, "zero_grad"):
        try:
            base.zero_grad(set_to_none=True)
        except TypeError:
            base.zero_grad()


def _backward_loss(model_with_gradiend: Any, loss: torch.Tensor) -> None:
    backward = getattr(model_with_gradiend, "_backward_through_base_model", None)
    if callable(backward):
        backward(loss)
    else:
        loss.backward()


def _run_base_backward(model_with_gradiend: Any, inputs: Any) -> None:
    base = _base_forward_model(model_with_gradiend)
    if base is None:
        raise ValueError("Streaming pre-prune requires a model with a base forward model.")
    inputs = _place_inputs_for_base_forward(model_with_gradiend, inputs)
    if not isinstance(inputs, dict):
        raise TypeError("Streaming pre-prune currently requires dict inputs for the base model.")
    outputs = base(**inputs)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise ValueError("Streaming pre-prune requires base model outputs with a loss.")
    _backward_loss(model_with_gradiend, loss)


def _param_lookup(model: Any) -> Dict[str, torch.nn.Parameter]:
    base = model.module if hasattr(model, "module") else model
    lookup: Dict[str, torch.nn.Parameter] = {}
    for name, param in base.named_parameters():
        lookup[name] = param
        lookup.setdefault(_normalize_param_name(name), param)
        lookup.setdefault(f"module.{_normalize_param_name(name)}", param)
    return lookup


def _resolve_param(lookup: Dict[str, torch.nn.Parameter], name: str) -> torch.nn.Parameter:
    for candidate in (name, _normalize_param_name(name), f"module.{name}"):
        param = lookup.get(candidate)
        if param is not None:
            return param
    raise KeyError(f"Parameter {name!r} not found in base model parameters.")


def _resolve_topk(topk: Union[int, float], input_dim: int) -> int:
    if isinstance(topk, bool):
        raise TypeError("topk must be int or float, not bool")
    if isinstance(topk, float):
        if not (0.0 < topk <= 1.0):
            raise ValueError("topk as float must be in (0, 1].")
        return max(1, int(torch.ceil(torch.tensor(float(topk) * input_dim)).item()))
    if isinstance(topk, int):
        if topk <= 0:
            raise ValueError("topk as int must be >= 1 for streaming pre-prune.")
        return min(topk, input_dim)
    raise TypeError(f"topk must be int or float, got {type(topk).__name__}")


def _streaming_topk_from_accumulators(
    accumulators: Dict[str, torch.Tensor],
    selectors: List[Any],
    *,
    k: int,
    count: int,
    chunk_size: int = 8_000_000,
) -> torch.Tensor:
    """Select an exact global top-k without materializing candidates on the CPU.

    A chunked ``torch.topk`` is only memory efficient when ``k`` itself is small.  For
    percentage pruning of very large models (for example 1% of a 70B model), the old
    merge kept billions of float32 values and int64 indices on the host.  Instead we
    find the kth score with two GPU-side radix histograms over the bits of the
    non-negative float32 scores.  A final scan transfers only the selected indices.

    The returned indices are unique and sorted in input-space order.  Ties at the kth
    score are resolved by retaining the lowest input-space indices.
    """
    if count <= 0:
        raise ValueError("count must be positive when selecting streaming top-k.")

    segments: List[Tuple[torch.Tensor, int, int]] = []
    offset = 0
    for selector in selectors:
        n = int(selector.num_selected)
        acc = accumulators.get(selector.name)
        if n == 0:
            offset += n
            continue
        if acc is None:
            raise RuntimeError(
                f"GRADIEND parameter {selector.name!r} is included in the param_map but received no "
                "gradient from the configured gradient creation method during streaming pre-prune. "
                "This usually means the parameter is not used by the model forward/loss. Exclude it "
                "from params/param_map, or fix backbone/head detection if it was included by default."
            )
        segments.append((acc.detach().flatten(), offset, n))
        offset += n

    if not segments:
        raise RuntimeError("No gradients accumulated in streaming pre-prune.")
    if not 0 < k <= offset:
        raise ValueError(f"k must be in [1, {offset}], got {k}.")

    def _score_bits(flat: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        # Preserve the old comparison semantics (division in accumulator dtype), then
        # use float32 only as an order-preserving bit representation for radix select.
        scores = (flat[start:stop] / float(count)).abs().float()
        return scores.contiguous().view(torch.int32)

    def _histogram(*, high16: Optional[int] = None) -> torch.Tensor:
        # One tiny histogram per device lets CUDA devices work concurrently.  Host
        # synchronization happens only after every chunk has been queued.
        per_device: Dict[torch.device, torch.Tensor] = {}
        for flat, _, n in segments:
            hist = per_device.get(flat.device)
            if hist is None:
                hist = torch.zeros(65_536, dtype=torch.int64, device=flat.device)
                per_device[flat.device] = hist
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                bits = _score_bits(flat, start, stop)
                if high16 is None:
                    buckets = (bits >> 16) & 0xFFFF
                else:
                    matching = (bits >> 16) == high16
                    buckets = bits[matching] & 0xFFFF
                if buckets.numel():
                    hist.add_(torch.bincount(buckets.long(), minlength=65_536))

        total = torch.zeros(65_536, dtype=torch.int64)
        for hist in per_device.values():
            total.add_(hist.cpu())
        return total

    def _bucket_for_rank(hist: torch.Tensor, rank: int) -> Tuple[int, int]:
        # rank is one-based among scores in descending order.  Return the selected
        # bucket and the one-based rank within that bucket.
        descending_cumulative = hist.flip(0).cumsum(0)
        position = int(
            torch.searchsorted(
                descending_cumulative,
                torch.tensor(rank, dtype=descending_cumulative.dtype),
                right=False,
            ).item()
        )
        if position >= int(hist.numel()):
            raise RuntimeError("Radix top-k histogram did not contain the requested rank.")
        higher_count = 0 if position == 0 else int(descending_cumulative[position - 1].item())
        return int(hist.numel()) - 1 - position, rank - higher_count

    logger.info("Streaming pre-prune: GPU radix-select pass 1/3 (high score bits).")
    high16, rank_within_high = _bucket_for_rank(_histogram(), k)
    logger.info("Streaming pre-prune: GPU radix-select pass 2/3 (low score bits).")
    low16, equal_to_keep = _bucket_for_rank(_histogram(high16=high16), rank_within_high)
    threshold_bits = (high16 << 16) | low16

    logger.info("Streaming pre-prune: GPU radix-select pass 3/3 (selected indices).")
    try:
        keep_idx = torch.empty(k, dtype=torch.long, device="cpu")
    except (RuntimeError, MemoryError) as exc:
        required = k * torch.empty((), dtype=torch.long).element_size()
        raise RuntimeError(
            f"Could not allocate the final top-k index vector ({_format_bytes(required)} for {format_count(k)} indices)."
        ) from exc

    written = 0
    equal_written = 0
    for flat, segment_offset, n in segments:
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            bits = _score_bits(flat, start, stop)
            selected = bits > threshold_bits
            remaining_equal = equal_to_keep - equal_written
            if remaining_equal > 0:
                equal_positions = (bits == threshold_bits).nonzero(as_tuple=False).flatten()
                take = min(remaining_equal, int(equal_positions.numel()))
                if take:
                    selected[equal_positions[:take]] = True
                    equal_written += take
            local_idx = selected.nonzero(as_tuple=False).flatten()
            n_selected = int(local_idx.numel())
            if n_selected:
                keep_idx[written : written + n_selected].copy_(
                    local_idx.to(dtype=torch.long, device="cpu").add_(segment_offset + start)
                )
                written += n_selected

    if written != k or equal_written != equal_to_keep:
        raise RuntimeError(
            "GPU radix top-k produced an inconsistent selection: "
            f"written={written}, expected={k}, equal_written={equal_written}, "
            f"equal_expected={equal_to_keep}."
        )
    return keep_idx


def _pre_prune_streaming_topk(
    model_with_gradiend: Any,
    data: Any,
    config: "PrePruneConfig",
    indices: List[int],
    *,
    inplace: bool,
    return_mask: bool,
    return_keep_idx: bool = False,
    runtime_monitor: Optional[Any] = None,
) -> Any:
    gradiend = model_with_gradiend.gradiend
    if not hasattr(gradiend, "_get_compiled_param_selectors"):
        raise ValueError("Streaming pre-prune requires a mapping-aware GRADIEND model.")
    selectors = [s for s in gradiend._get_compiled_param_selectors() if int(s.num_selected) > 0]
    input_dim = int(gradiend.input_dim)
    k = _resolve_topk(config.topk, input_dim)

    base = _base_forward_model(model_with_gradiend)
    if base is None:
        raise ValueError("Streaming pre-prune requires a base model.")
    lookup = _param_lookup(base)

    accumulators: Dict[str, torch.Tensor] = {}
    handles = []
    current_sign = [1.0]

    def _make_hook(name: str, selector: Any):
        def _hook(grad: torch.Tensor):
            selected = selector.select_flat(grad.detach().flatten())
            if current_sign[0] < 0:
                selected = selected.neg()
            if name in accumulators:
                accumulators[name].add_(selected)
            else:
                accumulators[name] = selected.clone()
            return grad

        return _hook

    mapped_params = []
    for selector in selectors:
        param = _resolve_param(lookup, selector.name)
        mapped_params.append(param)
        handles.append(param.register_hook(_make_hook(selector.name, selector)))
        if hasattr(param, "register_post_accumulate_grad_hook"):
            def _clear_grad(p: torch.nn.Parameter):
                p.grad = None

            handles.append(param.register_post_accumulate_grad_hook(_clear_grad))

    source = config.source
    requires_factual = source in factual_computation_required_keywords
    requires_alternative = source in alternative_computation_required_keywords
    count = 0

    logger.info(
        "Streaming pre-prune: collecting top-%s over %s input dimensions from %s sample(s).",
        format_count(k),
        format_count(input_dim),
        format_count(len(indices)),
    )
    if runtime_monitor is not None:
        runtime_monitor.mark(
            "pre_prune:streaming_topk:start",
            input_dim=input_dim,
            k=k,
            n_samples=len(indices),
            source=source,
        )

    try:
        for idx in gradiend_tqdm(
            indices,
            total=len(indices),
            desc="Pre-prune",
            unit="datapoint",
            leave=True,
            ncols=80,
            position=0,
        ):
            item = data[idx]
            if requires_factual:
                if runtime_monitor is not None:
                    runtime_monitor.mark("pre_prune:backward", sample=count + 1, kind="factual")
                current_sign[0] = 1.0
                _zero_base_grad(model_with_gradiend)
                _run_base_backward(model_with_gradiend, item["factual"])
            if requires_alternative:
                if runtime_monitor is not None:
                    runtime_monitor.mark("pre_prune:backward", sample=count + 1, kind="alternative")
                current_sign[0] = -1.0 if source == "diff" else 1.0
                _zero_base_grad(model_with_gradiend)
                _run_base_backward(model_with_gradiend, item["alternative"])
            _zero_base_grad(model_with_gradiend)
            count += 1
            if count == 1:
                missing = [selector.name for selector in selectors if selector.name not in accumulators]
                if missing:
                    raise RuntimeError(
                        "Some GRADIEND parameters are included in the param_map but received no gradient "
                        "from the configured gradient creation method on the first pre-prune sample: "
                        f"{missing}. This usually means these parameters are not used by the model "
                        "forward/loss. Exclude them from params/param_map, or fix backbone/head detection "
                        "if they were included by default."
                    )
    finally:
        for handle in handles:
            handle.remove()
        for param in mapped_params:
            param.grad = None

    if runtime_monitor is not None:
        runtime_monitor.mark("pre_prune:streaming_topk:select", input_dim=input_dim, k=k, count=count)
    keep_idx = _streaming_topk_from_accumulators(accumulators, selectors, k=k, count=count)
    del accumulators
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        "Streaming pre-prune: selected %s/%s input dimensions.",
        format_count(keep_idx.numel()),
        format_count(input_dim),
    )
    if runtime_monitor is not None:
        runtime_monitor.mark("pre_prune:prune_gradiend", kept=int(keep_idx.numel()), input_dim=input_dim)
    pruned = model_with_gradiend.prune_gradiend(
        keep_idx=keep_idx,
        keep_idx_sorted_unique=True,
        inplace=inplace,
        return_mask=return_mask,
    )
    if return_keep_idx:
        return pruned, keep_idx.detach().cpu().long()
    return pruned


def _streaming_pre_prune_eligible(
    config: PrePruneConfig,
    *,
    can_stream_inputs: bool,
    can_stream_model: bool,
) -> bool:
    return (
        config.topk is not None
        and config.threshold is None
        and not (isinstance(config.topk, float) and config.topk == 1.0)
        and can_stream_inputs
        and can_stream_model
    )


def _resolve_pre_prune_use_streaming(
    config: PrePruneConfig,
    *,
    can_stream_inputs: bool,
    can_stream_model: bool,
) -> bool:
    env = os.environ.get("GRADIEND_PREPRUNE_STREAMING")
    if env is not None:
        normalized = env.strip().lower()
        if normalized in ("0", "false", "no", "off", "classic"):
            return False
        if normalized in ("1", "true", "yes", "on", "streaming"):
            return True
        raise ValueError(
            "GRADIEND_PREPRUNE_STREAMING must be one of "
            "0|false|classic or 1|true|streaming; "
            f"got {env!r}"
        )
    if config.use_streaming is not None:
        return config.use_streaming
    return _streaming_pre_prune_eligible(
        config,
        can_stream_inputs=can_stream_inputs,
        can_stream_model=can_stream_model,
    )


def pre_prune(
    model_with_gradiend: Any,
    dataset: Any,
    config: PrePruneConfig,
    *,
    definition: Optional[Any] = None,
    inplace: bool = True,
    return_mask: bool = False,
    return_keep_idx: bool = False,
    runtime_monitor: Optional[Any] = None,
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
    if _is_noop_pre_prune(config):
        logger.debug("Pre-prune: topk=1.0 with no threshold — skipping (no dimensions removed).")
        if return_keep_idx or return_mask:
            gradiend = model_with_gradiend.gradiend
            input_dim = int(getattr(gradiend, "input_dim", 0) or 0)
            keep_idx = torch.arange(input_dim, dtype=torch.long)
            m = model_with_gradiend if inplace else copy.deepcopy(model_with_gradiend)
            if return_keep_idx:
                return m, keep_idx
            full_mask = torch.ones(input_dim, dtype=torch.bool)
            return m, full_mask
        return model_with_gradiend if inplace else copy.deepcopy(model_with_gradiend)

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
        config.seed,
    )
    if runtime_monitor is not None:
        runtime_monitor.mark(
            "pre_prune:start",
            n_indices=len(indices),
            source=source,
            topk=config.topk,
            threshold=config.threshold,
            input_dim=int(getattr(gradiend, "input_dim", 0) or 0),
        )

    can_stream_inputs = bool(indices) and isinstance(data[indices[0]].get("factual"), dict)
    can_stream_model = _base_forward_model(model_with_gradiend) is not None
    use_streaming = _resolve_pre_prune_use_streaming(
        config,
        can_stream_inputs=can_stream_inputs,
        can_stream_model=can_stream_model,
    )
    if use_streaming:
        if not _streaming_pre_prune_eligible(
            config,
            can_stream_inputs=can_stream_inputs,
            can_stream_model=can_stream_model,
        ):
            raise ValueError(
                "PrePruneConfig.use_streaming=True (or GRADIEND_PREPRUNE_STREAMING=streaming) "
                "but streaming pre-prune is not eligible for this model/dataset/config."
            )
        logger.debug("Pre-prune: using streaming implementation.")
        return _pre_prune_streaming_topk(
            model_with_gradiend,
            data,
            config,
            indices,
            inplace=inplace,
            return_mask=return_mask,
            return_keep_idx=return_keep_idx,
            runtime_monitor=runtime_monitor,
        )

    logger.debug("Pre-prune: using classic gradient_creator implementation.")
    _guard_pre_prune_memory(gradiend, source)

    running_sum: Optional[torch.Tensor] = None
    count = 0

    _tqdm_kw = dict(
        total=len(indices),
        desc="Pre-prune",
        unit="datapoint",
        leave=True,
        ncols=80,
        position=0,
    )
    for idx in gradiend_tqdm(indices, **_tqdm_kw):
        item = data[idx]
        factual_in = item["factual"]
        alternative_in = item["alternative"]

        fact_g = None
        if requires_factual:
            try:
                fact_g = gradient_creator(factual_in, target_device=torch.device("cpu"))
            except TypeError:
                # Backwards compatibility for gradient_creator(text) callables
                fact_g = gradient_creator(factual_in)
        alt_g = None
        if requires_alternative:
            try:
                alt_g = gradient_creator(alternative_in, target_device=torch.device("cpu"))
            except TypeError:
                alt_g = gradient_creator(alternative_in)

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
        "return_mask": return_mask or return_keep_idx,
    }
    if runtime_monitor is not None:
        runtime_monitor.mark("pre_prune:prune_gradiend", count=count, input_dim=int(importance.numel()))
    result = model_with_gradiend.prune_gradiend(**kwargs)
    if return_keep_idx or return_mask:
        pruned, mask = result
        if return_keep_idx:
            keep_idx = mask.nonzero(as_tuple=False).flatten().cpu().long()
            return pruned, keep_idx
        return pruned, mask
    return result


def _pre_prune_config_dict(config: PrePruneConfig) -> Dict[str, Any]:
    raw = dataclasses.asdict(config)
    raw["dataset"] = None
    return raw


def build_pre_prune_cache_meta(model_with_gradiend: Any, config: PrePruneConfig) -> Dict[str, Any]:
    gradiend = getattr(model_with_gradiend, "gradiend", None)
    base_id = getattr(model_with_gradiend, "name_or_path", None) or getattr(
        getattr(model_with_gradiend, "base_model", None), "name_or_path", None
    )
    return {
        "pre_prune_config": _pre_prune_config_dict(config),
        "input_dim": int(getattr(gradiend, "input_dim", 0) or 0),
        "base_model_id": str(base_id or ""),
        "param_map_hash": getattr(gradiend, "param_map_hash", None),
    }


def _validate_pre_prune_cache_meta(meta: Dict[str, Any], model_with_gradiend: Any, config: PrePruneConfig) -> None:
    from gradiend.trainer.core.cache_policy import _normalize_pre_prune_config

    expected = build_pre_prune_cache_meta(model_with_gradiend, config)
    cached_cfg = _normalize_pre_prune_config(meta.get("pre_prune_config"))
    expected_cfg = _normalize_pre_prune_config(expected["pre_prune_config"])
    if cached_cfg != expected_cfg:
        raise ValueError("Pre-prune cache config does not match current PrePruneConfig.")
    if int(meta.get("input_dim", -1)) != int(expected["input_dim"]):
        raise ValueError(
            f"Pre-prune cache input_dim={meta.get('input_dim')} does not match current model "
            f"input_dim={expected['input_dim']}."
        )


def save_pre_prune_cache(cache_dir: str, keep_idx: torch.Tensor, meta: Dict[str, Any]) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    keep_path = os.path.join(cache_dir, "keep_idx.pt")
    meta_path = os.path.join(cache_dir, "pre_prune_meta.json")
    torch.save(keep_idx.detach().cpu().long(), keep_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    logger.info("Saved pre-prune cache to %s", cache_dir)
    return cache_dir


def load_pre_prune_cache(cache_dir: str) -> Tuple[Dict[str, Any], torch.Tensor]:
    meta_path = os.path.join(cache_dir, "pre_prune_meta.json")
    keep_path = os.path.join(cache_dir, "keep_idx.pt")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    keep_idx = torch.load(keep_path, weights_only=True)
    if not torch.is_tensor(keep_idx):
        raise ValueError(f"Invalid keep_idx cache at {keep_path}")
    return meta, keep_idx.detach().cpu().long()


def pre_prune_with_cache(
    model_with_gradiend: Any,
    dataset: Any,
    config: PrePruneConfig,
    *,
    definition: Optional[Any] = None,
    cache_dir: Optional[str] = None,
    reuse_cache: bool = False,
    inplace: bool = True,
    runtime_monitor: Optional[Any] = None,
) -> Any:
    """
    Run pre-prune, optionally reusing a fixed-path cache under experiment_dir/cache/pre_prune.

    When reuse_cache=True and a valid cache exists, applies cached keep_idx without recomputing gradients.
    """
    if _is_noop_pre_prune(config):
        logger.debug("Pre-prune cache: topk=1.0 with no threshold — skipping pre-prune and cache.")
        return model_with_gradiend if inplace else copy.deepcopy(model_with_gradiend)

    if cache_dir and reuse_cache and should_use_cached(cache_dir, True) and has_saved_pre_prune_cache(cache_dir):
        meta, keep_idx = load_pre_prune_cache(cache_dir)
        try:
            _validate_pre_prune_cache_meta(meta, model_with_gradiend, config)
        except ValueError as exc:
            logger.warning(
                "Pre-prune cache at %s is stale (%s); recomputing pre-prune.",
                cache_dir,
                exc,
            )
        else:
            logger.info("Reusing cached pre-prune result from %s", cache_dir)
            if runtime_monitor is not None:
                runtime_monitor.mark("pre_prune:cache_hit", cache_dir=cache_dir)
            return model_with_gradiend.prune_gradiend(keep_idx=keep_idx, inplace=inplace)

    meta = build_pre_prune_cache_meta(model_with_gradiend, config)
    model_out, keep_idx = pre_prune(
        model_with_gradiend,
        dataset,
        config,
        definition=definition,
        inplace=inplace,
        return_keep_idx=True,
        runtime_monitor=runtime_monitor,
    )
    if cache_dir and reuse_cache:
        save_pre_prune_cache(cache_dir, keep_idx, meta)
    return model_out
