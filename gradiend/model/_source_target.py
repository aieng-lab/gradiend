"""Source/target keywords and decoder-eval contracts.

Training vs decoder evaluation
------------------------------

* **Training** uses ``TrainingArguments.source`` for which gradients feed the encoder

  (``factual``, ``alternative``, or ``diff``).

* **Decoder rewrite** uses ``model.source`` (persisted in ``gradiend_context.json``) to

  pick the default ``feature_factor`` sign per class. ``model.source`` must match
  ``TrainingArguments.source``; see :func:`sync_model_source_target_from_training_args`.

Feature-factor sign (strengthen class ``C``)
--------------------------------------------
Rewrite is ``base + learning_rate * decoder(feature_factor)``. Learning rate is never
negated by source.

+------------------+-------------------------------+
| ``model.source`` | default ``ff`` for class C  |
+==================+===============================+
| factual, diff    | ``-encoding_direction[C]``    |
| alternative      | ``+encoding_direction[C]``    |
+------------------+-------------------------------+

``encoding_direction`` is ``model.feature_class_encoding_direction`` (from training pair:
``pair[0] -> +1``, ``pair[1] -> -1``).

Decoder grid / plots
--------------------

* Strengthen ``target_class="3SG"`` evaluates only ``ff = derive(3SG)``.
* Summary and probability-shift plots use that same ``ff`` (not another class's sign).

Cross-encoding heatmaps (anchor-aligned)
--------------------------------------
Column alignment (``factual`` vs ``counterfactual``) must match how the model was
trained. When ``model.source="alternative"`` but columns are factual-aligned (or
the reverse), encoded values are multiplied by ``-1`` — same XOR rule as decoder
``feature_factor`` (see :func:`encoding_view_sign_for_source`).
"""

from __future__ import annotations

from typing import Any, Optional

SOURCE_TARGET_KEYWORDS: frozenset[str] = frozenset({"factual", "alternative", "diff"})


def validate_source_target(name: str, value: object) -> str:
    """Validate a source or target keyword."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    if value not in SOURCE_TARGET_KEYWORDS:
        raise ValueError(
            f"{name} must be one of {sorted(SOURCE_TARGET_KEYWORDS)!r}, got {value!r}"
        )
    return value


def feature_factor_from_encoding_direction(direction: float, source: str) -> float:
    """Map ``feature_class_encoding_direction[class]`` to decoder-eval ``feature_factor``."""
    validate_source_target("source", source)
    if source == "alternative":
        return float(direction)
    return float(-direction)


def encoding_view_sign_for_source(source: str, alignment: str) -> float:
    """Sign to map encoder outputs into a factual- or counterfactual-aligned plot.

    Returns ``+1`` when ``model.source`` matches the column view, ``-1`` when they
    differ (e.g. ``source=alternative`` with ``alignment=factual``). Transition
    alignment is unchanged (``+1``).
    """
    validate_source_target("source", source)
    key = str(alignment).strip().lower()
    if key in {"transition", "transitions"}:
        return 1.0
    counterfactual_view = key in {
        "counterfactual", "cf", "alternative", "alternatives",
    }
    alternative_source = source == "alternative"
    return -1.0 if counterfactual_view != alternative_source else 1.0


def training_arg_value(training_args: Any, key: str, default: Any = None) -> Any:
    """Read one field from TrainingArguments or a plain dict."""
    if training_args is None:
        return default
    if isinstance(training_args, dict):
        return training_args.get(key, default)
    return getattr(training_args, key, default)


def resolve_model_source(model: Any, trainer: Any = None, *, default: str = "factual") -> str:
    """Return ``model.source``, else ``TrainingArguments.source``, else *default*."""
    source = getattr(model, "source", None) if model is not None else None
    if source is not None:
        return validate_source_target("source", source)
    if trainer is not None:
        args = getattr(trainer, "_training_args", None) or getattr(trainer, "training_args", None)
        args_source = training_arg_value(args, "source", None)
        if args_source is not None:
            return validate_source_target("source", args_source)
    return default


def sync_model_source_target_from_training_args(
    model: Any,
    training_args: Any,
    *,
    log_mismatch: bool = True,
) -> None:
    """Set ``model._source`` / ``model._target`` from TrainingArguments when they differ."""
    if model is None or training_args is None:
        return
    from gradiend.util.logging import get_logger

    log = get_logger(__name__)
    for key, private_attr in (("source", "_source"), ("target", "_target")):
        expected = training_arg_value(training_args, key, None)
        if expected is None:
            continue
        validate_source_target(key, expected)
        current = getattr(model, key, None)
        if current == expected:
            continue
        if log_mismatch:
            log.info(
                "Updating model.%s from %r to %r (TrainingArguments)",
                key,
                current,
                expected,
            )
        setattr(model, private_attr, expected)
