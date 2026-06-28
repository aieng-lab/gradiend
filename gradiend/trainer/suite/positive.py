"""Positive / true-vs-false trainer-suite specialization."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

from .base import TrainerSuite
from .definitions import *


@contextmanager
def _quiet_expected_suite_reload(trainers: Dict[str, Any]):
    previous_flags = {
        trainer: getattr(trainer, "_suppress_expected_reload_warning", False)
        for trainer in trainers.values()
    }
    previous_transformers_verbosity = None
    try:
        from transformers.utils import logging as transformers_logging

        previous_transformers_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
    except Exception:
        transformers_logging = None
    try:
        for trainer in trainers.values():
            setattr(trainer, "_suppress_expected_reload_warning", True)
        yield
    finally:
        for trainer, previous in previous_flags.items():
            setattr(trainer, "_suppress_expected_reload_warning", previous)
        if previous_transformers_verbosity is not None:
            transformers_logging.set_verbosity(previous_transformers_verbosity)


class PositiveTrainerSuite(TrainerSuite):
    """TrainerSuite for ordered / positive-vs-negative pair semantics."""

    def __init__(
        self,
        trainer_cls: Type[Trainer],
        *trainer_args: Any,
        mode: str = "single",
        positive_feature_definitions: Optional[Sequence[Any]] = None,
        negative_class_fn: Optional[Callable[[str], str]] = None,
        **trainer_kwargs: Any,
    ) -> None:
        self.mode = str(mode).strip().lower()
        if self.mode not in {"single", "all_but_one"}:
            raise ValueError("PositiveTrainerSuite mode must be one of: single, all_but_one")
        self.negative_class_fn = negative_class_fn or (lambda value: f"non_{value}")
        self.positive_feature_definitions = _normalize_positive_feature_definitions(positive_feature_definitions)
        super().__init__(trainer_cls, *trainer_args, **trainer_kwargs)

    def _resolved_positive_feature_definitions(self) -> List[PositiveFeatureDefinition]:
        if self.positive_feature_definitions is not None:
            features = list(self.positive_feature_definitions)
        elif self.target_pairs:
            seen = set()
            features = []
            for raw_pair in self.target_pairs:
                pair = (str(raw_pair[0]), str(raw_pair[1]))
                positive = _infer_positive_class_for_pair(pair)
                negative = pair[1] if positive == pair[0] else pair[0]
                if positive in seen:
                    continue
                seen.add(positive)
                features.append(
                    PositiveFeatureDefinition(
                        positive_feature_class=positive,
                        negative_feature_class=negative,
                        label=positive,
                    )
                )
        else:
            positives = [
                value for value in self.target_classes
                if not str(value).startswith("non_") and not str(value).startswith("non-")
            ]
            features = [
                PositiveFeatureDefinition(
                    positive_feature_class=str(positive),
                    negative_feature_class=str(self.negative_class_fn(str(positive))),
                    label=str(positive),
                )
                for positive in positives
            ]
        if not features:
            raise ValueError("PositiveTrainerSuite could not resolve any positive feature definitions")
        return features

    def _build_single_pair_definitions(self, features: List[PositiveFeatureDefinition]) -> List[SuitePairDefinition]:
        return [
            SuitePairDefinition(
                target_classes=(feature.positive_feature_class, feature.negative_feature_class),
                positive_class=feature.positive_feature_class,
                child_id=self.pair_id_fn((feature.positive_feature_class, feature.negative_feature_class)),
                label=self.pair_label_fn((feature.positive_feature_class, feature.negative_feature_class)),
            )
            for feature in features
        ]

    def _build_all_but_one_feature_pair_definitions(self, features: List[PositiveFeatureDefinition]) -> List[SuitePairDefinition]:
        definitions: List[SuitePairDefinition] = []
        for held_out in features:
            included = [feature for feature in features if feature.positive_feature_class != held_out.positive_feature_class]
            if not included:
                continue
            child_id = f"holdout__{_default_child_id((held_out.positive_feature_class, held_out.negative_feature_class))}"
            definitions.append(
                SuitePairDefinition(
                    target_classes=("positive", "negative"),
                    positive_class="positive",
                    child_id=child_id,
                    label=f"All But {held_out.label or held_out.positive_feature_class}",
                    class_merge_map={
                        "positive": [feature.positive_feature_class for feature in included],
                        "negative": [feature.negative_feature_class for feature in included],
                    },
                )
            )
        return definitions

    def _build_all_but_one_feature_class_group_pair_definitions(self, features: List[PositiveFeatureDefinition]) -> List[SuitePairDefinition]:
        groups: Dict[str, List[PositiveFeatureDefinition]] = {}
        for feature in features:
            if feature.feature_class_group is None:
                raise ValueError(
                    "PositiveTrainerSuite mode='all_but_one' uses feature-class-group holdouts only when every feature defines feature_class_group"
                )
            groups.setdefault(str(feature.feature_class_group), []).append(feature)
        definitions: List[SuitePairDefinition] = []
        for group_name, held_out_features in groups.items():
            held_out_positive = {feature.positive_feature_class for feature in held_out_features}
            included = [feature for feature in features if feature.positive_feature_class not in held_out_positive]
            if not included:
                continue
            definitions.append(
                SuitePairDefinition(
                    target_classes=("positive", "negative"),
                    positive_class="positive",
                    child_id=f"holdout_group__{group_name}",
                    label=f"All But {group_name}",
                    class_merge_map={
                        "positive": [feature.positive_feature_class for feature in included],
                        "negative": [feature.negative_feature_class for feature in included],
                    },
                )
            )
        return definitions

    def _resolve_pair_definitions(self) -> List[SuitePairDefinition]:
        if self._input_pair_definitions is not None:
            resolved: List[SuitePairDefinition] = []
            for definition in self._input_pair_definitions:
                positive_class = definition.positive_class or _infer_positive_class_for_pair(definition.target_classes)
                resolved.append(
                    SuitePairDefinition(
                        target_classes=definition.target_classes,
                        positive_class=positive_class,
                        child_id=definition.child_id,
                        label=definition.label,
                        class_merge_map=definition.class_merge_map,
                        class_merge_transition_groups=definition.class_merge_transition_groups,
                    )
                )
            return resolved
        features = self._resolved_positive_feature_definitions()
        if self.mode == "single":
            return self._build_single_pair_definitions(features)
        if self.mode == "all_but_one":
            features_with_group = [feature for feature in features if feature.feature_class_group is not None]
            if features_with_group and len(features_with_group) != len(features):
                raise ValueError(
                    "PositiveTrainerSuite mode='all_but_one' requires either no feature_class_group or a feature_class_group on every positive feature definition"
                )
            if features_with_group:
                return self._build_all_but_one_feature_class_group_pair_definitions(features)
            return self._build_all_but_one_feature_pair_definitions(features)
        raise ValueError(f"Unsupported PositiveTrainerSuite mode: {self.mode!r}")

    def _prepare_child_trainer_kwargs(
        self,
        child_kwargs: Dict[str, Any],
        *,
        definition: SuitePairDefinition,
        child_id: str,
    ) -> Dict[str, Any]:
        args = child_kwargs.get("args")
        if args is not None and hasattr(args, "include_other_classes"):
            args.include_other_classes = True
        return child_kwargs

    def evaluate_encoder(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        with _quiet_expected_suite_reload(self.trainers):
            return super().evaluate_encoder(*args, **kwargs)

    def compute_cross_encoding_matrix(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        full_eval: bool = True,
        run_evaluation: bool = True,
        allow_incomplete: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute positive-pair cross-encoding matrix.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            full_eval: Whether child encoder evaluation should include all
                available transitions.
            run_evaluation: Whether to run missing encoder evaluation first.
            allow_incomplete: If True, tolerate missing child encoder results.
            **kwargs: Forwarded to cross-encoding computation.
        """
        split = kwargs.get("split", "test")
        eval_use_cache = kwargs.get("use_cache", True)
        resolved_run_evaluation = bool(run_evaluation or not allow_incomplete)
        resolved_seed_selection = self._resolve_suite_seed_selection(kwargs.get("seed_selection"))
        if resolved_run_evaluation and resolved_seed_selection == "best":
            with _quiet_expected_suite_reload(self.trainers):
                self.evaluate_encoder(
                    split=split,
                    max_size=kwargs.get("max_size"),
                    use_cache=eval_use_cache,
                    plot=False,
                    return_df=False,
                    full_eval=full_eval,
                )
        kwargs = dict(kwargs)
        kwargs["seed_selection"] = resolved_seed_selection
        kwargs.setdefault(
            "positive_class_by_trainer",
            {
                child_id: definition.positive_class
                for child_id, definition in self.pair_definitions.items()
                if definition.positive_class is not None
            },
        )
        if kwargs.get("dispersion") is None:
            kwargs["dispersion"] = self._resolve_suite_dispersion(None)
        return compute_cross_encoding_matrix(
            self.trainers,
            run_evaluation=resolved_run_evaluation,
            allow_incomplete=allow_incomplete,
            full_eval=full_eval,
            **kwargs,
        )

    def plot_cross_encoding_heatmap(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        pretty_groups: Optional[Dict[str, List[str]]] = None,
        split: str = "test",
        max_size: Optional[int] = None,
        use_cache: bool = True,
        metric: str = "positive_mean",
        full_eval: bool = True,
        run_evaluation: bool = True,
        allow_incomplete: bool = False,
        seed_selection: Optional[str] = None,
        seed_aggregate: str = "mean",
        dispersion: Optional[str] = None,
        normalize: bool = False,
        order: Any = "input",
        cluster: bool = False,
        row_label_mapping: Optional[Dict[str, str]] = None,
        column_label_mapping: Optional[Dict[str, str]] = None,
        **plot_kwargs: Any,
    ) -> Dict[str, Any]:
        """Plot positive-pair cross-encoding heatmap.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            pretty_groups: Optional display groups.
            split: Encoder split used when running evaluation.
            max_size: Optional encoder-evaluation cap.
            use_cache: Whether to use child evaluation/model caches.
            metric: Cross-encoding metric to plot.
            full_eval: Whether child encoder evaluation includes all transitions.
            run_evaluation: Whether to run missing encoder evaluation first.
            allow_incomplete: If True, tolerate missing child encoder results.
            seed_selection: Optional seed selection for multi-seed children.
            seed_aggregate: Aggregate used for seed-level cross encoding.
            dispersion: Optional seed dispersion statistic.
            normalize: If True, normalize rows by diagonal values.
            order: Heatmap ordering strategy or explicit order.
            cluster: If True, cluster heatmap rows/columns.
            row_label_mapping: Optional row display labels.
            column_label_mapping: Optional column display labels.
            **plot_kwargs: Forwarded to comparison heatmap plotting.
        """
        effective_labels = self._effective_label_mapping(label_mapping, include_defaults=True)
        resolved_seed_selection = self._resolve_suite_seed_selection(seed_selection)
        resolved_dispersion = self._resolve_suite_dispersion(dispersion)
        comparison_data = self.compute_cross_encoding_matrix(
            label_mapping=effective_labels,
            split=split,
            max_size=max_size,
            use_cache=use_cache,
            metric=metric,
            full_eval=full_eval,
            run_evaluation=run_evaluation,
            allow_incomplete=allow_incomplete,
            seed_selection=resolved_seed_selection,
            seed_aggregate=seed_aggregate,
            dispersion=resolved_dispersion,
        )
        if normalize:
            comparison_data = _normalize_cross_encoding_rows_by_diagonal(comparison_data)
        return plot_comparison_heatmap(
            comparison_data,
            order=order,
            cluster=cluster,
            pretty_groups=pretty_groups,
            row_label_mapping=row_label_mapping,
            column_label_mapping=column_label_mapping,
            **plot_kwargs,
        )


