"""Symmetric pair trainer-suite specialization."""

from __future__ import annotations

from .base import TrainerSuite
from .definitions import *


def _canonical_symmetric_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
    """Return unordered pair classes in stable alphabetical order."""
    a, b = str(pair[0]), str(pair[1])
    return (a, b) if a <= b else (b, a)


class SymmetricTrainerSuite(TrainerSuite):
    """TrainerSuite for symmetric pair semantics such as variable contrasts."""

    def compute_anchor_aligned_encoding_matrix(
        self,
        feature_classes: Sequence[str],
        *,
        encoder_summary: Optional[Dict[str, Any]] = None,
        split: str = "test",
        max_size: Optional[int] = None,
        use_cache: bool = True,
        full_eval: bool = True,
        aggregate: str = "mean",
        alignment: str = "factual",
        column_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Feature-class cross-encoding for symmetric pairs.

        Rows are anchor feature classes (aggregated across GRADIENDs whose pair
        contains that class, with automatic sign alignment). Columns are evaluated
        feature classes.

        Args:
            feature_classes: Ordered feature classes used as matrix rows/columns.
            encoder_summary: Optional precomputed suite encoder result.
            split: Encoder split used when running evaluation.
            max_size: Optional encoder-evaluation cap.
            use_cache: Whether to use cached encoder results.
            full_eval: Whether encoder evaluation includes all transitions.
            aggregate: Aggregate used when multiple pair models cover one anchor.
            alignment: Column alignment mode.
            column_ids: Optional explicit output columns.
        """
        if encoder_summary is None:
            encoder_summary = self.evaluate_encoder(
                split=split,
                max_size=max_size,
                use_cache=use_cache,
                plot=False,
                return_df=True,
                full_eval=full_eval,
            )
        return compute_anchor_aligned_encoding_matrix(
            pair_by_id=self.pair_by_id,
            encoder_summary=encoder_summary,
            feature_classes=feature_classes,
            aggregate=aggregate,
            alignment=alignment,
            column_ids=column_ids,
            source_by_id=source_by_id_from_trainers(self.trainers),
        )

    def plot_cross_encoding_heatmap(
        self,
        feature_classes: Sequence[str],
        *,
        alignment: str = "factual",
        column_ids: Optional[Sequence[str]] = None,
        encoder_summary: Optional[Dict[str, Any]] = None,
        split: str = "test",
        max_size: Optional[int] = None,
        use_cache: bool = True,
        full_eval: bool = True,
        cross_task_eval: bool = False,
        aggregate: str = "mean",
        order: Any = "input",
        cluster: bool = False,
        pretty_groups: Optional[Dict[str, List[str]]] = None,
        **plot_kwargs: Any,
    ) -> Dict[str, Any]:
        """Plot oriented cross-encoding heatmap for symmetric pairwise GRADIENDs.

        Args:
            feature_classes: Ordered feature classes used as matrix row anchors.
            alignment: Column alignment mode (``factual``, ``counterfactual``,
                or ``transition``).
            column_ids: Optional explicit output columns.
            encoder_summary: Optional precomputed suite encoder result.
            split: Encoder split used when running evaluation.
            max_size: Optional encoder-evaluation cap.
            use_cache: Whether to use cached encoder results.
            full_eval: Whether encoder evaluation includes all transitions.
            cross_task_eval: Use shared per-class test pool across trainers.
            aggregate: Aggregate used when multiple pair models cover one anchor.
            order: Heatmap ordering strategy or explicit order.
            cluster: If True, cluster heatmap rows/columns.
            pretty_groups: Optional display groups.
            **plot_kwargs: Forwarded to comparison heatmap plotting.
        """
        from gradiend.visualizer.heatmaps.encoding import plot_cross_encoding_heatmap

        return plot_cross_encoding_heatmap(
            self.trainers,
            feature_classes,
            alignment=alignment,
            column_ids=column_ids,
            encoder_summary=encoder_summary,
            split=split,
            max_size=max_size,
            use_cache=use_cache,
            full_eval=full_eval,
            cross_task_eval=cross_task_eval,
            aggregate=aggregate,
            order=order,
            cluster=cluster,
            pretty_groups=pretty_groups,
            **plot_kwargs,
        )

    def _resolve_pair_definitions(self) -> List[SuitePairDefinition]:
        if self._input_pair_definitions is not None:
            return list(self._input_pair_definitions)
        candidate_pairs = self.target_pairs or list(combinations(self.target_classes, 2))
        definitions: List[SuitePairDefinition] = []
        for pair in candidate_pairs:
            canon = _canonical_symmetric_pair((str(pair[0]), str(pair[1])))
            definitions.append(
                SuitePairDefinition(
                    target_classes=canon,
                    child_id=self.pair_id_fn(canon),
                    label=self.pair_label_fn(canon),
                )
            )
        return definitions
