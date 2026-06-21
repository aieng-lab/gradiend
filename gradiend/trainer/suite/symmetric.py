"""Symmetric pair trainer-suite specialization."""

from __future__ import annotations

from .base import TrainerSuite
from .definitions import *


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
        )

    def plot_anchor_aligned_encoding_heatmap(
        self,
        feature_classes: Sequence[str],
        *,
        encoder_summary: Optional[Dict[str, Any]] = None,
        split: str = "test",
        max_size: Optional[int] = None,
        use_cache: bool = True,
        full_eval: bool = True,
        aggregate: str = "mean",
        order: Any = "input",
        cluster: bool = False,
        pretty_groups: Optional[Dict[str, List[str]]] = None,
        **plot_kwargs: Any,
    ) -> Dict[str, Any]:
        """Plot anchor-aligned symmetric cross-encoding heatmap.

        Args:
            feature_classes: Ordered feature classes used as matrix rows/columns.
            encoder_summary: Optional precomputed suite encoder result.
            split: Encoder split used when running evaluation.
            max_size: Optional encoder-evaluation cap.
            use_cache: Whether to use cached encoder results.
            full_eval: Whether encoder evaluation includes all transitions.
            aggregate: Aggregate used when multiple pair models cover one anchor.
            order: Heatmap ordering strategy or explicit order.
            cluster: If True, cluster heatmap rows/columns.
            pretty_groups: Optional display groups.
            **plot_kwargs: Forwarded to comparison heatmap plotting.
        """
        comparison_data = self.compute_anchor_aligned_encoding_matrix(
            feature_classes,
            encoder_summary=encoder_summary,
            split=split,
            max_size=max_size,
            use_cache=use_cache,
            full_eval=full_eval,
            aggregate=aggregate,
        )
        return plot_comparison_heatmap(
            comparison_data,
            order=order if order != "input" else list(feature_classes),
            cluster=cluster,
            pretty_groups=pretty_groups,
            **plot_kwargs,
        )

    def _resolve_pair_definitions(self) -> List[SuitePairDefinition]:
        if self._input_pair_definitions is not None:
            return list(self._input_pair_definitions)
        candidate_pairs = self.target_pairs or list(combinations(self.target_classes, 2))
        return [
            SuitePairDefinition(
                target_classes=(str(pair[0]), str(pair[1])),
                child_id=self.pair_id_fn((str(pair[0]), str(pair[1]))),
                label=self.pair_label_fn((str(pair[0]), str(pair[1]))),
            )
            for pair in candidate_pairs
        ]


