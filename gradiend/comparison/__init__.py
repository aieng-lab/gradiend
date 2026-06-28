"""Comparison utilities for inter-model GRADIEND analysis."""

from gradiend.comparison.common import comparison_matrix_from_cell_stat
from gradiend.comparison.anchor_aligned import (
    aggregate_anchor_aligned_encoding_rows,
    build_anchor_aligned_encoding_rows,
    compute_anchor_aligned_encoding_matrix,
    compute_anchor_aligned_encoding_std_matrix,
    compute_dense_anchor_aligned_encoding_matrix,
    pair_by_id_from_trainers,
    source_by_id_from_trainers,
)
from gradiend.comparison.cross_encoding import (
    can_normalize_cross_encoding_by_diagonal,
    compute_cross_encoding_matrix,
    normalize_cross_encoding_rows_by_diagonal,
)
from gradiend.comparison.encoder_aggregation import aggregate_encoder_dataframes
from gradiend.comparison.feature_cross_encoding import (
    compute_gradiend_feature_cross_encoding_matrix,
    compute_gradiend_transition_cross_encoding_matrix,
)
from gradiend.comparison.seed_policy import (
    enter_analysis_mode,
    enter_analysis_mode_for_trainers,
    evaluate_encoder_for_comparison,
    models_for_comparison,
)
from gradiend.comparison.similarity import (
    compute_grouped_similarity_matrices,
    compute_similarity_matrix,
)

__all__ = [
    "aggregate_anchor_aligned_encoding_rows",
    "build_anchor_aligned_encoding_rows",
    "comparison_matrix_from_cell_stat",
    "compute_anchor_aligned_encoding_matrix",
    "compute_anchor_aligned_encoding_std_matrix",
    "compute_dense_anchor_aligned_encoding_matrix",
    "compute_similarity_matrix",
    "compute_grouped_similarity_matrices",
    "compute_cross_encoding_matrix",
    "can_normalize_cross_encoding_by_diagonal",
    "compute_gradiend_feature_cross_encoding_matrix",
    "compute_gradiend_transition_cross_encoding_matrix",
    "can_normalize_cross_encoding_by_diagonal",
    "normalize_cross_encoding_rows_by_diagonal",
    "pair_by_id_from_trainers",
    "source_by_id_from_trainers",
    "aggregate_encoder_dataframes",
    "enter_analysis_mode",
    "enter_analysis_mode_for_trainers",
    "evaluate_encoder_for_comparison",
    "models_for_comparison",
]
