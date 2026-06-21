"""Comparison utilities for inter-model GRADIEND analysis."""

from gradiend.comparison.anchor_aligned import (
    aggregate_anchor_aligned_encoding_rows,
    build_anchor_aligned_encoding_rows,
    compute_anchor_aligned_encoding_matrix,
    pair_by_id_from_trainers,
)
from gradiend.comparison.cross_encoding import (
    compute_cross_encoding_matrix,
    normalize_cross_encoding_rows_by_diagonal,
)
from gradiend.comparison.feature_cross_encoding import (
    compute_gradiend_feature_cross_encoding_matrix,
)
from gradiend.comparison.similarity import (
    compute_grouped_similarity_matrices,
    compute_similarity_matrix,
)

__all__ = [
    "aggregate_anchor_aligned_encoding_rows",
    "build_anchor_aligned_encoding_rows",
    "compute_anchor_aligned_encoding_matrix",
    "compute_similarity_matrix",
    "compute_grouped_similarity_matrices",
    "compute_cross_encoding_matrix",
    "compute_gradiend_feature_cross_encoding_matrix",
    "normalize_cross_encoding_rows_by_diagonal",
    "pair_by_id_from_trainers",
]
