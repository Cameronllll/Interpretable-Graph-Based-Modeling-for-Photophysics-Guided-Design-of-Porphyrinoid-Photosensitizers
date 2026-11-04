"""Shared ECFP prevalence filtering, without loading D-MPNN training dependencies."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ECFPPrevalenceFilter(BaseEstimator, TransformerMixin):
    """Remove ECFP columns whose training-fold prevalence is in given ranges.

    D-MPNN embeddings and optional descriptors pass through unchanged. Because
    this transformer is placed inside GridSearchCV's pipeline, each fold learns
    its mask from that fold's training subset only.
    """

    def __init__(
        self,
        graph_dimension: int,
        ecfp_dimension: int,
        remove_ranges: Sequence[Tuple[float, float]] = (),
    ) -> None:
        self.graph_dimension = graph_dimension
        self.ecfp_dimension = ecfp_dimension
        self.remove_ranges = remove_ranges

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> "ECFPPrevalenceFilter":
        del labels
        matrix = np.asarray(features)
        if matrix.ndim != 2:
            raise ValueError(
                f"ECFPPrevalenceFilter expects a 2-D matrix; got {matrix.shape}."
            )
        graph_dimension = int(self.graph_dimension)
        ecfp_dimension = int(self.ecfp_dimension)
        if graph_dimension < 0 or ecfp_dimension < 1:
            raise ValueError(
                "graph_dimension must be non-negative and ecfp_dimension must "
                "be positive."
            )
        ecfp_end = graph_dimension + ecfp_dimension
        if ecfp_end > matrix.shape[1]:
            raise ValueError(
                "D-MPNN and ECFP dimensions exceed the input dimension: "
                f"{graph_dimension} + {ecfp_dimension} > {matrix.shape[1]}."
            )

        validated_ranges: List[Tuple[float, float]] = []
        for prevalence_range in self.remove_ranges:
            if len(prevalence_range) != 2:
                raise ValueError(
                    "Each removal range must contain exactly two endpoints."
                )
            lower, upper = (float(value) for value in prevalence_range)
            if not 0.0 <= lower <= upper <= 1.0:
                raise ValueError(
                    "Removal ranges must satisfy 0 <= lower <= upper <= 1."
                )
            validated_ranges.append((lower, upper))

        ecfp = matrix[:, graph_dimension:ecfp_end]
        if not np.all((ecfp == 0.0) | (ecfp == 1.0)):
            raise ValueError(
                "The configured ECFP block contains values other than 0 and 1."
            )
        prevalence = np.mean(ecfp, axis=0, dtype=np.float64)
        removed_mask = np.zeros(ecfp_dimension, dtype=bool)
        for lower, upper in validated_ranges:
            removed_mask |= (prevalence >= lower) & (prevalence <= upper)
        selected_bit_indices = np.flatnonzero(~removed_mask).astype(np.int64)
        if selected_bit_indices.size == 0:
            raise ValueError(
                "The configured prevalence ranges remove every ECFP bit. "
                "Narrow the ranges near the top of this file."
            )

        trailing_indices = np.arange(ecfp_end, matrix.shape[1], dtype=np.int64)
        self.n_features_in_ = int(matrix.shape[1])
        self.ecfp_prevalence_ = prevalence
        self.removed_ecfp_bit_indices_ = np.flatnonzero(removed_mask).astype(
            np.int64
        )
        self.selected_ecfp_bit_indices_ = selected_bit_indices
        self.selected_feature_indices_ = np.concatenate(
            [
                np.arange(graph_dimension, dtype=np.int64),
                graph_dimension + selected_bit_indices,
                trailing_indices,
            ]
        )
        self.n_features_out_ = int(self.selected_feature_indices_.size)
        self.validated_remove_ranges_ = tuple(validated_ranges)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "selected_feature_indices_"):
            raise RuntimeError("ECFPPrevalenceFilter must be fitted first.")
        matrix = np.asarray(features)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                "Input shape does not match the fitted prevalence filter: "
                f"got {matrix.shape}, expected (*, {self.n_features_in_})."
            )
        return matrix[:, self.selected_feature_indices_]
