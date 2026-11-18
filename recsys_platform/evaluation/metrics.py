"""Ranking metrics for one user.

Inputs are one-dimensional item ID sequences, with predictions ordered
from highest to lowest rank. Relevance is binary; repeated relevant IDs
are deduplicated and duplicate predictions within k are rejected.
"""

import numpy as np
from numpy.typing import ArrayLike


ID_DTYPE = np.uint32


def _prepare_inputs(y_true: ArrayLike, y_pred: ArrayLike, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Prepare one user's binary relevance and ranked predictions.

    Inputs must be one-dimensional uint32-compatible item IDs. Relevant IDs
    are deduplicated; duplicate predictions within the positive cutoff are rejected.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    y_pred_array = np.asarray(y_pred[:k], dtype=ID_DTYPE)  # type: ignore[index]
    y_true_array = np.asarray(y_true, dtype=ID_DTYPE)

    if y_pred_array.ndim != 1:
        raise ValueError("y_pred must be one-dimensional.")

    if y_true_array.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    # Relevance is binary.
    y_true_array = np.unique(y_true_array)

    if np.unique(y_pred_array).size != y_pred_array.size:
        raise ValueError("Predictions contain duplicate item IDs.")

    return y_true_array, y_pred_array


def hit_rate(y_true: ArrayLike, y_pred: ArrayLike, k: int = 10) -> float:
    y_true_array, y_pred_array = _prepare_inputs(y_true, y_pred, k)

    if y_true_array.size == 0:
        return 0.0

    hits = np.isin(y_pred_array, y_true_array)

    return float(np.any(hits))


def precision(y_true: ArrayLike, y_pred: ArrayLike, k: int = 10) -> float:
    """Compute precision using k as the denominator, even for shorter rankings."""

    y_true_array, y_pred_array = _prepare_inputs(y_true, y_pred, k)

    if y_pred_array.size == 0:
        return 0.0

    hits = np.isin(y_pred_array, y_true_array)

    return float(np.count_nonzero(hits) / k)


def recall(y_true: ArrayLike, y_pred: ArrayLike, k: int = 10) -> float:
    y_true_array, y_pred_array = _prepare_inputs(y_true, y_pred, k)

    if y_true_array.size == 0:
        return 0.0

    hits = np.isin(y_pred_array, y_true_array)

    return float(np.count_nonzero(hits) / y_true_array.size)


def ndcg(y_true: ArrayLike, y_pred: ArrayLike, k: int = 10) -> float:
    """Compute NDCG with binary relevance and an ideal ranking of up to k relevant items."""
    y_true_array, y_pred_array = _prepare_inputs(y_true, y_pred, k)

    if y_true_array.size == 0:
        return 0.0

    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    relevance = np.isin(y_pred_array, y_true_array).astype(np.float64)
    dcg = np.dot(relevance, discounts[: y_pred_array.size])

    ideal_length = min(y_true_array.size, k)
    idcg = discounts[:ideal_length].sum()

    return float(dcg / idcg)
