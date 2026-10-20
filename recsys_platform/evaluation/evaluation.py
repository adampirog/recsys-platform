from collections.abc import Collection, Iterable
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from . import metrics as metrics_module


DEFAULT_METRICS = ("precision", "recall", "hit_rate", "ndcg")
OPTIMIZED_METRICS = frozenset(("precision", "recall", "hit_rate", "ndcg"))


@lru_cache
def _discounts(k: int) -> np.ndarray:
    """Return cached logarithmic ranking discounts up to ``k``."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate(
    y_true: ArrayLike, y_pred: ArrayLike, k: int = 10, metrics: Collection[str] = DEFAULT_METRICS
) -> dict[str, float]:
    """Evaluate one ranked recommendation list at a single cutoff.

    Args:
        y_true: Relevant item IDs.
        y_pred: Ranked predicted item IDs.
        k: Ranking cutoff.
        metrics: Metric names to calculate.

    Returns:
        Metric values keyed by name.
    """
    return {
        metric: getattr(metrics_module, metric)(
            y_true=y_true,
            y_pred=y_pred,
            k=k,
        )
        for metric in metrics
    }


def evaluate_multiple(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    k_values: Iterable[int],
    metrics: Collection[str] = DEFAULT_METRICS,
) -> dict[int, dict[str, float]]:
    """Evaluate one ranked recommendation list at multiple cutoffs.

    Args:
        y_true: Relevant item IDs.
        y_pred: Ranked predicted item IDs.
        k_values: Ranking cutoffs.
        metrics: Metric names to calculate.

    Returns:
        Metric values keyed by cutoff.
    """
    k_values = tuple(sorted(set(k_values)))

    if not k_values:
        raise ValueError("k_values must not be empty.")

    if any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain only positive integers.")

    metrics = tuple(dict.fromkeys(metrics))

    if set(metrics) <= OPTIMIZED_METRICS:
        return evaluate_multiple_optimized(
            y_true=y_true, y_pred=y_pred, k_values=k_values, metrics=metrics
        )

    return {k: evaluate(y_true=y_true, y_pred=y_pred, k=k, metrics=metrics) for k in k_values}


def evaluate_multiple_optimized(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    k_values: tuple[int, ...],
    metrics: tuple[str, ...] = tuple(OPTIMIZED_METRICS),
) -> dict[int, dict[str, float]]:
    """Evaluate built-in ranking metrics at multiple cutoffs in one pass.

    Only metrics in ``OPTIMIZED_METRICS`` are supported.
    """
    max_k = max(k_values)

    relevant, predicted = metrics_module._prepare_inputs(y_true, y_pred, max_k)

    hits = np.isin(predicted, relevant)
    cumulative_hits = np.cumsum(hits)

    if "ndcg" in metrics:
        discounts = _discounts(max_k)
        cumulative_dcg = np.cumsum(hits * discounts[: predicted.size])
        cumulative_idcg = np.cumsum(discounts)

    result: dict[int, dict[str, float]] = {}

    for k in k_values:
        effective_k = min(k, predicted.size)
        n_hits = int(cumulative_hits[effective_k - 1]) if effective_k else 0

        values: dict[str, float] = {}

        if "precision" in metrics:
            values["precision"] = n_hits / k

        if "recall" in metrics:
            values["recall"] = n_hits / relevant.size if relevant.size else 0.0

        if "hit_rate" in metrics:
            values["hit_rate"] = float(n_hits > 0)

        if "ndcg" in metrics:
            ideal_length = min(relevant.size, k)

            if effective_k and ideal_length:
                dcg = cumulative_dcg[effective_k - 1]  # type: ignore
                idcg = cumulative_idcg[ideal_length - 1]  # type: ignore

                values["ndcg"] = float(dcg / idcg)
            else:
                values["ndcg"] = 0.0

        result[k] = values

    return result


def evaluate_batch_sums(
    y_true: list[np.ndarray], y_pred: np.ndarray, k_values: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    """Return summed ranking metrics for a batch of users."""

    n_users, n_predictions = y_pred.shape
    max_k = max(k_values)

    if len(y_true) != n_users:
        raise ValueError("y_true and y_pred must contain the same number of users.")

    # Relevance is binary.
    relevant = [np.unique(items) for items in y_true]
    relevant_counts = np.fromiter((items.size for items in relevant), dtype=np.int64, count=n_users)

    if n_predictions > 1:
        sorted_predictions = np.sort(y_pred, axis=1)
        if np.any(np.diff(sorted_predictions, axis=1) == 0):
            raise ValueError("Predictions contain duplicate item IDs.")

    n_relevant = int(relevant_counts.sum())

    if n_relevant:
        relevant_items = np.concatenate(relevant).astype(np.uint64, copy=False)
        relevant_users = np.repeat(np.arange(n_users, dtype=np.uint64), relevant_counts)
        relevant_keys = (relevant_users << 32) | relevant_items
    else:
        relevant_keys = np.empty(0, dtype=np.uint64)

    prediction_users = np.arange(n_users, dtype=np.uint64)[:, None]
    prediction_keys = (prediction_users << 32) | y_pred.astype(np.uint64, copy=False)

    hits = np.isin(prediction_keys, relevant_keys)

    cumulative_hits = np.cumsum(hits, axis=1, dtype=np.int64)

    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    cumulative_dcg = np.cumsum(hits * discounts[:n_predictions], axis=1)
    cumulative_idcg = np.cumsum(discounts)

    result: dict[int, dict[str, float]] = {}

    for k in k_values:
        effective_k = min(k, n_predictions)

        if effective_k:
            hits_at_k = cumulative_hits[:, effective_k - 1]
            dcg_at_k = cumulative_dcg[:, effective_k - 1]
        else:
            hits_at_k = np.zeros(n_users, dtype=np.int64)
            dcg_at_k = np.zeros(n_users, dtype=np.float64)

        recall = np.divide(
            hits_at_k,
            relevant_counts,
            out=np.zeros(n_users, dtype=np.float64),
            where=relevant_counts > 0,
        )

        ideal_length = np.minimum(relevant_counts, k)
        idcg = np.zeros(n_users, dtype=np.float64)

        has_relevant = ideal_length > 0
        idcg[has_relevant] = cumulative_idcg[ideal_length[has_relevant] - 1]

        ndcg = np.divide(dcg_at_k, idcg, out=np.zeros(n_users, dtype=np.float64), where=idcg > 0)

        result[k] = {
            "precision": float(hits_at_k.sum() / k),
            "recall": float(recall.sum()),
            "hit_rate": float(np.count_nonzero(hits_at_k)),
            "ndcg": float(ndcg.sum()),
        }

    return result
