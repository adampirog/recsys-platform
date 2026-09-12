from collections.abc import Collection, Iterable, Sequence
from functools import lru_cache

import numpy as np

from recsys_platform.evaluation import metrics as metrics_module


DEFAULT_METRICS = ("precision", "recall", "hit_rate", "ndcg")
OPTIMIZED_METRICS = frozenset(("precision", "recall", "hit_rate", "ndcg"))


@lru_cache
def _discounts(k: int) -> np.ndarray:
    """Return cached logarithmic ranking discounts up to ``k``."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate(
    y_true: Collection[int],
    y_pred: Sequence[int],
    k: int = 10,
    metrics: Collection[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """
    Evaluate one ranked recommendation list at a single cutoff.

    Metric functions are resolved dynamically from the metrics module and
    must accept the common ``(y_true, y_pred, k)`` interface.

    Args:
        y_true: Relevant ground-truth item IDs.
        y_pred: Predicted item IDs ordered from most to least relevant.
        k: Ranking cutoff.
        metrics: Names of metrics to calculate.

    Returns:
        Mapping from metric name to its value at ``k``.
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
    y_true: Collection[int],
    y_pred: Sequence[int],
    k_values: Iterable[int],
    metrics: Collection[str] = DEFAULT_METRICS,
) -> dict[int, dict[str, float]]:
    """
    Evaluate one ranked recommendation list at multiple cutoffs.

    Uses an optimized implementation when all requested metrics belong to
    the supported optimized metric set. Otherwise, falls back to evaluating
    each cutoff independently.

    Args:
        y_true: Relevant ground-truth item IDs.
        y_pred: Predicted item IDs ordered from most to least relevant.
        k_values: Ranking cutoffs to evaluate.
        metrics: Names of metrics to calculate.

    Returns:
        Mapping from each cutoff to its metric values.
    """
    k_values = tuple(sorted(set(k_values)))
    metrics = tuple(dict.fromkeys(metrics))

    if set(metrics) <= OPTIMIZED_METRICS:
        return evaluate_multiple_optimized(
            y_true=y_true, y_pred=y_pred, k_values=k_values, metrics=metrics
        )

    return {k: evaluate(y_true=y_true, y_pred=y_pred, k=k, metrics=metrics) for k in k_values}


def evaluate_multiple_optimized(
    y_true: Collection[int],
    y_pred: Sequence[int],
    k_values: tuple[int, ...],
    metrics: tuple[str, ...],
) -> dict[int, dict[str, float]]:
    """
    Efficiently evaluate compatible ranking metrics at multiple cutoffs.

    Optimized for: "precision", "recall", "hit_rate", "ndcg"

    The relevance mask, cumulative hit counts, and discounted gains are
    computed once and reused for every requested cutoff.

    Args:
        y_true: Relevant ground-truth item IDs.
        y_pred: Predicted item IDs ordered from most to least relevant.
        k_values: Ranking cutoffs to evaluate.
        metrics: Optimized metric names to calculate.

    Returns:
        Mapping from each cutoff to its metric values.
    """
    max_k = max(k_values)

    relevant, predicted = metrics_module._prepare_inputs(y_true, y_pred, max_k)

    if len(set(predicted.tolist())) != predicted.size:
        raise ValueError("y_pred must not contain duplicate item IDs.")

    hits = np.isin(predicted, relevant)
    cumulative_hits = np.cumsum(hits)

    if "ndcg" in metrics:
        discounts = _discounts(max_k)
        cumulative_dcg = np.cumsum(hits * discounts[: predicted.size])
        cumulative_idcg = np.cumsum(discounts)
    else:
        cumulative_dcg = None
        cumulative_idcg = None

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
                assert cumulative_dcg is not None
                assert cumulative_idcg is not None

                dcg = cumulative_dcg[effective_k - 1]
                idcg = cumulative_idcg[ideal_length - 1]

                values["ndcg"] = float(dcg / idcg)
            else:
                values["ndcg"] = 0.0

        result[k] = values

    return result
