from collections import UserDict, defaultdict
from collections.abc import Iterable

import polars as pl
from prettytable import PrettyTable

from recsys_platform.evaluation.metrics import hit_rate, ndcg, precision, recall
from recsys_platform.models.base import RecommendationDataset, RecommendationRequest, Recommender


class EvaluationResult(UserDict):
    """
    Recommendation metrics evaluated at multiple ranking cutoffs.

    Maps each cutoff ``k`` to a dictionary of metric names and their
    corresponding values.

    Example:
        ``result[10]["recall"]`` returns Recall@10.
    """

    def __str__(self) -> str:
        metric_names = sorted(next(iter(self.values())).keys())
        table = PrettyTable()

        table.field_names = ["K"] + [item.replace("_", " ").title() for item in metric_names]
        table.align["K"] = "r"

        for k, metrics in self.items():
            table.add_row([k] + [f"{metrics[name]:.4f}" for name in metric_names])

        return table.get_string()


def build_targets(data: pl.LazyFrame) -> pl.DataFrame:
    """
    Build user-level ground-truth item sets.

    Each row contains a user ID and the unique items that the user interacted
    with in the evaluated dataset split.

    Args:
        data: Interaction data used as evaluation ground truth.

    Returns:
        DataFrame containing ``user_id`` and ``relevant_items`` columns.
    """
    return (
        data.group_by("user_id")
        .agg(pl.col("item_id").unique().alias("relevant_items"))
        .collect(engine="streaming")
    )


METRICS = (precision, recall, hit_rate, ndcg)


def evaluate(
    model: Recommender,
    dataset: RecommendationDataset,
    *,
    k_values: Iterable[int] = (5, 10, 20),
    split: str = "valid",
    batch_size: int = 10_000,
) -> EvaluationResult:
    """
    Evaluate a recommender against future user interactions.

    Predictions are generated once per user using the largest requested
    cutoff and reused to calculate all metrics at smaller cutoffs. Metrics
    are macro-averaged across users.

    Args:
        model: Fitted recommender to evaluate.
        dataset: Dataset containing the requested evaluation split.
        k_values: Ranking cutoffs at which metrics are calculated.
        split: Dataset split used as ground truth.
        batch_size: Number of users passed to the model per prediction batch.

    Returns:
        Mapping from each cutoff to its averaged recommendation metrics.
    """
    k_values = tuple(sorted(set(k_values)))
    targets = build_targets(getattr(dataset, split))

    result = {k: defaultdict(float) for k in k_values}
    for batch in targets.iter_slices(n_rows=batch_size):
        requests = [
            RecommendationRequest(user_id=user_id, k=max(k_values)) for user_id in batch["user_id"]
        ]

        predictions = model.predict(requests)
        for user_id, y_true in batch.iter_rows():
            y_pred = [recommendation.item_id for recommendation in predictions[user_id]]

            for k in k_values:
                total = result[k]
                for metric in METRICS:
                    total[metric.__name__] += metric(y_pred=y_pred, y_true=y_true, k=k)

    # Normalize
    n_users = targets.height
    result = {
        k: {name: value / n_users for name, value in metrics.items()}
        for k, metrics in result.items()
    }

    return EvaluationResult(result)
