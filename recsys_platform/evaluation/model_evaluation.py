from collections import defaultdict
from collections.abc import Iterable

from tqdm.auto import tqdm

from recsys_platform.data import RecommenderDataset
from recsys_platform.models import RecommendationRequest, Recommender

from .evaluation import evaluate_batch_sums
from .result import EvaluationResult


def evaluate_model(
    model: Recommender,
    data: RecommenderDataset,
    *,
    k_values: Iterable[int] = (5, 10, 20),
    batch_size: int = 10_000,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate a recommender using macro-averaged per-user ranking metrics.

    Args:
        model: Recommender to evaluate.
        data: Dataset providing evaluation targets.
        k_values: Ranking cutoffs.
        batch_size: Users per prediction batch.

    Returns:
        Metrics averaged across evaluated users.
    """
    k_values = tuple(sorted(set(k_values)))

    if not k_values:
        raise ValueError("k_values must not be empty.")

    if any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain only positive integers.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    max_k = max(k_values)
    totals = {k: defaultdict(float) for k in k_values}

    n_users = 0
    total = data.n_targets if verbose else None
    with tqdm(desc="Evaluating", unit=" users", disable=not verbose, total=total) as progress:
        for targets in data.iter_targets(batch_size):
            recommendations = model.predict(
                RecommendationRequest(user_ids=targets.user_ids, k=max_k)
            )

            batch_totals = evaluate_batch_sums(
                y_true=targets.relevant_items, y_pred=recommendations.item_ids, k_values=k_values
            )

            for k, metrics in batch_totals.items():
                for name, value in metrics.items():
                    totals[k][name] += value

            batch_users = targets.user_ids.size
            n_users += batch_users
            progress.update(batch_users)

    if n_users == 0:
        raise ValueError("Evaluation dataset contains no users.")

    return EvaluationResult(
        {
            k: {name: value / n_users for name, value in metrics.items()}
            for k, metrics in totals.items()
        }
    )
