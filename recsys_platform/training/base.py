from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable

from recsys_platform.data import RecommenderDataset
from recsys_platform.evaluation import EvaluationResult, evaluate_multiple
from recsys_platform.models import RecommendationRequest, Recommender


class Trainer[DatasetType: RecommenderDataset, RecommenderType: Recommender](ABC):
    """Train and evaluate a model-specific recommender."""

    @abstractmethod
    def fit(self, data: DatasetType) -> RecommenderType:
        """Fit a recommender from the training dataset."""

    def evaluate(
        self,
        model: RecommenderType,
        data: DatasetType,
        *,
        k_values: Iterable[int] = (5, 10, 20),
        batch_size: int = 10_000,
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
        for targets in data.iter_targets(batch_size):
            recommendations = model.predict(
                RecommendationRequest(user_ids=targets.user_ids, k=max_k)
            )

            for y_true, y_pred in zip(
                targets.relevant_items, recommendations.item_ids, strict=True
            ):
                result = evaluate_multiple(y_true=y_true, y_pred=y_pred, k_values=k_values)

                for k, metrics in result.items():
                    for name, value in metrics.items():
                        totals[k][name] += value

            n_users += targets.user_ids.size

        if n_users == 0:
            raise ValueError("Evaluation dataset contains no users.")

        return EvaluationResult(
            {
                k: {name: value / n_users for name, value in metrics.items()}
                for k, metrics in totals.items()
            }
        )
