from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from recsys_platform.data import RecommenderDataset
from recsys_platform.evaluation import EvaluationResult, evaluate_multiple


@dataclass(frozen=True, slots=True)
class Recommendation:
    user_ids: NDArray[np.uint32]
    item_ids: NDArray[np.uint32]  # shape: (n_users, n_recommendations)
    scores: NDArray[np.float32]  # shape: (n_users, n_recommendations)


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    user_ids: NDArray[np.uint32]
    k: int = 20


class Recommender[DatasetType: RecommenderDataset](ABC):
    @abstractmethod
    def fit(self, data: DatasetType) -> Self: ...

    @abstractmethod
    def predict(self, request: RecommendationRequest) -> Recommendation: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self: ...

    def evaluate(
        self,
        data: DatasetType,
        *,
        k_values: Iterable[int] = (5, 10, 20),
        batch_size: int = 10_000,
    ) -> EvaluationResult:
        k_values = tuple(sorted(set(k_values)))
        max_k = max(k_values)
        totals = {k: defaultdict(float) for k in k_values}

        n_users = 0
        for targets in data.iter_targets(batch_size):
            recommendations = self.predict(
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

        return EvaluationResult(
            {
                k: {name: value / n_users for name, value in metrics.items()}
                for k, metrics in totals.items()
            }
        )
