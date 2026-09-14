from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field

from recsys_platform.data import RecommenderDataset
from recsys_platform.evaluation import EvaluationResult, evaluate_multiple
from recsys_platform.types import Float32Matrix, UInt32Matrix, UInt32Vector


class RecommendationRequest(BaseModel):
    """Batch request for top-k recommendations.

    Attributes:
        user_ids: IDs of users to generate recommendations for.
        k: Maximum number of recommendations requested per user.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    user_ids: UInt32Vector
    k: Annotated[int, Field(gt=0, strict=True)] = 20


class Recommendation(BaseModel):
    """Batched ranked recommendations.

    Rows in ``item_ids`` and ``scores`` correspond positionally to
    ``user_ids``.

    Attributes:
        user_ids: Users represented by each result row.
        item_ids: Ranked item IDs with shape
            ``(n_users, n_recommendations)``.
        scores: Recommendation scores with the same shape as ``item_ids``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    item_ids: UInt32Matrix
    scores: Float32Matrix


class Recommender[DatasetType: RecommenderDataset](ABC):
    @abstractmethod
    def fit(self, data: DatasetType) -> Self: ...

    @abstractmethod
    def predict(self, request: RecommendationRequest) -> Recommendation:
        """
        Generate recommendations for a batch of users.

        Returned rows must preserve the order of ``request.user_ids``.
        """

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
