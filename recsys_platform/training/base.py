from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm

from recsys_platform.data import RecommenderDataset
from recsys_platform.evaluation import EvaluationResult
from recsys_platform.evaluation.evaluation import _evaluate_batch_sums
from recsys_platform.models import RecommendationRequest, Recommender, TrainingMetadata


class TrainerConfig(BaseModel):
    """Configuration used to train a recommender."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate_json(handle.read())

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=2))


class Trainer[
    DatasetType: RecommenderDataset,
    RecommenderType: Recommender,
    ConfigType: TrainerConfig,
](ABC):
    """Train and evaluate a model-specific recommender."""

    def __init__(self, config: ConfigType) -> None:
        self.config = config

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
        with tqdm(desc="Evaluating", unit="users", disable=not verbose, total=total) as progress:
            for targets in data.iter_targets(batch_size):
                recommendations = model.predict(
                    RecommendationRequest(user_ids=targets.user_ids, k=max_k)
                )

                batch_totals = _evaluate_batch_sums(
                    y_true=targets.relevant_items,
                    y_pred=recommendations.item_ids,
                    k_values=k_values,
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

    def get_metadata(self) -> TrainingMetadata:
        return TrainingMetadata(
            trainer=f"{type(self).__module__}.{type(self).__qualname__}",
            config=self.config.model_dump(mode="json"),
        )
