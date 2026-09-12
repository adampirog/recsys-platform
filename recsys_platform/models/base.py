import json
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import polars as pl
from prettytable import PrettyTable

from recsys_platform.evaluation.evaluation import evaluate_multiple


class RecommendationDataset:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @property
    def data(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path)


@dataclass(frozen=True, slots=True)
class Recommendation:
    item_id: int
    score: float


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    user_id: int
    k: int = 20


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

    def save(self, path: str | Path, *, save_format: Literal["str", "json"] = "str") -> None:
        if save_format == "str":
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(str(self))
        elif save_format == "json":
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2)
        else:
            raise ValueError(f"Format '{save_format}' not supported")


class Recommender(ABC):
    @abstractmethod
    def fit(self, data: RecommendationDataset) -> Self: ...

    @abstractmethod
    def predict(self, requests: list[RecommendationRequest]) -> dict[int, list[Recommendation]]: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self: ...

    @staticmethod
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

    def evaluate(
        self,
        data: RecommendationDataset,
        *,
        k_values: Iterable[int] = (5, 10, 20),
        batch_size: int = 10_000,
    ) -> EvaluationResult:
        """Evaluate the recommender against future user interactions.

        Predictions are generated once per user at the largest requested
        cutoff. Ranking metrics are macro-averaged across evaluation users.

        Args:
            dataset: Dataset containing the evaluation interactions.
            k_values: Ranking cutoffs at which metrics are calculated.
            split: Dataset split used as ground truth.
            batch_size: Number of users predicted in one batch.

        Returns:
            Recommendation metrics evaluated at each requested cutoff.
        """
        k_values = tuple(sorted(set(k_values)))
        max_k = max(k_values)

        targets = self.build_targets(data.data)
        totals = {k: defaultdict(float) for k in k_values}

        for batch in targets.iter_slices(n_rows=batch_size):
            requests = [
                RecommendationRequest(user_id=user_id, k=max_k) for user_id in batch["user_id"]
            ]

            predictions = self.predict(requests)

            for user_id, y_true in batch.iter_rows():
                y_pred = [recommendation.item_id for recommendation in predictions[user_id]]
                user_result = evaluate_multiple(y_true=y_true, y_pred=y_pred, k_values=k_values)

                for k, metrics in user_result.items():
                    for name, value in metrics.items():
                        totals[k][name] += value

        n_users = targets.height
        return EvaluationResult(
            {
                k: {name: value / n_users for name, value in metrics.items()}
                for k, metrics in totals.items()
            }
        )
