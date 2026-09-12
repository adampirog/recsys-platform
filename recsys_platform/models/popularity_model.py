from pathlib import Path
from typing import Self

import polars as pl

from recsys_platform.models.base import (
    Recommendation,
    RecommendationDataset,
    RecommendationRequest,
    Recommender,
)


class PopularityRecommender(Recommender):
    """
    Recommend globally popular items.

    Item popularity is defined as the number of interactions with an item
    in the training dataset. Scores are normalized by the popularity of the
    most popular item, producing values in the range [0, 1].

    The model is non-personalized: every user receives the same ranked list.
    """

    def __init__(self) -> None:
        self.popular_items: list[Recommendation] = []

    def fit(self, data: RecommendationDataset) -> Self:
        """Fit the recommender using interaction counts from the training set."""

        popularity = (
            data.data.group_by("item_id")
            .len()
            .sort("len", descending=True)
            .with_columns(
                (pl.col("len") / pl.col("len").max()).alias("score"),
            )
            .select("item_id", "score")
            .collect(engine="streaming")
        )

        self.popular_items = [
            Recommendation(
                item_id=item_id,
                score=score,
            )
            for item_id, score in popularity.iter_rows()
        ]

        return self

    def predict(self, requests: list[RecommendationRequest]) -> dict[int, list[Recommendation]]:
        """Generate popularity-based recommendations."""

        return {request.user_id: self.popular_items[: request.k] for request in requests}

    def save(self, path: str | Path) -> None:
        """Save the fitted model."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        pl.DataFrame(
            {
                "item_id": [recommendation.item_id for recommendation in self.popular_items],
                "score": [recommendation.score for recommendation in self.popular_items],
            }
        ).write_parquet(path / "popular_items.parquet")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously saved model"""

        items = pl.read_parquet(Path(path) / "popular_items.parquet")

        model = cls()
        model.popular_items = [
            Recommendation(
                item_id=item_id,
                score=score,
            )
            for item_id, score in items.iter_rows()
        ]

        return model
