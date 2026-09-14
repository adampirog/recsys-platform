from pathlib import Path
from typing import Self

import numpy as np
import polars as pl

from recsys_platform.models.base import Recommendation, RecommendationRequest, Recommender

from .dataset import PopularityDataset


class PopularityRecommender(Recommender[PopularityDataset]):
    """
    Recommend globally popular items.

    Item popularity is defined as the number of interactions with an item
    in the training dataset. Scores are normalized by the popularity of the
    most popular item, producing values in the range [0, 1].

    The model is non-personalized: every user receives the same ranked list.
    """

    def __init__(self) -> None:
        self.item_ids = np.empty(0, dtype=np.uint32)
        self.scores = np.empty(0, dtype=np.float32)

    def fit(self, data: PopularityDataset) -> Self:
        popularity = (
            data.data.group_by("item_id")
            .len()
            .sort(
                ["len", "item_id"],
                descending=[True, False],
            )
            .with_columns(
                (pl.col("len") / pl.col("len").max()).alias("score"),
            )
            .select("item_id", "score")
            .collect(engine="streaming")
        )

        self.item_ids = popularity["item_id"].to_numpy().astype(np.uint32, copy=False)
        self.scores = popularity["score"].to_numpy().astype(np.float32, copy=False)

        return self

    def predict(self, request: RecommendationRequest) -> Recommendation:
        n_recommendations = min(request.k, self.item_ids.size)

        item_ids = self.item_ids[:n_recommendations]
        scores = self.scores[:n_recommendations]

        batch_size = request.user_ids.size

        return Recommendation(
            user_ids=request.user_ids,
            item_ids=np.broadcast_to(item_ids, (batch_size, n_recommendations)),
            scores=np.broadcast_to(scores, (batch_size, n_recommendations)),
        )

    def save(self, path: str | Path) -> None:
        """Serialize the fitted popularity model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez(path / "model.npz", item_ids=self.item_ids, scores=self.scores)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a serialized popularity model from disk."""
        model_path = Path(path) / "model.npz"

        with np.load(model_path, allow_pickle=False) as state:
            model = cls()
            model.item_ids = state["item_ids"].astype(np.uint32, copy=False)
            model.scores = state["scores"].astype(np.float32, copy=False)

        return model
