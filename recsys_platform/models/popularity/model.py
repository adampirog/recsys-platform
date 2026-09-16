from pathlib import Path
from typing import Self

import numpy as np

from recsys_platform.models.base import Recommendation, RecommendationRequest, Recommender
from recsys_platform.types import Float32Vector, UInt32Vector


class PopularityRecommender(Recommender):
    """
    Recommend globally popular items.

    Item popularity is defined as the number of interactions with an item
    in the training dataset. Scores are normalized by the popularity of the
    most popular item, producing values in the range [0, 1].

    The model is non-personalized: every user receives the same ranked list.
    """

    def __init__(self, item_ids: UInt32Vector, scores: Float32Vector) -> None:
        self.item_ids = item_ids
        self.scores = scores

        if item_ids.shape != scores.shape:
            raise ValueError("item_ids and scores must have identical shapes.")

    def predict(self, request: RecommendationRequest) -> Recommendation:
        """Return the top globally popular items for each requested user."""

        n_recommendations = min(request.k, self.item_ids.size)

        item_ids = self.item_ids[:n_recommendations]
        scores = self.scores[:n_recommendations]

        batch_size = request.user_ids.size

        return Recommendation(
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
            model = cls(
                item_ids=state["item_ids"].astype(np.uint32, copy=False),
                scores=state["scores"].astype(np.float32, copy=False),
            )

        return model
