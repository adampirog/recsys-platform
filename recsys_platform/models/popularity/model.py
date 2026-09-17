from pathlib import Path
from typing import Self

import numpy as np

from recsys_platform.models import ModelManifest, Recommendation, RecommendationRequest, Recommender
from recsys_platform.types import Float32Vector, UInt32Vector


class PopularityRecommender(Recommender):
    """
    Recommend globally popular items.

    Item popularity is defined as the number of interactions with an item
    in the training dataset. Scores are normalized by the popularity of the
    most popular item, producing values in the range [0, 1].

    The model is non-personalized: every user receives the same ranked list.
    """

    MODEL_TYPE = "popularity"
    ARTIFACT_VERSION = "0.1.0"

    def __init__(
        self, item_ids: UInt32Vector, scores: Float32Vector, *, model_id: str | None = None
    ) -> None:
        super().__init__(model_id=model_id)

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

    def _save(self, path: Path) -> None:
        np.savez(path / "model.npz", item_ids=self.item_ids, scores=self.scores)

    @classmethod
    def _load(cls, path: Path, manifest: ModelManifest) -> Self:

        with np.load(path / "model.npz", allow_pickle=False) as state:
            model = cls(
                model_id=manifest.model_id,
                item_ids=state["item_ids"].astype(np.uint32, copy=False),
                scores=state["scores"].astype(np.float32, copy=False),
            )

        return model
