from pathlib import Path
from typing import Self

import numpy as np

from recsys_platform.models import (
    ModelManifest,
    Recommendation,
    RecommendationRequest,
    Recommender,
    TrainingMetadata,
)
from recsys_platform.types import Float32Matrix, Float32Vector, UInt32Vector


class MatrixFactorizationRecommender(Recommender):
    """Recommend items by user-item embedding similarity with a popularity fallback."""

    MODEL_FAMILY = "matrix_factorization"
    ARTIFACT_VERSION = "0.1.0"

    def __init__(
        self,
        user_ids: UInt32Vector,
        item_ids: UInt32Vector,
        user_embeddings: Float32Matrix,
        item_embeddings: Float32Matrix,
        fallback_predictions: UInt32Vector,
        fallback_scores: Float32Vector,
        *,
        training_metadata: TrainingMetadata | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id, training_metadata=training_metadata)

        self.item_ids = item_ids
        self.user_ids = user_ids

        self.item_embeddings = item_embeddings
        self.user_embeddings = user_embeddings

        self.fallback_predictions = fallback_predictions
        self.fallback_scores = fallback_scores

        if self.user_embeddings.shape[0] != self.user_ids.size:
            raise ValueError("user_embeddings must contain one row per user.")

        if self.item_embeddings.shape[0] != self.item_ids.size:
            raise ValueError("item_embeddings must contain one row per item.")

        if self.user_embeddings.shape[1] != self.item_embeddings.shape[1]:
            raise ValueError("User and item embeddings must have the same dimension.")

        if np.any(self.user_ids[1:] <= self.user_ids[:-1]):
            raise ValueError("user_ids must be strictly increasing.")

        if self.fallback_predictions.shape != self.fallback_scores.shape:
            raise ValueError("fallback_predictions and fallback_scores must have identical shapes.")

        if self.fallback_predictions.size == 0:
            raise ValueError("fallback predictions must not be empty.")

    def _lookup_users(self, user_ids: UInt32Vector) -> tuple[np.ndarray, np.ndarray]:
        """
        Map external user IDs to embedding row indices.

        Args:
            user_ids: External user IDs to look up.

        Returns:
            A pair containing the candidate row indices in ``user_embeddings``
            and a boolean mask indicating which user IDs are known to the model.
        """
        internal_indices = np.searchsorted(self.user_ids, user_ids)

        in_bounds = internal_indices < self.user_ids.size
        known = in_bounds.copy()

        known[in_bounds] = self.user_ids[internal_indices[in_bounds]] == user_ids[in_bounds]

        return internal_indices, known

    def _recommend_user(self, user_index: int, k: int) -> tuple[UInt32Vector, Float32Vector]:
        """
        Generate top-k recommendations for one known user.

        Args:
            user_index: Row index of the user in ``user_embeddings``.
            k: Number of recommendations to return.

        Returns:
            Recommended external item IDs and their corresponding dot-product
            scores, ordered from highest to lowest score.
        """
        user_embedding = self.user_embeddings[user_index]
        scores = self.item_embeddings @ user_embedding

        if k == scores.size:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices], descending=True)]

        return self.item_ids[top_indices], scores[top_indices].astype(np.float32, copy=False)

    def predict(self, request: RecommendationRequest) -> Recommendation:
        n_recommendations = min(request.k, self.item_ids.size)

        internal_ids, known = self._lookup_users(request.user_ids)
        if not known.all():
            n_recommendations = min(n_recommendations, self.fallback_predictions.size)

        item_ids = np.empty((request.user_ids.size, n_recommendations), dtype=np.uint32)
        scores = np.empty((request.user_ids.size, n_recommendations), dtype=np.float32)

        item_ids[~known] = self.fallback_predictions[:n_recommendations]
        scores[~known] = self.fallback_scores[:n_recommendations]

        for row in range(request.user_ids.size):
            if known[row]:
                row_items, row_scores = self._recommend_user(internal_ids[row], n_recommendations)
                item_ids[row] = row_items
                scores[row] = row_scores

        return Recommendation(item_ids=item_ids, scores=scores)

    def _save(self, path: Path) -> None:
        np.savez(
            path / "model.npz",
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            fallback_scores=self.fallback_scores,
            fallback_predictions=self.fallback_predictions,
        )

    @classmethod
    def _load(cls, path: Path, manifest: ModelManifest) -> Self:
        with np.load(path / "model.npz", allow_pickle=False) as state:
            model = cls(
                model_id=manifest.model_id,
                training_metadata=manifest.training_metadata,
                user_ids=state["user_ids"].astype(np.uint32, copy=False),
                item_ids=state["item_ids"].astype(np.uint32, copy=False),
                user_embeddings=state["user_embeddings"].astype(np.float32, copy=False),
                item_embeddings=state["item_embeddings"].astype(np.float32, copy=False),
                fallback_predictions=state["fallback_predictions"].astype(np.uint32, copy=False),
                fallback_scores=state["fallback_scores"].astype(np.float32, copy=False),
            )

        return model
