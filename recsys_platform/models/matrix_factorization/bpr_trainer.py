from typing import Annotated

import numpy as np
from implicit.bpr import BayesianPersonalizedRanking
from pydantic import Field

from recsys_platform.data import RecommenderDataset

from .model import MatrixFactorizationRecommender
from .trainer import MatrixFactorizationTrainer, MatrixFactorizationTrainerConfig


class BPRTrainerConfig(MatrixFactorizationTrainerConfig):
    embedding_size: Annotated[int, Field(gt=0)] = 64
    learning_rate: Annotated[float, Field(gt=0)] = 0.01
    regularization: Annotated[float, Field(gt=0)] = 0.01
    iterations: Annotated[int, Field(gt=0)] = 100
    verify_negative_samples: bool = True


class BPRTrainer(MatrixFactorizationTrainer[BPRTrainerConfig]):
    """Train a matrix-factorization recommender using implicit-feedback BPR."""

    def __init__(self, config: BPRTrainerConfig | None = None) -> None:
        super().__init__(config=config or BPRTrainerConfig())

    def fit(self, data: RecommenderDataset) -> MatrixFactorizationRecommender:
        user_items, user_ids, item_ids = self._build_user_item_matrix(data)

        bpr = BayesianPersonalizedRanking(
            factors=self.config.embedding_size,
            learning_rate=self.config.learning_rate,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            verify_negative_samples=self.config.verify_negative_samples,
            num_threads=self.config.num_threads,
            random_state=self.config.random_state,
            dtype=np.float32,
        )

        bpr.fit(user_items)

        fallback_predictions, fallback_scores = self._get_fallback_predictions(
            user_items=user_items,
            item_ids=item_ids,
            max_items=self.config.fallback_items,
        )

        return MatrixFactorizationRecommender(
            user_ids=user_ids,
            item_ids=item_ids,
            user_embeddings=bpr.user_factors.astype(np.float32, copy=False),
            item_embeddings=bpr.item_factors.astype(np.float32, copy=False),
            fallback_predictions=fallback_predictions,
            fallback_scores=fallback_scores,
            training_metadata=self.get_metadata(),
        )
