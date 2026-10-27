from typing import Annotated

import numpy as np
from implicit.als import AlternatingLeastSquares
from pydantic import Field

from recsys_platform.data import RecommenderDataset

from .model import MatrixFactorizationRecommender
from .trainer import MatrixFactorizationTrainer, MatrixFactorizationTrainerConfig


class ALSTrainerConfig(MatrixFactorizationTrainerConfig):
    embedding_size: Annotated[int, Field(gt=0)] = 64
    regularization: Annotated[float, Field(gt=0)] = 0.01
    alpha: Annotated[float, Field(gt=0)] = 20.0
    iterations: Annotated[int, Field(gt=0)] = 20


class ALSTrainer(MatrixFactorizationTrainer[ALSTrainerConfig]):
    def __init__(self, config: ALSTrainerConfig | None = None) -> None:
        super().__init__(config=config or ALSTrainerConfig())

    def fit(self, data: RecommenderDataset) -> MatrixFactorizationRecommender:
        user_items, user_ids, item_ids = self._build_user_item_matrix(data)

        als = AlternatingLeastSquares(
            factors=self.config.embedding_size,
            regularization=self.config.regularization,
            alpha=self.config.alpha,
            iterations=self.config.iterations,
            num_threads=self.config.num_threads,
            random_state=self.config.random_state,
            dtype=np.float32,
        )

        als.fit(user_items)

        fallback_predictions, fallback_scores = self._get_fallback_predictions(
            data, max_items=self.config.fallback_items
        )

        return MatrixFactorizationRecommender(
            user_ids=user_ids,
            item_ids=item_ids,
            user_embeddings=als.user_factors.astype(np.float32, copy=False),
            item_embeddings=als.item_factors.astype(np.float32, copy=False),
            fallback_predictions=fallback_predictions,
            fallback_scores=fallback_scores,
            training_metadata=self.get_metadata(),
        )
