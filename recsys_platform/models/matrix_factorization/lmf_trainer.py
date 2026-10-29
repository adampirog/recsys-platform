from typing import Annotated

import numpy as np
from implicit.lmf import LogisticMatrixFactorization
from pydantic import Field

from recsys_platform.data import RecommenderDataset

from .model import MatrixFactorizationRecommender
from .trainer import MatrixFactorizationTrainer, MatrixFactorizationTrainerConfig


class LMFTrainerConfig(MatrixFactorizationTrainerConfig):
    embedding_size: Annotated[int, Field(gt=0)] = 32
    learning_rate: Annotated[float, Field(gt=0)] = 1.0
    regularization: Annotated[float, Field(gt=0)] = 0.6
    iterations: Annotated[int, Field(gt=0)] = 30
    negative_sample_ratio: Annotated[int, Field(gt=0)] = 30


class LMFTrainer(MatrixFactorizationTrainer[LMFTrainerConfig]):
    """Train a matrix-factorization recommender using implicit-feedback LMF."""

    def __init__(self, config: LMFTrainerConfig | None = None) -> None:
        super().__init__(config=config or LMFTrainerConfig())

    def fit(self, data: RecommenderDataset) -> MatrixFactorizationRecommender:
        user_items, user_ids, item_ids = self._build_user_item_matrix(data)

        lmf = LogisticMatrixFactorization(
            factors=self.config.embedding_size,
            learning_rate=self.config.learning_rate,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            neg_prop=self.config.negative_sample_ratio,
            num_threads=self.config.num_threads,
            random_state=self.config.random_state,
            dtype=np.float32,
        )

        lmf.fit(user_items)

        fallback_predictions, fallback_scores = self._get_fallback_predictions(
            user_items=user_items,
            item_ids=item_ids,
            max_items=self.config.fallback_items,
        )

        return MatrixFactorizationRecommender(
            user_ids=user_ids,
            item_ids=item_ids,
            user_embeddings=lmf.user_factors.astype(np.float32, copy=False),
            item_embeddings=lmf.item_factors.astype(np.float32, copy=False),
            fallback_predictions=fallback_predictions,
            fallback_scores=fallback_scores,
            training_metadata=self.get_metadata(),
        )
