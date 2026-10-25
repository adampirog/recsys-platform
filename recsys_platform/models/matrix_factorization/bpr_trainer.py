from typing import Annotated

import numpy as np
from implicit.bpr import BayesianPersonalizedRanking
from pydantic import Field

from recsys_platform.data import RecommenderDataset

from .model import MatrixFactorizationRecommender
from .trainer import MatrixFactorizationTrainer, MatrixFactorizationTrainerConfig


class BPRTrainerConfig(MatrixFactorizationTrainerConfig):
    embedding_size: Annotated[int, Field(gt=0)] = 64
    regularization: Annotated[float, Field(gt=0)] = 0.01
    learning_rate: Annotated[float, Field(gt=0)] = 0.01
    iterations: Annotated[int, Field(gt=0)] = 20


class BPRTrainer(MatrixFactorizationTrainer[BPRTrainerConfig]):
    def __init__(self, config: BPRTrainerConfig | None = None) -> None:
        super().__init__(config=config or BPRTrainerConfig())

    def fit(self, data: RecommenderDataset) -> MatrixFactorizationRecommender:
        user_items, user_ids, item_ids = self._build_user_item_matrix(data)

        bpr = BayesianPersonalizedRanking(
            factors=self.config.embedding_size,
            regularization=self.config.regularization,
            learning_rate=self.config.learning_rate,
            iterations=self.config.iterations,
            dtype=np.float32,
            use_gpu=False,
            random_state=self.config.random_state,
        )

        bpr.fit(user_items)

        return MatrixFactorizationRecommender(
            user_ids=user_ids,
            item_ids=item_ids,
            user_embeddings=bpr.user_factors.astype(np.float32, copy=False),
            item_embeddings=bpr.item_factors.astype(np.float32, copy=False),
            training_metadata=self.get_metadata(),
        )
