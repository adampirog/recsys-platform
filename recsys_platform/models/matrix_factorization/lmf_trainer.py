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
    negative_samples: Annotated[int, Field(gt=0)] = 30


class LMFTrainer(MatrixFactorizationTrainer[LMFTrainerConfig]):
    def __init__(self, config: LMFTrainerConfig | None = None) -> None:
        super().__init__(config=config or LMFTrainerConfig())

    def fit(self, data: RecommenderDataset) -> MatrixFactorizationRecommender:
        user_items, user_ids, item_ids = self._build_user_item_matrix(data)

        lmf = LogisticMatrixFactorization(
            factors=self.config.embedding_size,
            regularization=self.config.regularization,
            learning_rate=self.config.learning_rate,
            iterations=self.config.iterations,
            dtype=np.float32,
            use_gpu=False,
            random_state=self.config.random_state,
        )

        lmf.fit(user_items)

        return MatrixFactorizationRecommender(
            user_ids=user_ids,
            item_ids=item_ids,
            user_embeddings=lmf.user_factors.astype(np.float32, copy=False),
            item_embeddings=lmf.item_factors.astype(np.float32, copy=False),
            training_metadata=self.get_metadata(),
        )
