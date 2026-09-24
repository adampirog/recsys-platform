from typing import Annotated

import numpy as np
import polars as pl
from pydantic import Field

from recsys_platform.training import Trainer, TrainerConfig

from .dataset import PopularityDataset
from .model import PopularityRecommender


class PopularityTrainerConfig(TrainerConfig):
    max_items: Annotated[int, Field(gt=0)] = 100


class PopularityTrainer(Trainer[PopularityDataset, PopularityRecommender, PopularityTrainerConfig]):
    """Train a global popularity recommender from interaction counts."""

    def __init__(self, config: PopularityTrainerConfig | None = None) -> None:
        super().__init__(config=config or PopularityTrainerConfig())

    def fit(self, data: PopularityDataset) -> PopularityRecommender:
        popularity = (
            data.data.group_by("item_id")
            .len()
            .sort(
                ["len", "item_id"],
                descending=[True, False],
            )
            .with_columns((pl.col("len") / pl.col("len").max()).alias("score"))
            .select("item_id", "score")
            .head(self.config.max_items)
            .collect(engine="streaming")
        )

        return PopularityRecommender(
            item_ids=popularity["item_id"].to_numpy().astype(np.uint32, copy=False),
            scores=popularity["score"].to_numpy().astype(np.float32, copy=False),
            training_metadata=self.get_metadata(),
        )
