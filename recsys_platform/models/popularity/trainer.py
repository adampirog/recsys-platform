import numpy as np
import polars as pl

from recsys_platform.training import Trainer

from .dataset import PopularityDataset
from .model import PopularityRecommender


class PopularityTrainer(Trainer[PopularityDataset, PopularityRecommender]):
    def __init__(self, max_items: int = 100) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be greater than 0.")

        self.max_items = max_items

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
            .head(self.max_items)
            .collect(engine="streaming")
        )

        return PopularityRecommender(
            item_ids=popularity["item_id"].to_numpy().astype(np.uint32, copy=False),
            scores=popularity["score"].to_numpy().astype(np.float32, copy=False),
        )
