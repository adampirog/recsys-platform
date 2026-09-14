from collections.abc import Iterator

import numpy as np
import polars as pl

from recsys_platform.data import RecommenderDataset, TargetBatch


class PopularityDataset(RecommenderDataset):
    @property
    def data(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path)

    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        targets = (
            self.data.group_by("user_id")
            .agg(pl.col("item_id").unique().alias("relevant_items"))
            .collect(engine="streaming")
        )

        for batch in targets.iter_slices(n_rows=batch_size):
            yield TargetBatch(
                user_ids=batch["user_id"].to_numpy().astype(np.uint32, copy=False),
                relevant_items=[
                    np.asarray(items, dtype=np.uint32) for items in batch["relevant_items"]
                ],
            )
