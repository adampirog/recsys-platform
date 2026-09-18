from collections.abc import Iterator

import numpy as np
import polars as pl

from recsys_platform.data import RecommenderDataset, TargetBatch


class PopularityDataset(RecommenderDataset):
    """
    Dataset adapter for the global popularity recommender.

    Requires ``user_id`` and ``item_id`` columns; evaluation uses unique items per user.
    """

    @property
    def data(self) -> pl.LazyFrame:
        """Return the interaction dataset as a lazy Polars frame."""
        return pl.scan_parquet(self.path)

    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        """Yield per-user unique relevant items in batches."""

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
