from collections.abc import Iterator

import numpy as np
import polars as pl

from recsys_platform.data import RecommenderDataset, TargetBatch


class PopularityDataset(RecommenderDataset):
    """Dataset adapter for the global popularity recommender.

    Interaction data must contain ``user_id`` and ``item_id`` columns.
    During evaluation, each unique item interacted with by a user is treated
    as a binary relevant item.
    """

    @property
    def data(self) -> pl.LazyFrame:
        """Return the interaction dataset as a lazy Polars frame."""
        return pl.scan_parquet(self.path)

    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        """Yield user-level relevance targets in batches.

        Interactions are grouped by user and duplicate item interactions are
        collapsed so each relevant item appears once per user.

        Args:
            batch_size: Maximum number of users in each yielded batch.

        Yields:
            Batches of user IDs and their relevant item IDs.
        """
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
