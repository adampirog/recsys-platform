from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from recsys_platform.data import RecommenderDataset, TargetBatch


class PopularityDataset(RecommenderDataset):
    """
    Dataset adapter for the global popularity recommender.

    Requires ``user_id`` and ``item_id`` columns; evaluation uses unique items per user.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def data(self) -> pl.LazyFrame:
        """Return the interaction dataset as a lazy Polars frame."""
        return pl.scan_parquet(self.path)

    @cached_property
    def _n_targets(self) -> int:
        return self.data.select(pl.col("user_id").n_unique()).collect(engine="streaming").item()

    @property
    def n_targets(self) -> int:
        return self._n_targets

    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        """Yield per-user unique relevant items in batches."""

        targets = self.data.group_by("user_id").agg(
            pl.col("item_id").unique().alias("relevant_items")
        )

        for batch in targets.collect_batches(
            chunk_size=batch_size, maintain_order=False, engine="streaming"
        ):
            yield TargetBatch(
                user_ids=batch["user_id"].to_numpy().astype(np.uint32, copy=False),
                relevant_items=[
                    np.asarray(items, dtype=np.uint32) for items in batch["relevant_items"]
                ],
            )
