from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from recsys_platform.types import UInt32Vector


@dataclass(frozen=True, slots=True)
class TargetBatch:
    """Ground-truth items grouped by user.

    Each relevant_items entry corresponds to the user ID at the same position.
    """

    user_ids: UInt32Vector
    relevant_items: list[UInt32Vector]


class RecommenderDataset:
    """Read a Parquet-backed interaction dataset.

    Requires user_id and item_id columns containing uint32-compatible IDs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def data(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path)

    @cached_property
    def n_targets(self) -> int:
        """Count distinct users, not interaction rows."""
        return self.data.select(pl.col("user_id").n_unique()).collect(engine="streaming").item()

    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        """Yield unique relevant items per user in batches without guaranteed user order."""

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
