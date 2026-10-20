from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator

from recsys_platform.types import UInt32Vector


class TargetBatch(BaseModel):
    """Ground-truth relevant items aligned with a batch of user IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    user_ids: UInt32Vector
    relevant_items: list[UInt32Vector]

    @model_validator(mode="after")
    def validate_batch_size(self) -> Self:
        if len(self.relevant_items) != self.user_ids.size:
            raise ValueError("relevant_items must contain one array for each user_id")

        return self


class RecommenderDataset:
    """
    Parquet-backed user-item interaction dataset.

    Requires ``user_id`` and ``item_id`` columns. Evaluation targets contain
    the unique items associated with each user.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def data(self) -> pl.LazyFrame:
        """Return the interaction dataset as a lazy Polars frame."""
        return pl.scan_parquet(self.path)

    @cached_property
    def n_targets(self) -> int:
        "Return the number of logical evaluation targets"
        return self.data.select(pl.col("user_id").n_unique()).collect(engine="streaming").item()

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
