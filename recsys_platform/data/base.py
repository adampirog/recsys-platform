from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from recsys_platform.types import UInt32Vector


class TargetBatch(BaseModel):
    """Ground-truth relevance data for a batch of users.

    ``relevant_items[i]`` contains the relevant item IDs for
    ``user_ids[i]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    user_ids: UInt32Vector
    relevant_items: list[UInt32Vector]

    @model_validator(mode="after")
    def validate_batch_size(self) -> Self:
        if len(self.relevant_items) != self.user_ids.size:
            raise ValueError("relevant_items must contain one array for each user_id")

        return self


class RecommenderDataset(ABC):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @abstractmethod
    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        """Yield evaluation targets in batches."""
