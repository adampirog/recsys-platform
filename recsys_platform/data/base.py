from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TargetBatch:
    user_ids: NDArray[np.uint32]
    relevant_items: list[NDArray[np.uint32]]


class RecommenderDataset(ABC):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @abstractmethod
    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        """Yield evaluation targets in batches."""
