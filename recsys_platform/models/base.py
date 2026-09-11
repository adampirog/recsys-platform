from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import polars as pl


class RecommendationDataset:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def train(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "train.parquet")

    @property
    def valid(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "valid.parquet")

    @property
    def test(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "test.parquet")


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    score: float


@dataclass(frozen=True)
class RecommendationRequest:
    user_id: int
    k: int = 20


class Recommender(ABC):
    @abstractmethod
    def fit(self, data: RecommendationDataset) -> Self: ...

    @abstractmethod
    def predict(self, requests: list[RecommendationRequest]) -> dict[int, list[Recommendation]]: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self: ...
