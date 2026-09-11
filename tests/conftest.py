# tests/conftest.py

from datetime import datetime

import polars as pl
import pytest

from recsys_platform.models.base import RecommendationDataset


class TestingDataset(RecommendationDataset):
    """Recommendation dataset backed by in-memory Polars DataFrames."""

    def __init__(
        self,
        train: pl.DataFrame,
        valid: pl.DataFrame | None = None,
        test: pl.DataFrame | None = None,
    ) -> None:
        self._train = train
        self._valid = valid if valid is not None else train.clear()
        self._test = test if test is not None else train.clear()

    @property
    def train(self) -> pl.LazyFrame:
        return self._train.lazy()

    @property
    def valid(self) -> pl.LazyFrame:
        return self._valid.lazy()

    @property
    def test(self) -> pl.LazyFrame:
        return self._test.lazy()


@pytest.fixture(scope="session")
def recommendation_dataset() -> TestingDataset:
    train = pl.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 1],
            "item_id": [10, 10, 10, 20, 20, 30],
            "category_id": [1, 1, 1, 2, 2, 3],
            "event_type": [
                "view",
                "view",
                "purchase",
                "view",
                "cart",
                "view",
            ],
            "event_time": [
                datetime(2017, 11, 25, 10, 0),
                datetime(2017, 11, 25, 11, 0),
                datetime(2017, 11, 25, 12, 0),
                datetime(2017, 11, 26, 10, 0),
                datetime(2017, 11, 26, 11, 0),
                datetime(2017, 11, 27, 10, 0),
            ],
        }
    )

    return TestingDataset(train)
