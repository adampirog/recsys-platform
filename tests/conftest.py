from pathlib import Path

import polars as pl
import pytest

from recsys_platform.data import RecommenderDataset


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            "item_id": [10, 10, 20, 30, 10, 20, 10, 30, 10, 20, 40, 40],
        },
        schema={"user_id": pl.UInt32, "item_id": pl.UInt32},
    )


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory, data: pl.DataFrame) -> Path:
    path = tmp_path_factory.mktemp("popularity") / "interactions.parquet"
    data.write_parquet(path)

    return path


@pytest.fixture(scope="module")
def dataset(data_path: Path) -> RecommenderDataset:
    return RecommenderDataset(data_path)
