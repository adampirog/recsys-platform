from pathlib import Path

import numpy as np
import polars as pl
import pytest

from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.models.popularity.dataset import PopularityDataset


@pytest.fixture(scope="module")
def popularity_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            "item_id": [10, 10, 20, 30, 10, 20, 10, 30, 10, 20, 40, 40],
        },
        schema={"user_id": pl.UInt32, "item_id": pl.UInt32},
    )


@pytest.fixture(scope="module")
def popularity_data_path(
    tmp_path_factory: pytest.TempPathFactory, popularity_data: pl.DataFrame
) -> Path:
    path = tmp_path_factory.mktemp("popularity") / "interactions.parquet"
    popularity_data.write_parquet(path)

    return path


@pytest.fixture(scope="module")
def popularity_dataset(popularity_data_path: Path) -> PopularityDataset:
    return PopularityDataset(popularity_data_path)


@pytest.fixture(scope="module")
def popularity_model() -> PopularityRecommender:
    return PopularityRecommender(
        model_id="testing",
        item_ids=np.array([10, 20, 30, 40], dtype=np.uint32),
        scores=np.array([1.0, 0.6, 0.4, 0.4], dtype=np.float32),
    )
