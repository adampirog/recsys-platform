import numpy as np
import polars as pl

from recsys_platform.models.popularity import PopularityDataset


def test_data_returns_parquet_as_lazy_frame(popularity_dataset: PopularityDataset) -> None:
    data = popularity_dataset.data

    assert isinstance(data, pl.LazyFrame)

    result = data.collect()

    assert result.height == 12
    assert result.schema == {"user_id": pl.UInt32, "item_id": pl.UInt32}


def test_iter_targets_builds_unique_relevant_items_per_user(
    popularity_dataset: PopularityDataset,
) -> None:
    batches = list(popularity_dataset.iter_targets(batch_size=2))

    targets: dict[int, set[int]] = {}

    for batch in batches:
        for user_id, relevant_items in zip(batch.user_ids, batch.relevant_items, strict=True):
            targets[int(user_id)] = {int(item_id) for item_id in relevant_items}

    assert targets == {1: {10, 20, 30}, 2: {10, 20}, 3: {10, 30}, 4: {10, 20, 40}}


def test_iter_targets_respects_batch_size(popularity_dataset: PopularityDataset) -> None:
    batches = list(popularity_dataset.iter_targets(batch_size=2))

    assert len(batches) == 2
    assert all(batch.user_ids.size <= 2 for batch in batches)


def test_iter_targets_uses_expected_dtypes(popularity_dataset: PopularityDataset) -> None:
    batches = list(popularity_dataset.iter_targets(batch_size=4))

    assert len(batches) == 1

    batch = batches[0]

    assert batch.user_ids.dtype == np.uint32
    assert all(items.dtype == np.uint32 for items in batch.relevant_items)
