import numpy as np
import numpy.testing as npt
import polars as pl
import pytest

from recsys_platform.models import RecommendationRequest
from recsys_platform.models.popularity import PopularityDataset, PopularityRecommender


class MemoryDataset(PopularityDataset):
    """Popularity dataset backed by an in-memory Polars DataFrame."""

    def __init__(self, data: pl.DataFrame) -> None:
        self._data = data

    @property
    def data(self) -> pl.LazyFrame:
        return self._data.lazy()


@pytest.fixture(scope="session")
def popularity_dataset() -> MemoryDataset:
    data = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
            "item_id": [10, 20, 30, 10, 20, 10, 30, 10, 20, 40],
        },
        schema={"user_id": pl.UInt32, "item_id": pl.UInt32},
    )

    return MemoryDataset(data)


def test_fit_orders_items_by_popularity(popularity_dataset) -> None:
    model = PopularityRecommender().fit(popularity_dataset)

    npt.assert_array_equal(model.item_ids, np.array([10, 20, 30, 40], dtype=np.uint32))


def test_fit_normalizes_popularity_scores(popularity_dataset) -> None:
    model = PopularityRecommender().fit(popularity_dataset)
    expected = np.array([1.0, 0.75, 0.5, 0.25], dtype=np.float32)

    npt.assert_allclose(model.scores, expected)


def test_predict_returns_requested_top_k(popularity_dataset) -> None:
    model = PopularityRecommender().fit(popularity_dataset)
    request = RecommendationRequest(user_ids=np.array([101, 102], dtype=np.uint32), k=3)
    result = model.predict(request)

    expected_items = np.array([[10, 20, 30], [10, 20, 30]], dtype=np.uint32)
    npt.assert_array_equal(result.item_ids, expected_items)

    expected_scores = np.array([[1.0, 0.75, 0.5], [1.0, 0.75, 0.5]], dtype=np.float32)
    npt.assert_allclose(result.scores, expected_scores)


def test_predict_caps_k_at_catalog_size(popularity_dataset) -> None:
    model = PopularityRecommender().fit(popularity_dataset)

    result = model.predict(RecommendationRequest(user_ids=np.array([1, 2], dtype=np.uint32), k=100))

    assert result.item_ids.shape == (2, 4)
    assert result.scores.shape == (2, 4)

    npt.assert_array_equal(result.item_ids[0], np.array([10, 20, 30, 40], dtype=np.uint32))


def test_predict_supports_empty_user_batch(popularity_dataset) -> None:
    model = PopularityRecommender().fit(popularity_dataset)

    result = model.predict(RecommendationRequest(user_ids=np.empty(0, dtype=np.uint32), k=3))

    assert result.item_ids.shape == (0, 3)
    assert result.scores.shape == (0, 3)


def test_save_and_load_preserve_model_state(popularity_dataset, tmp_path) -> None:
    original = PopularityRecommender().fit(popularity_dataset)

    original.save(tmp_path)
    loaded = PopularityRecommender.load(tmp_path)

    npt.assert_array_equal(loaded.item_ids, original.item_ids)
    npt.assert_allclose(loaded.scores, original.scores)

    assert loaded.item_ids.dtype == original.item_ids.dtype
    assert loaded.scores.dtype == original.scores.dtype


def test_loaded_model_produces_same_predictions(popularity_dataset, tmp_path) -> None:
    original = PopularityRecommender().fit(popularity_dataset)

    original.save(tmp_path)
    loaded = PopularityRecommender.load(tmp_path)

    request = RecommendationRequest(user_ids=np.array([10, 20, 30], dtype=np.uint32), k=3)

    expected = original.predict(request)
    actual = loaded.predict(request)

    npt.assert_array_equal(actual.item_ids, expected.item_ids)
    npt.assert_allclose(actual.scores, expected.scores)


def test_evaluate_integrates_with_dataset_targets(popularity_dataset) -> None:
    model = PopularityRecommender().fit(popularity_dataset)

    result = model.evaluate(popularity_dataset, k_values=(1,), batch_size=2)

    assert result[1]["precision"] == 1.0
    assert result[1]["hit_rate"] == 1.0
    assert result[1]["ndcg"] == 1.0

    # Per-user recall:
    # user 1: 1/3
    # user 2: 1/2
    # user 3: 1/2
    # user 4: 1/3
    # macro average = 5/12
    npt.assert_allclose(result[1]["recall"], 5 / 12)
