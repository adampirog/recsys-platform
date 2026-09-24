import numpy as np
import numpy.testing as npt

from recsys_platform.models import RecommendationRequest
from recsys_platform.models.popularity import PopularityRecommender


def test_predict_returns_top_k_for_each_user(popularity_model: PopularityRecommender) -> None:
    request = RecommendationRequest(user_ids=np.array([101, 102], dtype=np.uint32), k=3)

    result = popularity_model.predict(request)

    expected_items = np.array([[10, 20, 30], [10, 20, 30]], dtype=np.uint32)
    expected_scores = np.array([[1.0, 0.6, 0.4], [1.0, 0.6, 0.4]], dtype=np.float32)

    npt.assert_array_equal(result.item_ids, expected_items)
    npt.assert_allclose(result.scores, expected_scores)


def test_predict_caps_k_at_catalog_size(popularity_model: PopularityRecommender) -> None:
    request = RecommendationRequest(user_ids=np.array([1, 2], dtype=np.uint32), k=100)

    result = popularity_model.predict(request)

    assert result.item_ids.shape == (2, 4)
    assert result.scores.shape == (2, 4)

    npt.assert_array_equal(result.item_ids[0], np.array([10, 20, 30, 40], dtype=np.uint32))


def test_predict_supports_empty_user_batch(popularity_model: PopularityRecommender) -> None:
    request = RecommendationRequest(user_ids=np.empty(0, dtype=np.uint32), k=3)

    result = popularity_model.predict(request)

    assert result.item_ids.shape == (0, 3)
    assert result.scores.shape == (0, 3)


def test_predict_preserves_model_dtypes(popularity_model: PopularityRecommender) -> None:
    result = popularity_model.predict(
        RecommendationRequest(user_ids=np.array([1], dtype=np.uint32), k=2)
    )

    assert result.item_ids.dtype == np.uint32
    assert result.scores.dtype == np.float32


def test_save_and_load_preserve_model_state(
    popularity_model: PopularityRecommender, tmp_path
) -> None:
    popularity_model.save(tmp_path)

    loaded = PopularityRecommender.load(tmp_path)

    npt.assert_array_equal(loaded.item_ids, popularity_model.item_ids)
    npt.assert_allclose(loaded.scores, popularity_model.scores)

    assert loaded.item_ids.dtype == np.uint32
    assert loaded.scores.dtype == np.float32
    assert loaded.model_id == popularity_model.model_id
    assert loaded.training_metadata == popularity_model.training_metadata


def test_loaded_model_produces_same_predictions(
    popularity_model: PopularityRecommender, tmp_path
) -> None:
    popularity_model.save(tmp_path)
    loaded = PopularityRecommender.load(tmp_path)

    request = RecommendationRequest(user_ids=np.array([10, 20, 30], dtype=np.uint32), k=3)

    expected = popularity_model.predict(request)
    actual = loaded.predict(request)

    npt.assert_array_equal(actual.item_ids, expected.item_ids)
    npt.assert_allclose(actual.scores, expected.scores)
