import numpy as np
import numpy.testing as npt
import pytest

from recsys_platform.models import RecommendationRequest
from recsys_platform.models.matrix_factorization import MatrixFactorizationRecommender


@pytest.fixture
def model() -> MatrixFactorizationRecommender:
    return MatrixFactorizationRecommender(
        model_id="testing",
        user_ids=np.array([10, 20], dtype=np.uint32),
        item_ids=np.array([100, 200, 300], dtype=np.uint32),
        user_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        item_embeddings=np.array(
            [
                [1.0, 0.0],  # item 100
                [0.0, 1.0],  # item 200
                [0.5, 0.5],  # item 300
            ],
            dtype=np.float32,
        ),
        fallback_predictions=np.array([100, 200], dtype=np.uint32),
        fallback_scores=np.array([0.9, 0.1], dtype=np.float32),
    )


def test_predict_returns_personalized_top_k(model: MatrixFactorizationRecommender) -> None:
    result = model.predict(RecommendationRequest(user_ids=np.array([20, 10], dtype=np.uint32), k=2))

    npt.assert_array_equal(result.item_ids, np.array([[200, 300], [100, 300]], dtype=np.uint32))
    npt.assert_allclose(result.scores, np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32))


def test_predict_caps_k_at_number_of_items(model: MatrixFactorizationRecommender) -> None:
    result = model.predict(RecommendationRequest(user_ids=np.array([10], dtype=np.uint32), k=10))

    npt.assert_array_equal(result.item_ids, np.array([[100, 300, 200]], dtype=np.uint32))

    assert result.item_ids.shape == (1, 3)
    assert result.scores.shape == (1, 3)


def test_predict_cold_user(model: MatrixFactorizationRecommender) -> None:
    request = RecommendationRequest(user_ids=np.array([999, 666], dtype=np.uint32), k=2)
    result = model.predict(request)

    assert result.item_ids.shape == (2, 2)
    assert result.scores.shape == (2, 2)

    npt.assert_array_equal(result.item_ids[0], [100, 200])
    npt.assert_array_almost_equal(result.scores[0], [0.9, 0.1])


def test_save_load_round_trip(model: MatrixFactorizationRecommender, tmp_path) -> None:
    model.save(tmp_path)

    loaded = MatrixFactorizationRecommender.load(tmp_path)

    assert loaded.model_id == model.model_id

    npt.assert_array_equal(loaded.user_ids, model.user_ids)
    npt.assert_array_equal(loaded.item_ids, model.item_ids)
    npt.assert_array_equal(loaded.user_embeddings, model.user_embeddings)
    npt.assert_array_equal(loaded.item_embeddings, model.item_embeddings)
    npt.assert_array_equal(loaded.fallback_predictions, model.fallback_predictions)
    npt.assert_array_equal(loaded.fallback_scores, model.fallback_scores)


def test_rejects_incompatible_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="User and item embeddings must have the same dimension"):
        MatrixFactorizationRecommender(
            user_ids=np.array([10], dtype=np.uint32),
            item_ids=np.array([100], dtype=np.uint32),
            user_embeddings=np.ones((1, 2), dtype=np.float32),
            item_embeddings=np.ones((1, 3), dtype=np.float32),
            fallback_predictions=np.array([1, 2], dtype=np.uint32),
            fallback_scores=np.array([0.9, 0.2], dtype=np.float32),
        )
