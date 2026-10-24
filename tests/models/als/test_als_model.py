import numpy as np
import numpy.testing as npt
import pytest

from recsys_platform.models import RecommendationRequest
from recsys_platform.models.als import ALSRecommender


@pytest.fixture
def model() -> ALSRecommender:
    return ALSRecommender(
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
    )


def test_predict_returns_personalized_top_k(model: ALSRecommender) -> None:
    result = model.predict(RecommendationRequest(user_ids=np.array([20, 10], dtype=np.uint32), k=2))

    npt.assert_array_equal(result.item_ids, np.array([[200, 300], [100, 300]], dtype=np.uint32))
    npt.assert_allclose(result.scores, np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32))


def test_predict_caps_k_at_number_of_items(model: ALSRecommender) -> None:
    result = model.predict(RecommendationRequest(user_ids=np.array([10], dtype=np.uint32), k=10))

    npt.assert_array_equal(result.item_ids, np.array([[100, 300, 200]], dtype=np.uint32))

    assert result.item_ids.shape == (1, 3)
    assert result.scores.shape == (1, 3)


def test_predict_rejects_unknown_user(model: ALSRecommender) -> None:
    request = RecommendationRequest(user_ids=np.array([10, 999], dtype=np.uint32), k=2)

    with pytest.raises(ValueError, match="Unknown user_id: 999"):
        model.predict(request)


def test_save_load_round_trip(model: ALSRecommender, tmp_path) -> None:
    model.save(tmp_path)

    loaded = ALSRecommender.load(tmp_path)

    assert loaded.model_id == model.model_id

    npt.assert_array_equal(loaded.user_ids, model.user_ids)
    npt.assert_array_equal(loaded.item_ids, model.item_ids)
    npt.assert_array_equal(loaded.user_embeddings, model.user_embeddings)
    npt.assert_array_equal(loaded.item_embeddings, model.item_embeddings)


def test_rejects_incompatible_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="User and item embeddings must have the same dimension"):
        ALSRecommender(
            user_ids=np.array([10], dtype=np.uint32),
            item_ids=np.array([100], dtype=np.uint32),
            user_embeddings=np.ones((1, 2), dtype=np.float32),
            item_embeddings=np.ones((1, 3), dtype=np.float32),
        )
