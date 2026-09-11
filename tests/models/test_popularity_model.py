# tests/models/test_popularity.py

import pytest

from recsys_platform.models.base import RecommendationRequest
from recsys_platform.models.popularity_model import PopularityRecommender


def test_fit_orders_items_by_popularity(
    recommendation_dataset,
):
    model = PopularityRecommender().fit(recommendation_dataset)

    assert [item.item_id for item in model.popular_items] == [10, 20, 30]

    assert model.popular_items[0].score == pytest.approx(1.0)
    assert model.popular_items[1].score == pytest.approx(2 / 3)
    assert model.popular_items[2].score == pytest.approx(1 / 3)


def test_scores_are_normalized(
    recommendation_dataset,
):
    model = PopularityRecommender().fit(recommendation_dataset)

    scores = [recommendation.score for recommendation in model.popular_items]

    assert max(scores) == pytest.approx(1.0)
    assert min(scores) >= 0.0
    assert max(scores) <= 1.0


def test_predict_respects_k(
    recommendation_dataset,
):
    model = PopularityRecommender().fit(recommendation_dataset)

    result = model.predict(
        [
            RecommendationRequest(
                user_id=42,
                k=2,
            )
        ]
    )

    assert [item.item_id for item in result[42]] == [10, 20]


def test_predict_supports_different_k_per_user(
    recommendation_dataset,
):
    model = PopularityRecommender().fit(recommendation_dataset)

    result = model.predict(
        [
            RecommendationRequest(user_id=1, k=1),
            RecommendationRequest(user_id=2, k=3),
        ]
    )

    assert [item.item_id for item in result[1]] == [10]
    assert [item.item_id for item in result[2]] == [10, 20, 30]


def test_predict_with_empty_requests(
    recommendation_dataset,
):
    model = PopularityRecommender().fit(recommendation_dataset)

    assert model.predict([]) == {}


def test_predict_with_k_larger_than_catalog(
    recommendation_dataset,
):
    model = PopularityRecommender().fit(recommendation_dataset)

    result = model.predict(
        [
            RecommendationRequest(
                user_id=42,
                k=100,
            )
        ]
    )

    assert len(result[42]) == 3


def test_save_and_load(
    recommendation_dataset,
    tmp_path,
):
    original = PopularityRecommender().fit(recommendation_dataset)

    original.save(tmp_path)

    loaded = PopularityRecommender.load(tmp_path)

    assert loaded.popular_items == original.popular_items

    requests = [
        RecommendationRequest(
            user_id=42,
            k=2,
        )
    ]

    assert loaded.predict(requests) == original.predict(requests)
