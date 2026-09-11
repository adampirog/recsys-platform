from math import log2
from pathlib import Path
from typing import Self

import pytest

from recsys_platform.evaluation.evaluator import EvaluationResult, build_targets, evaluate
from recsys_platform.models.base import (
    Recommendation,
    RecommendationDataset,
    RecommendationRequest,
    Recommender,
)


class FakeRecommender(Recommender):
    """Recommender returning deterministic predictions for evaluator tests."""

    def __init__(self) -> None:
        self.calls: list[list[RecommendationRequest]] = []
        self.predictions = {1: [10, 40, 20], 2: [30, 20, 10], 3: [30, 40, 10]}

    def fit(self, data: RecommendationDataset) -> Self:
        raise NotImplementedError()

    def predict(
        self,
        requests: list[RecommendationRequest],
    ) -> dict[int, list[Recommendation]]:
        self.calls.append(requests)

        return {
            request.user_id: [
                Recommendation(item_id=item_id, score=1.0)
                for item_id in self.predictions[request.user_id][: request.k]
            ]
            for request in requests
        }

    def save(self, path: str | Path) -> None:
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        raise NotImplementedError()


def test_build_targets(testing_dataset) -> None:
    targets = build_targets(testing_dataset.train)

    result = {user_id: set(relevant_items) for user_id, relevant_items in targets.iter_rows()}

    assert result == {1: {10, 20, 30}, 2: {10, 20}, 3: {10}}


def test_evaluate(
    testing_dataset,
) -> None:
    model = FakeRecommender()

    result = evaluate(model, testing_dataset, k_values=(1, 3), split="train")

    assert isinstance(result, EvaluationResult)
    assert list(result) == [1, 3]

    # -----------------------------
    # K = 1
    #
    # user 1: [10] -> hit
    # user 2: [30] -> miss
    # user 3: [30] -> miss
    # -----------------------------

    assert result[1]["precision"] == pytest.approx(1 / 3)
    assert result[1]["recall"] == pytest.approx(1 / 9)
    assert result[1]["hit_rate"] == pytest.approx(1 / 3)
    assert result[1]["ndcg"] == pytest.approx(1 / 3)

    # -----------------------------
    # K = 3
    # -----------------------------

    assert result[3]["precision"] == pytest.approx(5 / 9)
    assert result[3]["recall"] == pytest.approx(8 / 9)
    assert result[3]["hit_rate"] == pytest.approx(1.0)

    user_1_ndcg = (1.0 + 1.0 / log2(4)) / (1.0 + 1.0 / log2(3) + 1.0 / log2(4))

    user_2_ndcg = (1.0 / log2(3) + 1.0 / log2(4)) / (1.0 + 1.0 / log2(3))

    user_3_ndcg = 1.0 / log2(4)

    expected_ndcg = (user_1_ndcg + user_2_ndcg + user_3_ndcg) / 3

    assert result[3]["ndcg"] == pytest.approx(expected_ndcg)


def test_evaluate_predicts_only_at_max_k(testing_dataset) -> None:
    model = FakeRecommender()

    evaluate(model, testing_dataset, k_values=(1, 2, 3), split="train", batch_size=10)

    assert len(model.calls) == 1

    requests = model.calls[0]

    assert len(requests) == 3
    assert all(request.k == 3 for request in requests)


def test_evaluate_batches_requests(testing_dataset) -> None:
    model = FakeRecommender()

    evaluate(model, testing_dataset, k_values=(3,), split="train", batch_size=2)

    assert len(model.calls) == 2

    assert [len(batch) for batch in model.calls] == [2, 1]

    assert all(request.k == 3 for batch in model.calls for request in batch)
