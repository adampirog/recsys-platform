import json
from math import log2
from pathlib import Path
from typing import Self

import pytest

from recsys_platform.models.base import (
    EvaluationResult,
    Recommendation,
    RecommendationRequest,
    Recommender,
    RecommenderDataset,
)


class FakeRecommender(Recommender):
    def __init__(self) -> None:
        self.calls: list[list[RecommendationRequest]] = []
        self.predictions = {1: [10, 40, 20], 2: [30, 20, 10], 3: [30, 40, 10]}

    def fit(self, data: RecommenderDataset) -> Self:
        return self

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


def test_model_evaluate(testing_dataset) -> None:
    model = FakeRecommender()

    result = model.evaluate(testing_dataset, k_values=(1, 3))
    assert list(result) == [1, 3]

    # K = 1
    assert result[1]["precision"] == pytest.approx(1 / 3)
    assert result[1]["recall"] == pytest.approx(1 / 9)
    assert result[1]["hit_rate"] == pytest.approx(1 / 3)
    assert result[1]["ndcg"] == pytest.approx(1 / 3)

    # K = 3
    assert result[3]["precision"] == pytest.approx(5 / 9)
    assert result[3]["recall"] == pytest.approx(8 / 9)
    assert result[3]["hit_rate"] == pytest.approx(1.0)

    user_1_ndcg = (1.0 + 1.0 / log2(4)) / (1.0 + 1.0 / log2(3) + 1.0 / log2(4))

    user_2_ndcg = (1.0 / log2(3) + 1.0 / log2(4)) / (1.0 + 1.0 / log2(3))

    user_3_ndcg = 1.0 / log2(4)

    assert result[3]["ndcg"] == pytest.approx((user_1_ndcg + user_2_ndcg + user_3_ndcg) / 3)


def test_model_evaluate_predicts_once_at_max_k(testing_dataset) -> None:
    model = FakeRecommender()

    model.evaluate(testing_dataset, k_values=(1, 2, 3), batch_size=10)

    assert len(model.calls) == 1

    requests = model.calls[0]

    assert len(requests) == 3
    assert all(request.k == 3 for request in requests)


def test_model_evaluate_batches_predictions(testing_dataset) -> None:
    model = FakeRecommender()

    model.evaluate(testing_dataset, k_values=(3,), batch_size=2)

    assert [len(call) for call in model.calls] == [2, 1]


@pytest.fixture(scope="module")
def evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        {
            5: {"precision": 0.12345, "recall": 0.25, "hit_rate": 0.5, "ndcg": 0.45678},
            10: {"precision": 0.1, "recall": 0.4, "hit_rate": 0.7, "ndcg": 0.5},
        }
    )


def test_evaluation_result_save_string(evaluation_result, tmp_path) -> None:
    path = tmp_path / "evaluation.txt"
    evaluation_result.save(path)

    assert path.read_text(encoding="utf-8") == str(evaluation_result)


def test_evaluation_result_save_json(evaluation_result, tmp_path) -> None:
    path = tmp_path / "evaluation.json"
    evaluation_result.save(path, save_format="json")

    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved == {
        "5": {"precision": 0.12345, "recall": 0.25, "hit_rate": 0.5, "ndcg": 0.45678},
        "10": {"precision": 0.1, "recall": 0.4, "hit_rate": 0.7, "ndcg": 0.5},
    }
