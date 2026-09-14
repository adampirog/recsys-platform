import json
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from recsys_platform.data import RecommenderDataset, TargetBatch
from recsys_platform.evaluation import EvaluationResult
from recsys_platform.models import Recommendation, RecommendationRequest, Recommender


class FakeDataset(RecommenderDataset):
    def __init__(self) -> None:
        pass

    def iter_targets(self, batch_size: int) -> Iterator[TargetBatch]:
        batches = [
            TargetBatch(
                user_ids=np.array([1, 2], dtype=np.uint32),
                relevant_items=[
                    np.array([10, 20], dtype=np.uint32),
                    np.array([30], dtype=np.uint32),
                ],
            ),
            TargetBatch(
                user_ids=np.array([3], dtype=np.uint32),
                relevant_items=[np.array([40, 50], dtype=np.uint32)],
            ),
        ]

        yield from batches


class FakeRecommender(Recommender[FakeDataset]):
    def __init__(self) -> None:
        self.requests: list[RecommendationRequest] = []  # records requests for testing

    def fit(self, data: FakeDataset) -> Self:
        return self

    def predict(
        self,
        request: RecommendationRequest,
    ) -> Recommendation:
        self.requests.append(request)

        predictions = {1: [10, 99, 20], 2: [99, 30, 98], 3: [40, 98, 97]}
        item_ids = np.array(
            [predictions[int(user_id)][: request.k] for user_id in request.user_ids],
            dtype=np.uint32,
        )

        return Recommendation(item_ids=item_ids, scores=np.ones_like(item_ids, dtype=np.float32))

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> Self:
        raise NotImplementedError


def test_evaluate_processes_multiple_target_batches() -> None:
    model = FakeRecommender()
    dataset = FakeDataset()

    result = model.evaluate(dataset, k_values=(1, 3), batch_size=2)

    assert list(result) == [1, 3]


def test_evaluate_macro_averages_user_metrics() -> None:
    result = FakeRecommender().evaluate(FakeDataset(), k_values=(1,))

    np.testing.assert_allclose(result[1]["precision"], 2 / 3)
    np.testing.assert_allclose(result[1]["recall"], 1 / 3)
    np.testing.assert_allclose(result[1]["hit_rate"], 2 / 3)


def test_evaluate_predicts_once_per_batch_at_max_k() -> None:
    model = FakeRecommender()

    model.evaluate(FakeDataset(), k_values=(1, 3, 5), batch_size=2)

    assert len(model.requests) == 2
    assert all(request.k == 5 for request in model.requests)


@pytest.fixture(scope="module")
def evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        {
            5: {"precision": 0.25, "recall": 0.5, "hit_rate": 0.75, "ndcg": 0.6},
            10: {"precision": 0.2, "recall": 0.7, "hit_rate": 1.0, "ndcg": 0.8},
        }
    )


def test_result_behaves_like_mapping(evaluation_result) -> None:
    assert evaluation_result[5]["recall"] == 0.5
    assert list(evaluation_result) == [5, 10]


def test_result_save_json(evaluation_result, tmp_path) -> None:
    path = tmp_path / "result.json"
    evaluation_result.save(path, save_format="json")

    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["5"]["precision"] == 0.25
    assert saved["10"]["ndcg"] == 0.8


def test_result_save_str(evaluation_result, tmp_path) -> None:
    path = tmp_path / "result.txt"
    evaluation_result.save(path, save_format="str")

    output = path.read_text(encoding="utf-8")

    assert "Precision" in output
    assert "Recall" in output
    assert "Hit Rate" in output
    assert "Ndcg" in output
