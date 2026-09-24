import json
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import numpy as np
import pytest
from numpy import testing as npt

from recsys_platform.data import RecommenderDataset, TargetBatch
from recsys_platform.evaluation import EvaluationResult
from recsys_platform.models import Recommendation, RecommendationRequest, Recommender
from recsys_platform.training import Trainer, TrainerConfig


class FakeDataset(RecommenderDataset):
    """In-memory dataset for testing generic Trainer evaluation."""

    def __init__(self, batches: list[TargetBatch] | None = None) -> None:
        self.batches = (
            batches
            if batches is not None
            else [
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
        )

        self.requested_batch_sizes: list[int] = []

    def iter_targets(
        self,
        batch_size: int,
    ) -> Iterator[TargetBatch]:
        self.requested_batch_sizes.append(batch_size)
        yield from self.batches


class FakeRecommender(Recommender):
    """Deterministic recommender used to verify evaluation behavior."""

    MODEL_FAMILY = "fake"

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__(model_id=model_id)
        self.requests: list[RecommendationRequest] = []

    def predict(self, request: RecommendationRequest) -> Recommendation:
        self.requests.append(request)

        predictions = {1: [10, 99, 20], 2: [99, 30, 98], 3: [40, 98, 97]}

        item_ids = np.array(
            [predictions[int(user_id)][: request.k] for user_id in request.user_ids],
            dtype=np.uint32,
        )

        return Recommendation(
            item_ids=item_ids,
            scores=np.ones_like(
                item_ids,
                dtype=np.float32,
            ),
        )

    def _save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def _load(cls, path: Path, manifest) -> Self:
        raise NotImplementedError


class FakeTrainer(Trainer[FakeDataset, FakeRecommender, TrainerConfig]):
    def __init__(self, config: TrainerConfig | None = None) -> None:
        super().__init__(config or TrainerConfig())

    def fit(self, data: FakeDataset) -> FakeRecommender:
        return FakeRecommender(model_id="testing")


def test_evaluate_macro_averages_user_metrics() -> None:
    trainer = FakeTrainer()
    model = FakeRecommender(model_id="testing")

    result = trainer.evaluate(model, FakeDataset(), k_values=(1,))

    # user 1: hit -> precision=1, recall=1/2
    # user 2: miss -> precision=0, recall=0
    # user 3: hit -> precision=1, recall=1/2
    assert result[1]["precision"] == pytest.approx(2 / 3)
    assert result[1]["recall"] == pytest.approx(1 / 3)
    assert result[1]["hit_rate"] == pytest.approx(2 / 3)
    assert result[1]["ndcg"] == pytest.approx(2 / 3)


def test_evaluate_predicts_once_per_batch_at_max_k() -> None:
    trainer = FakeTrainer()
    model = FakeRecommender()
    dataset = FakeDataset()

    result = trainer.evaluate(model, dataset, k_values=(3, 1, 3), batch_size=2)

    assert list(result) == [1, 3]

    assert len(model.requests) == 2
    assert all(request.k == 3 for request in model.requests)

    npt.assert_array_equal(model.requests[0].user_ids, np.array([1, 2], dtype=np.uint32))
    npt.assert_array_equal(model.requests[1].user_ids, np.array([3], dtype=np.uint32))

    assert dataset.requested_batch_sizes == [2]


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
    evaluation_result.save(path)

    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["5"]["precision"] == 0.25
    assert saved["10"]["ndcg"] == 0.8


def test_result_save_str(evaluation_result, tmp_path) -> None:
    path = tmp_path / "result.txt"
    evaluation_result.save(path)

    output = path.read_text(encoding="utf-8")

    assert "Precision" in output
    assert "Recall" in output
    assert "Hit Rate" in output
    assert "Ndcg" in output
