import numpy as np
import pytest
from pydantic import ValidationError

from recsys_platform.data import TargetBatch
from recsys_platform.models import Recommendation, RecommendationRequest


@pytest.mark.parametrize("k", [0, -1])
def test_request_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(user_ids=np.array([1], dtype=np.uint32), k=k)


def test_request_rejects_invalid_user_ids() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(user_ids=np.array([1], dtype=np.int64))


def test_recommendation_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            item_ids=np.array([[10, 20], [30, 40]], dtype=np.uint32),
            scores=np.array([[1.0, 0.5]], dtype=np.float32),
        )


def test_target_batch_rejects_mismatched_users_and_targets() -> None:
    with pytest.raises(ValidationError):
        TargetBatch(
            user_ids=np.array([1, 2], dtype=np.uint32),
            relevant_items=[np.array([10], dtype=np.uint32)],
        )
