from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recsys_platform.types import Float32Matrix, UInt32Matrix, UInt32Vector


class RecommendationRequest(BaseModel):
    """Request top-k recommendations for a batch of users.

    Users are processed in the order provided. Implementations of
    ``Recommender.predict`` must preserve this order in the returned
    recommendation rows.

    Attributes:
        user_ids: One-dimensional array of user IDs.
        k: Maximum number of recommendations requested per user.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    user_ids: UInt32Vector
    k: Annotated[int, Field(gt=0, strict=True)] = 20


class Recommendation(BaseModel):
    """Ranked recommendations for a batch of users.

    Rows correspond positionally to users in the associated
    ``RecommendationRequest``. For row ``i``, ``item_ids[i]`` contains the
    ranked recommended item IDs and ``scores[i]`` contains their scores.

    Attributes:
        item_ids: Ranked item IDs with shape
            ``(n_users, n_recommendations)``.
        scores: Recommendation scores with the same shape as ``item_ids``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    item_ids: UInt32Matrix
    scores: Float32Matrix

    @model_validator(mode="after")
    def validate_shapes(self) -> Self:
        if self.item_ids.shape != self.scores.shape:
            raise ValueError("item_ids and scores must have identical shapes")

        return self


class Recommender(ABC):
    @abstractmethod
    def predict(self, request: RecommendationRequest) -> Recommendation:
        """Generate recommendations for a batch of users.

        Returned rows must preserve the order of ``request.user_ids``.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self: ...
