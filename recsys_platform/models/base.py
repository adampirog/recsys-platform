from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recsys_platform.types import Float32Matrix, UInt32Matrix, UInt32Vector
from recsys_platform.utils import generate_model_id

from .manifest import ModelManifest


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
    MODEL_FAMILY: ClassVar[str]
    ARTIFACT_VERSION: ClassVar[str]

    def __init__(self, *, model_id: str | None = None) -> None:
        if model_id is None:
            self.model_id = generate_model_id(model_family=type(self).MODEL_FAMILY)
        else:
            self.model_id = model_id

    @abstractmethod
    def predict(self, request: RecommendationRequest) -> Recommendation:
        """Generate recommendations for a batch of users.

        Returned rows must preserve the order of ``request.user_ids``.
        """

    @abstractmethod
    def _save(self, path: Path) -> None: ...

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to disk."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._save(path)

        ModelManifest(
            model_id=self.model_id,
            model_family=self.MODEL_FAMILY,
            artifact_version=self.ARTIFACT_VERSION,
        ).save(path / "MANIFEST.json")

    @classmethod
    @abstractmethod
    def _load(cls, path: Path, manifest: ModelManifest) -> Self: ...

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a serialized popularity model from disk."""
        path = Path(path)
        manifest = ModelManifest.load(path / "MANIFEST.json")

        if manifest.model_family != cls.MODEL_FAMILY:
            raise ValueError(
                f"Cannot load saved model with Manifest(model_family={manifest.model_family})"
                f" with class {cls.__name__}(model_family={cls.MODEL_FAMILY})."
            )

        model = cls._load(path, manifest=manifest)

        if model.model_id != manifest.model_id:
            raise RuntimeError("'model_id' was overridden during loading.")

        return model
