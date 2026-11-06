from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

from recsys_platform.types import Float32Matrix, UInt32Matrix, UInt32Vector
from recsys_platform.utils import generate_model_id

from .manifest import ModelManifest, TrainingMetadata


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    """Top-k recommendation query for an ordered batch of users."""

    user_ids: UInt32Vector
    k: int = 20

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be greater than 0.")


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Ranked item IDs and scores aligned row-wise with requested users."""

    item_ids: UInt32Matrix
    scores: Float32Matrix

    def __post_init__(self) -> None:
        if self.item_ids.shape != self.scores.shape:
            raise ValueError("item_ids and scores must have identical shapes.")


class Recommender(ABC):
    """Base interface for serializable recommendation models."""

    MODEL_FAMILY: ClassVar[str]
    ARTIFACT_VERSION: ClassVar[str]

    def __init__(
        self, *, model_id: str | None = None, training_metadata: TrainingMetadata | None = None
    ) -> None:
        if model_id is None:
            self.model_id = generate_model_id(model_family=type(self).MODEL_FAMILY)
        else:
            self.model_id = model_id

        self.training_metadata = training_metadata

    @abstractmethod
    def predict(self, request: RecommendationRequest) -> Recommendation:
        """
        Generate recommendations for a batch of users.

        Returned rows preserve the order of ``request.user_ids``.
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
            training_metadata=self.training_metadata,
            model_family=self.MODEL_FAMILY,
            artifact_version=self.ARTIFACT_VERSION,
        ).save(path / "MANIFEST.json")

    @classmethod
    @abstractmethod
    def _load(cls, path: Path, manifest: ModelManifest) -> Self: ...

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a serialized Recommender from disk."""

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
