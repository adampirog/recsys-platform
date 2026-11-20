from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

from recsys_platform.types import Float32Matrix, UInt32Matrix, UInt32Vector
from recsys_platform.utils import generate_model_id

from .manifest import ModelManifest, TrainingMetadata


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    user_ids: UInt32Vector
    k: int = 20

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be greater than 0.")


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Ranked item IDs and model-specific scores aligned row-wise by user."""

    item_ids: UInt32Matrix
    scores: Float32Matrix


class Recommender(ABC):
    """Base interface for models with family-specific artifacts and a shared manifest."""

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
        """Return up to k ranked items per user.

        Rows follow request.user_ids. Item IDs and scores are aligned, with
        scores ordered from highest to lowest and interpreted per model.
        """

    @abstractmethod
    def _save(self, path: Path) -> None: ...

    def save(self, path: str | Path) -> None:
        """Save model artifacts and their manifest into a directory, creating it if needed."""

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
        """Load an artifact directory using a class matching its manifest's model family."""

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
