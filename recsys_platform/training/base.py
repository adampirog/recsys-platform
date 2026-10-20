from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from recsys_platform.data import RecommenderDataset
from recsys_platform.models import Recommender, TrainingMetadata


class TrainerConfig(BaseModel):
    """Configuration used to train a recommender."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate_json(handle.read())

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=2))


class Trainer[RecommenderType: Recommender, ConfigType: TrainerConfig](ABC):
    """Train a model-specific recommender."""

    def __init__(self, config: ConfigType) -> None:
        self.config = config

    @abstractmethod
    def fit(self, data: RecommenderDataset) -> RecommenderType:
        """Fit a recommender from the training dataset."""

    def get_metadata(self) -> TrainingMetadata:
        return TrainingMetadata(
            trainer=f"{type(self).__module__}.{type(self).__qualname__}",
            config=self.config.model_dump(mode="json"),
        )
