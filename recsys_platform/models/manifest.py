from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from recsys_platform.version import __version__


MANIFEST_VERSION = "0.2.0"


def package_version() -> str:
    return __version__


def manifest_version() -> str:
    return MANIFEST_VERSION


class TrainingMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trainer: str
    config: dict[str, JsonValue]


class ModelManifest(BaseModel):
    """Metadata required to identify and load a serialized model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_family: str
    training_metadata: TrainingMetadata | None = None
    artifact_version: str
    package_version: str = Field(default_factory=package_version)
    manifest_version: str = Field(default_factory=manifest_version)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate_json(handle.read())

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=2))
