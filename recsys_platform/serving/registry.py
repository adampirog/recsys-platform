from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module
from threading import RLock

from recsys_platform.models import Recommender


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Import specification for a recommender implementation."""

    module: str
    class_name: str


class ModelRegistry(Mapping[str, type[Recommender]]):
    """Thread-safe registry of lazily imported recommender classes."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._model_types: dict[str, type[Recommender]] = {}

        self._lock = RLock()

    def register(self, model_family: str, *, module: str, class_name: str) -> None:
        """
        Register or replace a recommender implementation for a model family.

        Args:
            model_family: Family identifier stored in model manifests.
            module: Module containing the recommender class.
            class_name: Recommender class name.
        """
        spec = ModelSpec(module=module, class_name=class_name)

        with self._lock:
            self._specs[model_family] = spec
            self._model_types.pop(model_family, None)

    def __getitem__(self, model_family: str) -> type[Recommender]:
        """Resolve and cache the recommender class for a model family."""
        with self._lock:
            if model_family in self._model_types:
                return self._model_types[model_family]

            spec = self._specs[model_family]
            module = import_module(spec.module)
            model_class = getattr(module, spec.class_name)

            self._model_types[model_family] = model_class

            return model_class

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            families = tuple(self._specs)

        return iter(families)

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)


def create_default_registry() -> ModelRegistry:
    """Create a registry containing all built-in recommender families."""

    registry = ModelRegistry()

    registry.register(
        "popularity",
        module="recsys_platform.models.popularity.model",
        class_name="PopularityRecommender",
    )

    return registry
