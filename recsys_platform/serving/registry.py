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
    """Registry of lazily importable recommender implementations.

    Model families are registered using import specifications rather than
    importing their implementations eagerly.

    Registry access is thread-safe.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._models: dict[str, type[Recommender]] = {}

        self._lock = RLock()

    def register(self, model_family: str, *, module: str, class_name: str) -> None:
        """Register a lazily importable model family.

        Registering the same family with the same specification is
        idempotent. Registering it with a different implementation is
        rejected.

        Args:
            model_family: Stable family identifier stored in manifests.
            module: Python module containing the recommender class.
            class_name: Name of the recommender class in that module.
        """
        spec = ModelSpec(module=module, class_name=class_name)

        with self._lock:
            self._specs[model_family] = spec

    def __getitem__(self, model_family: str) -> type[Recommender]:
        """Resolve and cache the recommender class for a model family."""
        with self._lock:
            if model_family in self._models:
                return self._models[model_family]

            spec = self._specs[model_family]
            module = import_module(spec.module)
            model_class = getattr(module, spec.class_name)

            self._models[model_family] = model_class

            return model_class

    def __iter__(self) -> Iterator[str]:
        """Iterate over all registered model families."""
        with self._lock:
            families = tuple(self._specs)

        return iter(families)

    def __len__(self) -> int:
        """Return the number of registered model families."""
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
