from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from recsys_platform.models import ModelManifest, Recommender

from .registry import ModelRegistry


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_id: str
    model_family: str
    path: str
    loaded: bool


class ServableModel:
    """Wrap a serialized model artifact with thread-safe lazy loading."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.manifest = ModelManifest.load(self.path / "MANIFEST.json")

        self._model: Recommender | None = None
        self._lock = Lock()

    @property
    def model_id(self) -> str:
        return self.manifest.model_id

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def info(self) -> ModelInfo:
        with self._lock:
            loaded = self._model is not None

        return ModelInfo(
            model_id=self.manifest.model_id,
            model_family=self.manifest.model_family,
            path=str(self.path),
            loaded=loaded,
        )

    def load(self, model_class: type[Recommender]) -> Recommender:
        """Load once and return the cached instance on subsequent calls."""
        with self._lock:
            if self._model is None:
                self._model = model_class.load(self.path)

            return self._model

    def unload(self) -> None:
        """Release the cached model without removing its artifact."""
        with self._lock:
            self._model = None


class ModelManager:
    """Discover model artifacts and manage their lazy loading lifecycle."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

        self._models: dict[str, ServableModel] = {}
        self._lock = Lock()

    def discover(self, path: str | Path) -> tuple[str, ...]:
        """Recursively register previously unseen model artifacts by ID.

        Return sorted IDs of newly discovered models. Existing IDs are not replaced.
        """
        new = set()
        for manifest_path in Path(path).rglob("MANIFEST.json"):
            model = ServableModel(manifest_path.parent)

            with self._lock:
                if model.model_id in self._models:
                    continue

                self._models[model.model_id] = model

            new.add(model.model_id)

        return tuple(sorted(new))

    def load(self, model_id: str) -> Recommender:
        """Load and cache a discovered model by ID.

        An unknown ID raises KeyError; an unregistered model family raises RuntimeError.
        """
        with self._lock:
            servable = self._models[model_id]

        try:
            model_class = self.registry[servable.manifest.model_family]
        except KeyError as exc:
            raise RuntimeError(
                f"Model family {servable.manifest.model_family!r} is not registered."
            ) from exc

        return servable.load(model_class)

    def unload(self, model_id: str) -> None:
        """Release a cached model without unregistering it; unknown IDs are ignored."""
        with self._lock:
            model = self._models.get(model_id)

        if model is not None:
            model.unload()

    def is_loaded(self, model_id: str) -> bool:
        """Return False for unknown models as well as unloaded models."""
        with self._lock:
            model = self._models.get(model_id)

        return model.is_loaded if model is not None else False

    def available_models(self) -> tuple[ModelInfo, ...]:
        with self._lock:
            models = tuple(self._models.values())

        return tuple(model.info for model in models)

    @property
    def n_models(self) -> int:
        with self._lock:
            return len(self._models)

    @property
    def n_loaded(self) -> int:
        with self._lock:
            models = tuple(self._models.values())

        return sum(model.is_loaded for model in models)
