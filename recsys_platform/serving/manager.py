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
    """Manage the serving lifecycle of one serialized model artifact.

    A servable model is discovered from its manifest without loading the
    potentially expensive model state. The underlying recommender is loaded
    lazily on first use and cached until explicitly unloaded.

    Loading and unloading are synchronized so concurrent requests cannot
    initialize the same model more than once.

    Attributes:
        path: Directory containing the serialized model artifact.
        manifest: Metadata describing the serialized model.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.manifest = ModelManifest.load(self.path / "MANIFEST.json")

        self._model: Recommender | None = None
        self._lock = Lock()

    @property
    def model_id(self) -> str:
        """Return the stable identifier of the model artifact."""
        return self.manifest.model_id

    @property
    def is_loaded(self) -> bool:
        """Return whether the model state is currently loaded in memory."""
        with self._lock:
            return self._model is not None

    @property
    def info(self) -> ModelInfo:
        """Return serving metadata without loading the model."""
        with self._lock:
            loaded = self._model is not None

        return ModelInfo(
            model_id=self.manifest.model_id,
            model_family=self.manifest.model_family,
            path=str(self.path),
            loaded=loaded,
        )

    def load(self, model_class: type[Recommender]) -> Recommender:
        """Load and cache the model if it has not already been loaded.

        Concurrent callers are serialized so the artifact is initialized
        at most once.

        Args:
            model_class: Recommender implementation responsible for loading
                this model family.

        Returns:
            Loaded recommender instance.
        """
        with self._lock:
            if self._model is None:
                self._model = model_class.load(self.path)

            return self._model

    def unload(self) -> None:
        """Release the cached model while keeping the artifact discoverable."""
        with self._lock:
            self._model = None


class ModelManager:
    """Discover and manage lazily loaded recommendation models.

    The manager indexes artifacts by their stable ``model_id``. Discovery
    reads manifests only; model state is loaded on demand and cached by the
    corresponding ``ServableModel``.

    The model index is synchronized independently from individual model
    loading, allowing expensive artifact loading to occur without blocking
    unrelated manager operations.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

        self._models: dict[str, ServableModel] = {}
        self._lock = Lock()

    @property
    def n_models(self) -> int:
        return len(self._models)

    @property
    def n_loaded(self) -> int:
        return sum(model.is_loaded for model in self._models.values())

    def discover(self, path: str | Path) -> None:
        """Discover model artifacts recursively without loading model state.

        Repeated discovery overrides the old model.

        Args:
            path: Directory recursively searched for model manifests.
        """
        discovered: dict[str, ServableModel] = {}

        for manifest_path in Path(path).rglob("MANIFEST.json"):
            model = ServableModel(manifest_path.parent)
            discovered[model.model_id] = model

    def load(self, model_id: str) -> Recommender:
        """Return a model, loading and caching it on first access.

        Args:
            model_id: Stable ID stored in the model manifest.

        Returns:
            Loaded recommender instance.

        Raises:
            KeyError: If no artifact with the requested ID was discovered.
        """
        with self._lock:
            servable = self._models[model_id]

        model_class = self.registry[servable.manifest.model_family]

        return servable.load(model_class)

    def unload(self, model_id: str) -> None:
        """Unload model state while keeping the artifact registered."""
        with self._lock:
            model = self._models.get(model_id)

        if model is not None:
            model.unload()

    def is_loaded(self, model_id: str) -> bool:
        """Return whether the requested model is currently loaded."""
        with self._lock:
            model = self._models.get(model_id)

        return model.is_loaded if model is not None else False

    def available_models(self) -> tuple[ModelInfo, ...]:
        """Return metadata for all discovered model artifacts."""
        with self._lock:
            info = tuple(model.info for model in self._models.values())
        return info
