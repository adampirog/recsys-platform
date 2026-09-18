from pathlib import Path

import numpy as np
import pytest

from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.serving.manager import ModelManager
from recsys_platform.serving.registry import ModelRegistry, create_default_registry


@pytest.fixture(scope="session")
def model_path(tmpdir_factory) -> Path:
    root = Path(tmpdir_factory.mktemp("data")) / "models"

    PopularityRecommender(
        model_id="model-a",
        item_ids=np.array([10, 20, 30], dtype=np.uint32),
        scores=np.array([1.0, 0.5, 0.25], dtype=np.float32),
    ).save(root / "model-a")

    PopularityRecommender(
        model_id="model-b",
        item_ids=np.array([40, 50], dtype=np.uint32),
        scores=np.array([1.0, 0.5], dtype=np.float32),
    ).save(root / "model-b")

    return root


@pytest.fixture(scope="session")
def registry() -> ModelRegistry:
    return create_default_registry()


@pytest.fixture(scope="session")
def manager(model_path: Path, registry: ModelRegistry) -> ModelManager:
    manager = ModelManager(registry)
    manager.discover(model_path)

    return manager
