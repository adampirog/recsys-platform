from pathlib import Path

import pytest

from recsys_platform.serving.manager import ModelManager
from recsys_platform.serving.registry import ModelRegistry


def test_discover_indexes_models_without_loading(model_path: Path, registry: ModelRegistry) -> None:
    manager = ModelManager(registry)

    added = manager.discover(model_path)

    assert set(added) == {"model-a", "model-b"}
    assert manager.n_models == 2
    assert manager.n_loaded == 0

    models = {model.model_id: model for model in manager.available_models()}

    assert set(models) == {"model-a", "model-b"}

    assert models["model-a"].model_family == "popularity"
    assert models["model-a"].loaded is False
    assert Path(models["model-a"].path).name == "model-a"


def test_discover_skips_already_known_models(manager: ModelManager, model_path: Path) -> None:
    added = manager.discover(model_path)

    assert added == ()
    assert manager.n_models == 2


def test_load_caches_and_unload_releases_model(manager: ModelManager) -> None:
    first = manager.load("model-a")

    assert first.model_id == "model-a"
    assert manager.is_loaded("model-a")
    assert manager.n_loaded == 1

    second = manager.load("model-a")

    assert second is first
    assert manager.n_loaded == 1

    manager.unload("model-a")

    assert not manager.is_loaded("model-a")
    assert manager.n_loaded == 0

    reloaded = manager.load("model-a")

    assert reloaded.model_id == "model-a"
    assert reloaded is not first
    assert manager.is_loaded("model-a")


def test_load_unknown_model_raises(manager: ModelManager) -> None:
    with pytest.raises(KeyError):
        manager.load("unknown")
