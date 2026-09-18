from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.serving.app import create_app
from recsys_platform.serving.manager import ModelManager
from recsys_platform.serving.registry import create_default_registry


def test_health_and_model_listing(manager: ModelManager, model_path: Path) -> None:
    client = TestClient(create_app(manager=manager, model_path=model_path))

    health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["models_available"] == 2
    assert health.json()["models_loaded"] == 0

    response = client.get("/v1/models")

    assert response.status_code == 200

    models = {model["model_id"]: model for model in response.json()}

    assert set(models) == {"model-a", "model-b"}
    assert models["model-a"] == {
        "model_id": "model-a",
        "model_family": "popularity",
        "loaded": False,
    }


def test_recommendations_load_model_lazily(manager: ModelManager, model_path: Path) -> None:
    client = TestClient(create_app(manager=manager, model_path=model_path))

    assert not manager.is_loaded("model-a")

    response = client.post(
        "/v1/models/model-a/recommendations", json={"user_ids": [101, 102], "k": 2}
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "model-a",
        "user_ids": [101, 102],
        "item_ids": [
            [10, 20],
            [10, 20],
        ],
        "scores": [
            [1.0, 0.5],
            [1.0, 0.5],
        ],
    }

    assert manager.is_loaded("model-a")
    assert manager.n_loaded == 1


def test_recommendations_unknown_model_returns_404(manager: ModelManager, model_path: Path) -> None:
    client = TestClient(create_app(manager=manager, model_path=model_path))

    response = client.post(
        "/v1/models/unknown/recommendations",
        json={"user_ids": [1], "k": 5},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("payload", [{"user_ids": [], "k": 5}, {"user_ids": [1], "k": 0}])
def test_recommendations_reject_invalid_request(
    manager: ModelManager,
    model_path: Path,
    payload: dict[str, object],
) -> None:
    client = TestClient(create_app(manager=manager, model_path=model_path))

    response = client.post("/v1/models/model-a/recommendations", json=payload)

    assert response.status_code == 422


def test_admin_load_and_unload(
    manager: ModelManager,
    model_path: Path,
) -> None:
    client = TestClient(create_app(manager=manager, model_path=model_path))

    response = client.post("/admin/models/model-a/load")

    assert response.status_code == 200
    assert response.json() == {"model_id": "model-a", "loaded": True}
    assert manager.is_loaded("model-a")

    response = client.delete("/admin/models/model-a/load")

    assert response.status_code == 200
    assert response.json() == {"model_id": "model-a", "loaded": False}
    assert not manager.is_loaded("model-a")


@pytest.mark.parametrize(("method", "url"), [("post", "/admin/models/unknown/load")])
def test_admin_unknown_model_returns_404(
    manager: ModelManager, model_path: Path, method: str, url: str
) -> None:
    client = TestClient(create_app(manager=manager, model_path=model_path))

    response = client.request(method, url)

    assert response.status_code == 404


def test_refresh_discovers_new_models(tmp_path: Path) -> None:
    model_path = tmp_path / "models"
    model_path.mkdir()

    manager = ModelManager(create_default_registry())

    client = TestClient(create_app(manager=manager, model_path=model_path))

    health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["models_available"] == 0

    PopularityRecommender(
        model_id="new-model",
        item_ids=np.array([10, 20], dtype=np.uint32),
        scores=np.array([1.0, 0.5], dtype=np.float32),
    ).save(model_path / "new-model")

    response = client.post("/admin/models/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "added": ["new-model"],
    }

    assert manager.n_models == 1
    assert not manager.is_loaded("new-model")

    # Repeated refresh skips already discovered artifacts.
    response = client.post("/admin/models/refresh")

    assert response.status_code == 200
    assert response.json() == {"added": []}
