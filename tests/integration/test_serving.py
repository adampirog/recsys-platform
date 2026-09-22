from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient
from numpy import testing as npt

from recsys_platform.models.popularity.dataset import PopularityDataset
from recsys_platform.models.popularity.trainer import PopularityTrainer
from recsys_platform.serving.app import create_app
from recsys_platform.serving.manager import ModelManager
from recsys_platform.serving.registry import create_default_registry


def test_train_save_and_serve_popularity_model(tmp_path: Path) -> None:
    """Train a model artifact and serve it through the HTTP API."""
    data_path = tmp_path / "interactions.parquet"

    pl.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            "item_id": [10, 10, 20, 30, 10, 20, 10, 30, 10, 20, 40, 40],
        },
        schema={"user_id": pl.UInt32, "item_id": pl.UInt32},
    ).write_parquet(data_path)

    # Train.
    model = PopularityTrainer().fit(PopularityDataset(data_path))

    # Serialize a real serving artifact.
    model_path = tmp_path / "models"
    artifact_path = model_path / model.model_id
    model.save(artifact_path)

    # Start from serialized state only.
    manager = ModelManager(create_default_registry())
    manager.discover(model_path)

    assert manager.n_models == 1
    assert not manager.is_loaded(model.model_id)

    client = TestClient(create_app(manager=manager, model_path=model_path))

    # First request must lazily deserialize and serve the model.
    response = client.post(
        f"/v1/models/{model.model_id}/recommendations", json={"user_ids": [101, 102], "k": 3}
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["model_id"] == model.model_id
    assert payload["user_ids"] == [101, 102]
    assert payload["item_ids"] == [[10, 20, 30], [10, 20, 30]]

    npt.assert_array_almost_equal(payload["scores"], [[1.0, 0.6, 0.4], [1.0, 0.6, 0.4]])

    assert manager.is_loaded(model.model_id)

    # The same serialized artifact must remain usable after unloading.
    manager.unload(model.model_id)

    assert not manager.is_loaded(model.model_id)

    response = client.post(
        f"/v1/models/{model.model_id}/recommendations",
        json={"user_ids": [999], "k": 2},
    )

    assert response.status_code == 200
    assert response.json()["item_ids"] == [[10, 20]]
    assert manager.is_loaded(model.model_id)
