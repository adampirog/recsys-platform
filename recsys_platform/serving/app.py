"""HTTP serving application for recommendation models.

This module defines the FastAPI application used to expose serialized
recommender models over HTTP.

At startup, the serving process discovers model artifacts beneath the
configured model directory. Discovery reads model manifests only; model
state is loaded lazily by ``ModelManager`` when a model is requested for
inference.

The API is intentionally transport-focused. Model discovery, loading,
caching, and prediction remain delegated to the serving and model layers,
while this module is responsible for:

- HTTP request and response handling,
- conversion between JSON payloads and internal NumPy-based model types,
- API-level error handling,
- service health reporting,
- application startup configuration.

The application can be constructed directly with ``create_app`` for tests
or embedded usage, or started as a standalone server through ``main``.

Endpoints:
    GET /health:
        Report service health and model availability.

    GET /v1/models:
        List discovered recommendation models and their loading state.

    POST /v1/models/{model_id}/recommendations:
        Generate ranked recommendations using the requested model.

Example:
    Run the server against a directory containing serialized model
    artifacts::

        python -m recsys_platform.serving.app \\
            --model-path ./artifacts \\
            --host 0.0.0.0 \\
            --port 8000
"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from recsys_platform.models import RecommendationRequest as ModelRequest
from recsys_platform.version import __version__

from .manager import ModelManager
from .registry import create_default_registry
from .schema import HealthResponse, ModelResponse, RecommendationRequest, RecommendationResponse


API_VERSION = "0.1.0"


def create_app(manager: ModelManager) -> FastAPI:
    """Create the HTTP API backed by the provided model manager.

    Args:
        manager: Manager responsible for discovered model artifacts and
            lazy model loading.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="RecSys Platform", version=API_VERSION)

    @app.get("/health")
    def health() -> HealthResponse:
        """Report service health and model availability."""
        return HealthResponse(
            status="ok" if manager.n_models else "degraded",
            version=__version__,
            models_available=manager.n_models,
            models_loaded=manager.n_loaded,
        )

    @app.get("/v1/models")
    def models() -> list[ModelResponse]:
        """List models currently available to the serving process."""
        return [
            ModelResponse(
                model_id=model.model_id,
                model_family=model.model_family,
                loaded=model.loaded,
            )
            for model in manager.available_models()
        ]

    @app.post("/v1/models/{model_id}/recommendations")
    def recommend(model_id: str, request: RecommendationRequest) -> RecommendationResponse:
        """Generate recommendations using the requested model."""
        try:
            model = manager.load(model_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id!r} not found.",
            ) from exc

        user_ids = np.asarray(request.user_ids, dtype=np.uint32)

        recommendations = model.predict(ModelRequest(user_ids=user_ids, k=request.k))

        return RecommendationResponse(
            model_id=model_id,
            user_ids=request.user_ids,
            item_ids=recommendations.item_ids.tolist(),
            scores=recommendations.scores.tolist(),
        )

    return app


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        "model_path", type=Path, help="Root directory containing serialized model artifacts."
    )

    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")

    return parser.parse_args()


def main() -> None:
    """Discover available models and start the API server."""
    args = parse_args()

    registry = create_default_registry()

    manager = ModelManager(registry)
    manager.discover(args.model_path)

    app = create_app(manager)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
