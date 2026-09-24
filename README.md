# Recommender System Platform

[![CI](https://github.com/adampirog/recsys-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/adampirog/recsys-platform/actions/workflows/ci.yml)

A modular platform for training, evaluating, serializing, and serving recommender systems.

The platform is **model- and data-agnostic**: datasets, trainers, recommenders, evaluation, model artifacts, and serving are separated behind reusable interfaces.

_The current V1 uses a global popularity model as a simple baseline while the infrastructure is developed for more advanced personalized models._

## Highlights

- Generic `Recommender`, `Trainer`, and dataset interfaces
- Batched NumPy inference
- Recommendations metrics: Precision@K, Recall@K, HitRate@K, and NDCG@K
- Versioned model artifacts and manifests
- FastAPI serving with lazy model loading
- Model-family registry and runtime model discovery
- Separate training and inference dependencies
- Dockerized inference
- Ruff, pytest, integration tests, and PR CI

## Development dataset

The platform has been developed against the **Taobao UserBehavior** dataset: a large-scale, real-world e-commerce interaction dataset containing roughly 100 million user-item events. [(Original data source)](https://tianchi.aliyun.com/dataset/649)

It provides a useful environment for developing and testing recommender infrastructure at realistic scale, while remaining completely separate from the platform's core abstractions.

Included utilities provide:

- raw Taobao preprocessing
- event normalization
- time-based train/validation/test splits
- training-only item support filtering

The Taobao integration is a reference implementation, not a requirement for using the platform.

## Installation

Requires **Python 3.13+**.

```bash id="ao2qdt"
git clone https://github.com/adampirog/recsys-platform.git
cd recsys-platform
```

Install the specific components you need:

```bash id="in4rij"
# Popularity training + evaluation
pip install -e ".[popularity-training]"

# Serving
pip install -e ".[serving,popularity-inference]"

# Everything
pip install -e ".[all]"
```

## Quick start -- base model

```python id="5v8xgi"
from pathlib import Path

from recsys_platform.models.popularity.dataset import PopularityDataset
from recsys_platform.models.popularity.trainer import PopularityTrainer


train = PopularityDataset("data/train.parquet")
valid = PopularityDataset("data/valid.parquet")

trainer = PopularityTrainer()
model = trainer.fit(train)

print(trainer.evaluate(model, valid, k_values=(5, 10, 20)))

model.save(Path("artifacts") / model.model_id)
```

## Serving

Start the API using a directory containing serialized model artifacts:

```bash id="dnyf8b"
python -m recsys_platform.serving.app ./artifacts
```

Generate recommendations:

```bash id="bjysal"
curl -X POST \
  http://localhost:8000/v1/models/<model-id>/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [101, 102], "k": 10}'
```

Models are discovered from their manifests and loaded lazily on first use.

Available operations include model listing, recommendation inference, artifact refresh, and explicit model loading/unloading.

### Docker

```bash id="ybe3d8"
docker build -t recsys-platform .

docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/artifacts:/models:ro" \
  recsys-platform
```

Interactive API documentation is available at: `http://localhost:8000/docs` (Swagger UI)

_The runtime image contains only serving and inference dependencies._

## Roadmap

### Models

- [x] Global popularity baseline
- [ ] Collaborative filtering / matrix factorization
- [ ] Two-tower neural retrieval
- [ ] Approximate nearest-neighbor retrieval
- [ ] ONNX inference for neural models
- [ ] Learned ranking / reranking
- [ ] Multi-stage retrieval and ranking

### Serving

- [x] FastAPI inference API
- [x] Lazy model discovery and loading
- [x] Runtime artifact refresh
- [x] Explicit model load / unload
- [x] Dockerized serving
- [ ] Concurrent workers support
- [ ] Structured logging and metrics
- [ ] Prometheus / OpenTelemetry integration
- [ ] Readiness and liveness probes
- [ ] Model aliases and deployment stages
- [ ] Memory-aware model eviction
- [ ] A/B and shadow model routing

## Development

```bash id="fp365y"
pip install -e ".[all]" --group dev

ruff check .
pytest
```

Pull requests to `master` run Python checks and a Docker serving smoke test through GitHub Actions.

## License

MIT
