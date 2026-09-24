# Recommender System Platform

[![CI](https://github.com/adampirog/recsys-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/adampirog/recsys-platform/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/github/license/adampirog/recsys-platform)

A modular platform for training, evaluating, serializing, and serving recommender systems.

The platform is **model- and data-agnostic**: datasets, trainers, recommenders, evaluation, model artifacts, and serving are separated behind reusable interfaces.

_The current V1 uses a global popularity model as a simple baseline while the infrastructure is developed for more advanced personalized models._

## Highlights

- Generic `Recommender`, `Trainer`, and dataset interfaces
- CLI workflows for data preparation, training, evaluation, and serving
- Batched NumPy inference
- Recommendation metrics: Precision@K, Recall@K, HitRate@K, and NDCG@K
- Versioned model artifacts with training metadata
- FastAPI serving with lazy model loading
- Model-family registry and runtime model discovery
- Separate training and inference dependencies
- Dockerized inference
- Ruff, pytest, integration tests, and PR CI

## Development dataset

The platform has been developed against the **Taobao UserBehavior** dataset: a large-scale, real-world e-commerce interaction dataset containing roughly **100 million user-item events**. [(Original data source)](https://tianchi.aliyun.com/dataset/649)

It provides a realistic workload for developing and testing the platform while remaining separate from its core abstractions. The included Taobao integration is a reference implementation, not a requirement for using the platform.

## Installation

Requires **Python 3.13+**.

```bash
git clone https://github.com/adampirog/recsys-platform.git
cd recsys-platform
```

Install the components you need:

```bash
# Popularity training + evaluation
pip install -e ".[popularity-training]"

# Taobao data preparation
pip install -e ".[taobao-data]"

# Serving
pip install -e ".[serving,popularity-inference]"

# Everything
pip install -e ".[all]"
```

## Quick start

The included scripts cover the complete workflow from raw interactions to a served model.

### 1. Prepare the Taobao dataset

Clean the raw dataset and generate a short dataset summary:

```bash
python -m recsys_platform.datasets.taobao.preprocessing \
  data/UserBehavior.csv \
  data/taobao.parquet
```

Create temporal train, validation, and test splits:

```bash
python -m recsys_platform.datasets.taobao.split \
  data/taobao.parquet \
  data/splits
```

This produces:

```text
data/splits/
├── train.parquet
├── valid.parquet
└── test.parquet
```

### 2. Train a model

Training is configured through a JSON file:

```json
{
  "max_items": 100
}
```

Train and serialize the popularity model:

```bash
python -m recsys_platform.models.popularity.train \
  configs/popularity.json \
  data/splits/train.parquet \
  --validation-dataset data/splits/valid.parquet \
  --output artifacts/popularity
```

The resulting artifact contains the model state together with a manifest describing its identity, versions, and training configuration.

### 3. Evaluate a saved model

```bash
python -m recsys_platform.models.popularity.evaluate \
  artifacts/popularity \
  data/splits/test.parquet \
  --output artifacts/popularity/test.json
```

Evaluation results are also printed to the terminal.

### 4. Serve models

Start the API against a directory containing model artifacts:

```bash
python -m recsys_platform.serving.app ./artifacts
```

Models are discovered from their manifests and loaded lazily on first use.

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

Generate recommendations directly:

```bash
curl -X POST \
  http://localhost:8000/v1/models/<model-id>/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [101, 102], "k": 10}'
```

All scripts expose their available options through `--help`.

## Docker

Build and run the serving image:

```bash
docker build -t recsys-platform .

docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/artifacts:/models:ro" \
  recsys-platform
```

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

```bash
pip install -e ".[all]" --group dev

ruff check .
pytest
```

Pull requests to `master` run Python checks and a Docker serving smoke test through GitHub Actions.
