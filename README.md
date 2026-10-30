# Recommender System Platform

[![CI](https://github.com/adampirog/recsys-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/adampirog/recsys-platform/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/github/license/adampirog/recsys-platform)

An end-to-end platform for building, evaluating, serializing, and serving recommender systems.

The project separates dataset access, training, inference, evaluation, artifacts, and serving behind reusable interfaces. Model families can use different training stacks while producing lightweight inference artifacts that integrate with the same evaluation and serving infrastructure.

## Highlights

- Generic `Recommender`, `Trainer`, and Parquet-backed interaction dataset interfaces
- Global popularity and personalized matrix-factorization recommenders
- ALS, BPR, and Logistic Matrix Factorization for implicit feedback
- Cold-user fallback for personalized models
- Exact embedding-based top-k retrieval with memory-bounded batched inference
- Precision@K, Recall@K, HitRate@K, and NDCG@K evaluation
- Versioned model artifacts with training metadata
- CLI workflows for data preparation, training, evaluation, and serving
- FastAPI serving with lazy model loading and model-family discovery
- Separate training and inference dependency sets
- Dockerized inference
- Ruff, pytest, integration tests, and PR CI

## Architecture

The platform follows the same model lifecycle regardless of the training algorithm:

```text
interaction data
      │
      ▼
RecommenderDataset
      │
      ▼
    Trainer
      │
      ▼
model artifact
(manifest + model state)
      │
      ├──────────────► evaluation
      │
      ▼
  Recommender
      │
      ▼
 FastAPI serving
```

The main boundaries are intentionally simple:

- **Dataset** — exposes canonical `user_id` / `item_id` interaction data.
- **Trainer** — transforms training data into a model-specific inference artifact.
- **Recommender** — generates top-k recommendations from serialized model state.
- **Evaluation** — measures recommendation quality independently of the training algorithm.
- **Artifact** — stores model state together with version and training metadata.
- **Serving** — discovers recommender families from artifacts and loads them lazily.

Training-only libraries are kept outside the inference path. For example, matrix-factorization training uses SciPy and `implicit`, while the serialized recommender requires only the core NumPy inference stack.

## Implemented model families

| Model family         | Training algorithms           | Inference                                                              |
| -------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| Popularity           | Global interaction popularity | Precomputed global top-k ranking                                       |
| Matrix factorization | ALS, BPR, Logistic MF         | User/item embedding similarity with popularity fallback for cold users |

Matrix-factorization algorithms share the same inference artifact: external user and item IDs together with learned user/item embeddings. This allows different optimization objectives to be compared without changing the serving or evaluation interface.

## Development dataset

The platform has primarily been developed against the **Taobao UserBehavior** dataset, a large-scale e-commerce interaction dataset containing roughly **100 million user-item events**.

[Original Taobao UserBehavior dataset](https://tianchi.aliyun.com/dataset/649)

The Taobao integration is a reference dataset pipeline rather than a requirement of the platform. Core model and evaluation interfaces operate on canonical Parquet interaction datasets containing at least:

```text
user_id
item_id
```

The included Taobao tooling supports:

- raw-data cleaning and time normalization;
- deterministic user sampling for smaller local experiments;
- temporal train / validation / test splitting;
- training-supported item filtering;
- warm- and cold-user evaluation cohorts.

## Installation

Requires **Python 3.13+**.

```bash
git clone https://github.com/adampirog/recsys-platform.git
cd recsys-platform
```

Install only the components required by the workflow.

```bash
# Taobao data preparation
pip install -e ".[taobao-data]"

# Popularity training + evaluation
pip install -e ".[popularity-training]"

# Matrix-factorization training + evaluation
pip install -e ".[matrix-factorization-training]"

# Serving currently implemented model families
pip install -e ".[serving,popularity-inference,matrix-factorization-inference]"

# Complete development installation
pip install -e ".[all]" --group dev
```

## Quick start

The included CLIs cover the complete path from raw interactions to a serialized and served model.

### 1. Prepare the Taobao dataset

Clean the original dataset:

```bash
python -m recsys_platform.datasets.taobao.preprocessing \
  data/UserBehavior.csv \
  data/taobao.parquet
```

For faster local experimentation, optionally create a deterministic user sample while preserving complete histories for the selected users:

```bash
python -m recsys_platform.datasets.taobao.sample \
  data/taobao.parquet \
  data/taobao-small.parquet \
  --n-users 0.2 \
  --random-state 42
```

Create temporal train, validation, and test splits:

```bash
python -m recsys_platform.datasets.taobao.split \
  data/taobao-small.parquet \
  data/splits
```

This produces:

```text
data/splits/
├── train.parquet
├── valid.parquet
└── test.parquet
```

Items retained in the splits must have been interacted with by at least five distinct users during the training period. Validation and test therefore use a training-supported item universe.

The original validation and test files preserve the natural mixture of warm and cold users.

To materialize separate user cohorts:

```bash
python -m recsys_platform.datasets.taobao.filter \
  data/splits/train.parquet \
  data/splits/valid.parquet

python -m recsys_platform.datasets.taobao.filter \
  data/splits/train.parquet \
  data/splits/test.parquet
```

This additionally creates:

```text
data/splits/
├── valid_warm.parquet
├── valid_cold.parquet
├── test_warm.parquet
└── test_cold.parquet
```

A **warm user** appears in the final training dataset. A **cold user** does not.

This gives three useful evaluation views:

- `*_warm.parquet` — personalized-model quality;
- `*_cold.parquet` — cold-user fallback quality;
- original `valid.parquet` / `test.parquet` — realistic end-to-end system quality.

### 2. Train a recommender

Trainer configuration is stored as JSON and becomes part of the serialized model's training metadata.

#### Popularity baseline

Example configuration:

```json
{
  "max_items": 100
}
```

Train the model:

```bash
python -m recsys_platform.models.popularity.train \
  configs/popularity.json \
  data/splits/train.parquet \
  artifacts \
  --validation-dataset data/splits/valid.parquet
```

#### Matrix factorization

Example ALS configuration:

```json
{
  "embedding_size": 128,
  "regularization": 0.01,
  "alpha": 20.0,
  "iterations": 20,
  "fallback_items": 20,
  "random_state": 42,
  "n_jobs": -1
}
```

Train ALS:

```bash
python -m recsys_platform.models.matrix_factorization.train \
  configs/als.json \
  data/splits/train.parquet \
  artifacts \
  --algorithm ALS \
  --validation-dataset data/splits/valid_warm.parquet
```

The same entry point supports:

```text
--algorithm ALS
--algorithm BPR
--algorithm LMF
```

Each algorithm has its own trainer configuration while producing the same matrix-factorization inference artifact.

Training creates an artifact directory containing model state and a manifest describing the model family, artifact version, package version, model ID, trainer, and training configuration.

### 3. Evaluate a saved model

Evaluate a matrix-factorization artifact:

```bash
python -m recsys_platform.models.matrix_factorization.evaluate \
  artifacts/<model-id> \
  data/splits/test_warm.parquet \
  --k-values 5 10 20 \
  --output artifacts/<model-id>/test-warm.json
```

Evaluate the popularity baseline:

```bash
python -m recsys_platform.models.popularity.evaluate \
  artifacts/<model-id> \
  data/splits/test.parquet \
  --k-values 5 10 20
```

Available ranking metrics are:

- Precision@K
- Recall@K
- HitRate@K
- NDCG@K

Evaluation is macro-averaged across users.

### 4. Serve models

Start the inference API against a directory containing model artifacts:

```bash
python -m recsys_platform.serving.app ./artifacts
```

Artifacts are discovered from their manifests and model implementations are loaded lazily.

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

Generate recommendations:

```bash
curl -X POST \
  http://localhost:8000/v1/models/<model-id>/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [101, 102], "k": 10}'
```

All CLIs expose their available options through `--help`.

## Project structure

```text
recsys_platform/
├── cli/                  # shared training and evaluation CLI infrastructure
├── data/                 # canonical interaction datasets
├── datasets/
│   └── taobao/           # reference preprocessing and benchmark pipeline
├── evaluation/           # ranking metrics and model evaluation
├── models/
│   ├── popularity/
│   └── matrix_factorization/
├── serving/              # model discovery and FastAPI inference
└── training/             # generic trainer abstractions
```

Model-family packages own their training implementation and inference artifact while relying on shared platform infrastructure for datasets, metadata, evaluation, and serving.

## Docker

Build the inference image:

```bash
docker build -t recsys-platform .
```

Run it against local model artifacts:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/artifacts:/models:ro" \
  recsys-platform
```

The runtime image installs serving and inference dependencies only. Training libraries such as SciPy and `implicit` are excluded.

## Extending the platform

New model families should fit the existing lifecycle rather than introduce parallel infrastructure.

A typical model-family addition consists of:

```text
models/<family>/
├── model.py          # lightweight serialized inference implementation
├── trainer.py        # shared family-level training logic, if needed
├── *_trainer.py      # algorithm-specific training implementations
├── train.py          # CLI entry point
└── evaluate.py       # CLI entry point
```

The expected integration points are:

1. implement a `Recommender` inference artifact;
2. implement one or more `Trainer` classes;
3. define separate training and inference dependency extras;
4. register the model family for serving;
5. reuse the shared evaluation pipeline;
6. add focused unit and integration tests.

Model-specific training frameworks should remain outside the inference dependency path whenever possible.

## Roadmap

### Recommendation models

- [x] Global popularity baseline
- [x] Implicit-feedback matrix factorization
  - [x] Alternating Least Squares
  - [x] Bayesian Personalized Ranking
  - [x] Logistic Matrix Factorization
- [ ] Two-tower neural retrieval
- [ ] Learned ranking / reranking
- [ ] Multi-stage recommendation models

### Retrieval and inference

- [x] Exact embedding-based retrieval
- [x] Memory-bounded batched MF inference
- [x] Cold-user popularity fallback
- [ ] Approximate nearest-neighbor retrieval
- [ ] ONNX inference for neural models
- [ ] Candidate-generation / reranking pipelines

### Serving and operations

- [x] FastAPI inference API
- [x] Lazy model discovery and loading
- [x] Runtime artifact refresh
- [x] Explicit model load / unload
- [x] Dockerized serving
- [ ] Concurrent worker support
- [ ] Structured logging and metrics
- [ ] Prometheus / OpenTelemetry integration
- [ ] Readiness and liveness probes
- [ ] Model aliases and deployment stages
- [ ] Memory-aware model eviction
- [ ] A/B and shadow model routing

## Development

Install the complete development environment:

```bash
pip install -e ".[all]" --group dev
```

Run static checks and tests:

```bash
ruff check .
pytest
```

Pull requests to `master` run Python checks and a Docker serving smoke test through GitHub Actions.
