import os

import numpy as np
import numpy.testing as npt
import pytest

from recsys_platform.data import RecommenderDataset
from recsys_platform.models.matrix_factorization import MatrixFactorizationRecommender
from recsys_platform.models.matrix_factorization.als_trainer import ALSTrainer, ALSTrainerConfig
from recsys_platform.models.matrix_factorization.bpr_trainer import BPRTrainer, BPRTrainerConfig
from recsys_platform.models.matrix_factorization.lmf_trainer import LMFTrainer, LMFTrainerConfig
from recsys_platform.models.matrix_factorization.trainer import MatrixFactorizationTrainerConfig


def test_build_user_item_matrix(dataset: RecommenderDataset) -> None:
    trainer = ALSTrainer()
    matrix, user_ids, item_ids = trainer._build_user_item_matrix(dataset)

    npt.assert_array_equal(user_ids, np.array([1, 2, 3, 4], dtype=np.uint32))
    npt.assert_array_equal(item_ids, np.array([10, 20, 30, 40], dtype=np.uint32))

    npt.assert_array_equal(
        matrix.toarray(),
        np.array(
            [
                [2, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 2],
            ],
            dtype=np.float32,
        ),
    )


def test_n_jobs_uses_scikit_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 8)

    assert MatrixFactorizationTrainerConfig(n_jobs=1).num_threads == 1
    assert MatrixFactorizationTrainerConfig(n_jobs=4).num_threads == 4
    assert MatrixFactorizationTrainerConfig(n_jobs=-1).num_threads == 0
    assert MatrixFactorizationTrainerConfig(n_jobs=-2).num_threads == 7


@pytest.mark.parametrize(
    ("trainer", "config"),
    [
        (
            ALSTrainer,
            ALSTrainerConfig(embedding_size=4, iterations=2, n_jobs=1, random_state=42),
        ),
        (
            BPRTrainer,
            BPRTrainerConfig(embedding_size=4, iterations=2, n_jobs=1, random_state=42),
        ),
        (
            LMFTrainer,
            LMFTrainerConfig(
                embedding_size=4, iterations=2, negative_sample_ratio=1, n_jobs=1, random_state=42
            ),
        ),
    ],
    ids=["als", "bpr", "lmf"],
)
def test_fit_returns_matrix_factorization_recommender(
    trainer, config, dataset: RecommenderDataset
) -> None:
    model = trainer(config).fit(dataset)

    assert isinstance(model, MatrixFactorizationRecommender)

    npt.assert_array_equal(model.user_ids, np.array([1, 2, 3, 4], dtype=np.uint32))
    npt.assert_array_equal(model.item_ids, np.array([10, 20, 30, 40], dtype=np.uint32))

    assert model.user_embeddings.shape[0] == 4
    assert model.item_embeddings.shape[0] == 4
    assert model.user_embeddings.shape[1] == model.item_embeddings.shape[1]

    assert model.user_embeddings.dtype == np.float32
    assert model.item_embeddings.dtype == np.float32

    assert np.isfinite(model.user_embeddings).all()
    assert np.isfinite(model.item_embeddings).all()

    assert model.training_metadata is not None
    assert model.training_metadata.config == config.model_dump(mode="json")
    assert model.training_metadata.trainer.endswith(f".{trainer.__name__}")


def test_fallback_predictions_use_item_popularity(dataset: RecommenderDataset) -> None:
    trainer = ALSTrainer()

    matrix, _, item_ids = trainer._build_user_item_matrix(dataset)

    predictions, scores = trainer._get_fallback_predictions(matrix, item_ids, max_items=3)

    npt.assert_array_equal(predictions, np.array([10, 20, 30], dtype=np.uint32))
    npt.assert_allclose(scores, np.array([1.0, 0.6, 0.4], dtype=np.float32))
