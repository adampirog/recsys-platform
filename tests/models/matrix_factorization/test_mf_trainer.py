import numpy as np
import polars as pl
from numpy import testing as npt

from recsys_platform.data import RecommenderDataset
from recsys_platform.models.matrix_factorization import MatrixFactorizationRecommender
from recsys_platform.models.matrix_factorization.als_trainer import ALSTrainer, ALSTrainerConfig
from recsys_platform.models.matrix_factorization.bpr_trainer import BPRTrainer, BPRTrainerConfig
from recsys_platform.models.matrix_factorization.lmf_trainer import LMFTrainer, LMFTrainerConfig
from recsys_platform.models.matrix_factorization.trainer import (
    MatrixFactorizationTrainer,
    MatrixFactorizationTrainerConfig,
)


class TestTrainer(MatrixFactorizationTrainer[MatrixFactorizationTrainerConfig]):
    def fit(self, data: RecommenderDataset) -> MatrixFactorizationRecommender:
        raise NotImplementedError


def test_build_user_item_matrix_aggregates_interactions(tmp_path) -> None:
    path = tmp_path / "interactions.parquet"

    pl.DataFrame({"user_id": [20, 10, 10, 20], "item_id": [200, 100, 100, 100]}).write_parquet(path)

    trainer = TestTrainer(MatrixFactorizationTrainerConfig())
    matrix, user_ids, item_ids = trainer._build_user_item_matrix(RecommenderDataset(path))

    npt.assert_array_equal(user_ids, np.array([10, 20], dtype=np.uint32))
    npt.assert_array_equal(item_ids, np.array([100, 200], dtype=np.uint32))

    npt.assert_array_equal(matrix.toarray(), np.array([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32))


def test_als_fit_builds_recommender(monkeypatch, dataset) -> None:
    captured = {}

    class FakeALS:
        user_factors = np.ones((2, 3), dtype=np.float32)
        item_factors = np.ones((3, 3), dtype=np.float32)

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, user_items):
            assert user_items.shape == (2, 3)

    monkeypatch.setattr(
        "recsys_platform.models.matrix_factorization.als_trainer.AlternatingLeastSquares", FakeALS
    )

    trainer = ALSTrainer(
        ALSTrainerConfig(embedding_size=3, alpha=10.0, iterations=5, n_jobs=4, random_state=42)
    )

    model = trainer.fit(dataset)

    assert isinstance(model, MatrixFactorizationRecommender)
    assert captured["factors"] == 3
    assert captured["alpha"] == 10.0
    assert captured["iterations"] == 5
    assert captured["num_threads"] == 4

    assert model.user_embeddings.shape == (2, 3)
    assert model.item_embeddings.shape == (3, 3)


def test_bpr_fit_builds_recommender(monkeypatch, dataset) -> None:
    captured = {}

    class FakeBPR:
        user_factors = np.ones((2, 4), dtype=np.float32)
        item_factors = np.ones((3, 4), dtype=np.float32)

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, user_items):
            assert user_items.shape == (2, 3)

    monkeypatch.setattr(
        "recsys_platform.models.matrix_factorization.bpr_trainer.BayesianPersonalizedRanking",
        FakeBPR,
    )

    trainer = BPRTrainer(
        BPRTrainerConfig(embedding_size=3, learning_rate=0.05, n_jobs=2, random_state=42)
    )

    model = trainer.fit(dataset)

    assert isinstance(model, MatrixFactorizationRecommender)
    assert captured["factors"] == 3
    assert captured["learning_rate"] == 0.05
    assert captured["num_threads"] == 2

    npt.assert_array_equal(model.user_ids, np.array([10, 20], dtype=np.uint32))


def test_lmf_fit_builds_recommender(monkeypatch, dataset) -> None:
    captured = {}

    class FakeLMF:
        user_factors = np.ones((2, 3), dtype=np.float32)
        item_factors = np.ones((3, 3), dtype=np.float32)

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, user_items):
            assert user_items.shape == (2, 3)

    monkeypatch.setattr(
        "recsys_platform.models.matrix_factorization.lmf_trainer.LogisticMatrixFactorization",
        FakeLMF,
    )

    trainer = LMFTrainer(
        LMFTrainerConfig(embedding_size=3, negative_sample_ratio=7, n_jobs=-1, random_state=42)
    )

    model = trainer.fit(dataset)

    assert isinstance(model, MatrixFactorizationRecommender)
    assert captured["factors"] == 3
    assert captured["neg_prop"] == 7
    assert captured["num_threads"] == 0
