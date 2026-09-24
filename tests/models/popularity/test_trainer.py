import numpy as np
import numpy.testing as npt

from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.models.popularity.dataset import PopularityDataset
from recsys_platform.models.popularity.trainer import PopularityTrainer, PopularityTrainerConfig


def test_fit_returns_popularity_recommender(popularity_dataset: PopularityDataset) -> None:
    trainer = PopularityTrainer()

    model = trainer.fit(popularity_dataset)

    assert isinstance(model, PopularityRecommender)


def test_fit_ranks_items_by_interaction_count(popularity_dataset: PopularityDataset) -> None:
    model = PopularityTrainer().fit(popularity_dataset)

    npt.assert_array_equal(model.item_ids, np.array([10, 20, 30, 40], dtype=np.uint32))


def test_fit_normalizes_popularity_scores(popularity_dataset: PopularityDataset) -> None:
    model = PopularityTrainer().fit(popularity_dataset)

    npt.assert_allclose(model.scores, np.array([1.0, 0.6, 0.4, 0.4], dtype=np.float32))

    assert model.scores.dtype == np.float32


def test_fit_breaks_popularity_ties_by_item_id(popularity_dataset: PopularityDataset) -> None:
    model = PopularityTrainer().fit(popularity_dataset)

    # Items 30 and 40 both have two interactions.
    npt.assert_array_equal(model.item_ids[-2:], np.array([30, 40], dtype=np.uint32))


def test_evaluate_integrates_trainer_model_and_dataset(
    popularity_dataset: PopularityDataset,
) -> None:
    trainer = PopularityTrainer()
    model = trainer.fit(popularity_dataset)

    result = trainer.evaluate(model, popularity_dataset, k_values=(1,), batch_size=2)

    assert result[1]["precision"] == 1.0
    assert result[1]["hit_rate"] == 1.0
    assert result[1]["ndcg"] == 1.0

    # Top-1 is item 10, relevant for every user.
    #
    # Recall:
    # user 1 -> 1/3
    # user 2 -> 1/2
    # user 3 -> 1/2
    # user 4 -> 1/3
    #
    # Macro recall = 5/12.
    npt.assert_allclose(result[1]["recall"], 5 / 12)


def test_fit_respects_max_items(popularity_dataset: PopularityDataset) -> None:
    config = PopularityTrainerConfig(max_items=2)
    model = PopularityTrainer(config).fit(popularity_dataset)

    npt.assert_array_equal(model.item_ids, np.array([10, 20], dtype=np.uint32))
    npt.assert_allclose(model.scores, np.array([1.0, 0.6], dtype=np.float32))

    assert model.training_metadata is not None
    assert model.training_metadata.config == {"max_items": 2}
    assert model.training_metadata.trainer.endswith(".PopularityTrainer")
