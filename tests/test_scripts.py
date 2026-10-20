from argparse import Namespace
from pathlib import Path

import numpy as np

from recsys_platform.cli import run_evaluation, run_training
from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.models.popularity.trainer import PopularityTrainer, PopularityTrainerConfig


def test_train_script(data_path: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    model_registry = tmp_path / "model"

    PopularityTrainerConfig(max_items=2).save(config_path)
    args = Namespace(
        training_config=config_path,
        dataset=data_path,
        validation_dataset=None,
        output=model_registry,
    )

    run_training(args, config_cls=PopularityTrainerConfig, trainer_cls=PopularityTrainer)
    models = list(model_registry.glob("*"))
    assert len(models) == 1

    model = PopularityRecommender.load(models[0])

    assert model.item_ids.size == 2
    assert model.training_metadata is not None
    assert model.training_metadata.config == {"max_items": 2}


def test_evaluate_script(data_path: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    output_path = tmp_path / "evaluation.json"

    PopularityRecommender(
        model_id="testing",
        item_ids=np.array([10, 20, 30, 40], dtype=np.uint32),
        scores=np.array([1.0, 0.6, 0.4, 0.4], dtype=np.float32),
    ).save(model_path)

    args = Namespace(
        model=model_path,
        dataset=data_path,
        output=output_path,
        k_values=(2,),
        batch_size=500,
        verbose=False,
    )
    run_evaluation(args, recommender_cls=PopularityRecommender)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8")
