from argparse import Namespace
from pathlib import Path

from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.models.popularity.evaluate import main as evaluate_main
from recsys_platform.models.popularity.train import main as train_main
from recsys_platform.models.popularity.trainer import PopularityTrainerConfig


def test_train_script(popularity_data_path: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    model_registry = tmp_path / "model"

    PopularityTrainerConfig(max_items=2).save(config_path)
    args = Namespace(
        training_config=config_path,
        dataset=popularity_data_path,
        validation_dataset=None,
        output=model_registry,
    )

    train_main(args)

    models = list(model_registry.glob("*"))
    assert len(models) == 1

    model = PopularityRecommender.load(models[0])

    assert model.item_ids.size == 2
    assert model.training_metadata is not None
    assert model.training_metadata.config == {"max_items": 2}


def test_evaluate_script(
    popularity_model: PopularityRecommender, popularity_data_path: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "model"
    output_path = tmp_path / "evaluation.json"

    popularity_model.save(model_path)

    args = Namespace(
        model=model_path,
        dataset=popularity_data_path,
        output=output_path,
        k_values=(2,),
        batch_size=500,
    )
    evaluate_main(args)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8")
