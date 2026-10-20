from argparse import ArgumentParser, Namespace
from pathlib import Path

from recsys_platform.data import RecommenderDataset
from recsys_platform.evaluation.model_evaluation import evaluate_model
from recsys_platform.models import Recommender
from recsys_platform.training import Trainer, TrainerConfig


def parse_training_args(model_name: str) -> Namespace:
    parser = ArgumentParser(description=f"Train, save, and validate a {model_name} model.")

    parser.add_argument(
        "training_config",
        type=Path,
        metavar="CONFIG",
        help="JSON trainer configuration.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        metavar="DATASET",
        help="Training Parquet dataset.",
    )
    parser.add_argument(
        "output",
        type=Path,
        metavar="OUTPUT",
        help="Root directory for model artifacts.",
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        metavar="DATASET",
        help="Optional validation Parquet dataset.",
    )

    return parser.parse_args()


def run_training[RecommenderType: Recommender, ConfigType: TrainerConfig](
    args: Namespace,
    *,
    config_cls: type[ConfigType],
    trainer_cls: type[Trainer[RecommenderType, ConfigType]],
) -> None:
    config = config_cls.load(args.training_config)
    trainer = trainer_cls(config)

    data = RecommenderDataset(args.dataset)
    model = trainer.fit(data)

    artifact_path = args.output / model.model_id
    model.save(artifact_path)

    print(f"Saved model to {artifact_path}")

    if args.validation_dataset is not None:
        data = RecommenderDataset(args.validation_dataset)

        result = evaluate_model(model, data)
        print(result)

        result.save(artifact_path / "evaluation.txt")
