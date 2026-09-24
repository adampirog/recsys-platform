"""Train, save, and optionally validate a popularity recommender."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from recsys_platform.models.popularity.dataset import PopularityDataset
from recsys_platform.models.popularity.trainer import (
    PopularityTrainer,
    PopularityTrainerConfig,
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)

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


def main(args: Namespace) -> None:
    config = PopularityTrainerConfig.load(args.training_config)
    trainer = PopularityTrainer(config)

    data = PopularityDataset(args.dataset)
    model = trainer.fit(data)

    artifact_path = args.output / model.model_id
    model.save(artifact_path)

    print(f"Saved model to {artifact_path}")

    if args.validation_dataset is not None:
        data = PopularityDataset(args.validation_dataset)

        result = trainer.evaluate(model, data)
        print(result)

        result.save(artifact_path / "evaluation.txt")


def cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli()
