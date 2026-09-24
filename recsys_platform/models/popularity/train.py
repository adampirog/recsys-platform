from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

from recsys_platform.models.popularity.dataset import PopularityDataset
from recsys_platform.models.popularity.trainer import PopularityTrainer, PopularityTrainerConfig


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("training_config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-v", "--validation-dataset", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)

    return parser.parse_args()


def main(args: Namespace):
    config = PopularityTrainerConfig.load(args.training_config)
    trainer = PopularityTrainer(config)
    data = PopularityDataset(args.dataset)

    model = trainer.fit(data)
    model.save(args.output)

    if args.validation_dataset:
        data = PopularityDataset(args.dataset)
        result = trainer.evaluate(model, data)
        print(result)

        result.save(args.output / "evaluation.txt")


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
