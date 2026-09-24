from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.models.popularity.dataset import PopularityDataset
from recsys_platform.models.popularity.trainer import PopularityTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-o", "--output", type=Path)

    return parser.parse_args()


def main(args: Namespace):
    trainer = PopularityTrainer()
    data = PopularityDataset(args.dataset)
    model = PopularityRecommender.load(args.model)

    result = trainer.evaluate(model, data)
    print(result)

    if args.output:
        result.save(args.output)


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
