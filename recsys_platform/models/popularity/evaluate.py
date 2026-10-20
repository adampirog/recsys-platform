"""Evaluate a serialized popularity recommender."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from recsys_platform.data import RecommenderDataset
from recsys_platform.models.popularity import PopularityRecommender
from recsys_platform.models.popularity.trainer import PopularityTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "model",
        type=Path,
        metavar="MODEL",
        help="Model artifact directory.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        metavar="DATASET",
        help="Evaluation Parquet dataset.",
    )
    parser.add_argument(
        "-k",
        "--k-values",
        type=int,
        nargs="+",
        default=(5, 10, 20),
        metavar="K",
        help="Ranking cutoffs (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        metavar="N",
        help="Users per batch (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="PATH",
        help="Optional result file (.txt or .json).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Display progress bar.",
    )

    return parser.parse_args()


def main(args: Namespace) -> None:
    model = PopularityRecommender.load(args.model)
    data = RecommenderDataset(args.dataset)
    trainer = PopularityTrainer()

    result = trainer.evaluate(
        model, data, k_values=args.k_values, batch_size=args.batch_size, verbose=args.verbose
    )

    print(result)

    if args.output is not None:
        result.save(args.output)


def cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli()
