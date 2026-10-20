from argparse import ArgumentParser, Namespace
from pathlib import Path

from recsys_platform.data import RecommenderDataset
from recsys_platform.evaluation.model_evaluation import evaluate_model
from recsys_platform.models import Recommender


def parse_evaluation_args(model_name: str) -> Namespace:
    parser = ArgumentParser(description=f"Evaluate a serialized {model_name} model")

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


def run_evaluation[RecommenderType: Recommender](
    args: Namespace, *, recommender_cls: type[RecommenderType]
) -> None:
    model = recommender_cls.load(args.model)
    data = RecommenderDataset(args.dataset)

    result = evaluate_model(
        model, data, k_values=args.k_values, batch_size=args.batch_size, verbose=args.verbose
    )

    print(result)

    if args.output is not None:
        result.save(args.output)
