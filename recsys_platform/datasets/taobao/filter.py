"""Create warm- and cold-user evaluation subsets."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

import polars as pl


def print_summary(mixed: pl.LazyFrame, warm: pl.LazyFrame, cold: pl.LazyFrame) -> None:
    summary = pl.concat(
        [
            mixed.select(
                pl.lit("all").alias("split"),
                pl.len().alias("events"),
                pl.col("user_id").n_unique().alias("users"),
            ),
            warm.select(
                pl.lit("warm").alias("split"),
                pl.len().alias("events"),
                pl.col("user_id").n_unique().alias("users"),
            ),
            cold.select(
                pl.lit("cold").alias("split"),
                pl.len().alias("events"),
                pl.col("user_id").n_unique().alias("users"),
            ),
        ]
    ).collect(engine="streaming")

    print(summary)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        "train",
        type=Path,
        help="Path to the training Parquet dataset.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the mixed validation or test Parquet dataset.",
    )
    parser.add_argument(
        "--skip-summary",
        default=False,
        action="store_true",
        help="Skip the cohort summary.",
    )

    return parser.parse_args()


def main(args: Namespace) -> None:
    train = pl.scan_parquet(args.train)
    mixed = pl.scan_parquet(args.dataset)

    known_users = train.select("user_id").unique()

    warm = mixed.join(known_users, on="user_id", how="semi")
    cold = mixed.join(known_users, on="user_id", how="anti")

    root = args.dataset.parent
    name = args.dataset.stem
    warm.sink_parquet(root / f"{name}_warm.parquet", mkdir=True, engine="streaming")
    cold.sink_parquet(root / f"{name}_cold.parquet", mkdir=True, engine="streaming")

    print(f"Saved datasets to '{root}/'")

    if not args.skip_summary:
        print()
        print_summary(mixed, warm, cold)


if __name__ == "__main__":
    main(parse_args())
