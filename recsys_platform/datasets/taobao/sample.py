"""Create a reproducible user-sampled version of the cleaned Taobao dataset."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

import polars as pl

from .preprocessing import dataset_overview, user_activity_summary


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the cleaned Taobao Parquet dataset.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path for the sampled Parquet dataset.",
    )
    parser.add_argument(
        "users",
        type=float,
        metavar="N",
        help="Number (or fraction) of training users to sample.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random sampling seed (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-summary",
        default=False,
        action="store_true",
        help="Skip the final sampled-dataset summary.",
    )

    return parser.parse_args()


def get_users(df: pl.LazyFrame) -> pl.DataFrame:
    """Return all users observed in the dataset."""
    return df.select("user_id").unique().collect(engine="streaming")


def resolve_sample_size(n_users: int, sample_size: float) -> int:
    """Resolve an absolute or fractional user sample size."""
    if 0 < sample_size < 1:
        resolved = int(sample_size * n_users)
    else:
        resolved = int(sample_size)

    if resolved <= 0:
        raise ValueError("Sample size must select at least one user.")

    if resolved > n_users:
        raise ValueError(f"Requested {resolved:,} users, but only {n_users:,} users are available.")

    return resolved


def print_summary(df: pl.LazyFrame) -> None:
    """Print a short summary of the sampled dataset."""
    print("=== OVERVIEW ===")
    print(dataset_overview(df))

    print("\n=== USER ACTIVITY ===")
    print(user_activity_summary(df))


def main(args: Namespace) -> None:
    df = pl.scan_parquet(args.dataset)

    users = get_users(df)
    n_users = resolve_sample_size(users.height, args.users)
    sampled_users = users.sample(n=n_users, seed=args.random_state, shuffle=True)

    sampled = df.join(users.lazy(), on="user_id", how="semi")
    sampled.sink_parquet(args.output, mkdir=True, engine="streaming")

    print(f"Sampled {sampled_users.height:,} of {users.height:,} users.")
    print(f"Saved sampled dataset to {args.output}")

    if not args.skip_summary:
        print()
        print_summary(pl.scan_parquet(args.output))


def cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli()
