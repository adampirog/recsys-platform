"""Create time-based train, validation, and test splits for Taobao interactions."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .preprocessing import TIMEZONE


MIN_ITEM_INTERACTIONS = 5

TRAIN_DATES = (
    datetime(2017, 11, 25, tzinfo=ZoneInfo(TIMEZONE)),
    datetime(2017, 12, 1, tzinfo=ZoneInfo(TIMEZONE)),
)

VALID_DATES = (
    datetime(2017, 12, 1, tzinfo=ZoneInfo(TIMEZONE)),
    datetime(2017, 12, 2, tzinfo=ZoneInfo(TIMEZONE)),
)

TEST_DATES = (
    datetime(2017, 12, 2, tzinfo=ZoneInfo(TIMEZONE)),
    datetime(2017, 12, 4, tzinfo=ZoneInfo(TIMEZONE)),
)


def get_valid_items(df: pl.LazyFrame) -> pl.DataFrame:
    """Return training items with at least ``MIN_ITEM_INTERACTIONS`` interactions."""
    return (
        df.filter(
            pl.col("event_time").is_between(
                *TRAIN_DATES,
                closed="left",
            )
        )
        .group_by("item_id")
        .len()
        .filter(pl.col("len") >= MIN_ITEM_INTERACTIONS)
        .select("item_id")
        .collect(engine="streaming")
    )


def get_subset(
    df: pl.LazyFrame, dates: tuple[datetime, datetime], valid_items: pl.DataFrame
) -> pl.LazyFrame:
    """Filter interactions to a date range and the supported item set."""
    return df.filter(
        pl.col("event_time").is_between(
            *dates,
            closed="left",
        )
    ).join(
        valid_items.lazy(),
        on="item_id",
        how="semi",
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("input", type=Path, help="Path to the cleaned Parquet dataset.")
    parser.add_argument("output", type=Path, help="Directory for the dataset splits.")

    return parser.parse_args()


def main(args: Namespace) -> None:
    df = pl.scan_parquet(args.input)

    valid_items = get_valid_items(df)

    train = get_subset(df, TRAIN_DATES, valid_items)
    valid = get_subset(df, VALID_DATES, valid_items)
    test = get_subset(df, TEST_DATES, valid_items)

    train.sink_parquet(args.output / "train.parquet", mkdir=True, engine="streaming")
    valid.sink_parquet(args.output / "valid.parquet", mkdir=True, engine="streaming")
    test.sink_parquet(args.output / "test.parquet", mkdir=True, engine="streaming")


def cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli()
