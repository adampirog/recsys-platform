"""Clean and preprocess the Taobao UserBehavior dataset."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


TIMEZONE = "Asia/Shanghai"

START_TIME = datetime(2017, 11, 25, tzinfo=ZoneInfo(TIMEZONE))
END_TIME = datetime(2017, 12, 4, tzinfo=ZoneInfo(TIMEZONE))

EVENT_TYPES = {"pv": "view", "fav": "favorite", "cart": "cart", "buy": "purchase"}
EVENT_TYPE_DTYPE = pl.Enum(["view", "favorite", "cart", "purchase"])


def scan_dataset(path: str | Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        has_header=False,
        new_columns=[
            "user_id",
            "item_id",
            "category_id",
            "event_type",
            "timestamp",
        ],
        schema_overrides={
            "user_id": pl.UInt32,
            "item_id": pl.UInt32,
            "category_id": pl.UInt32,
            "event_type": pl.String,
            "timestamp": pl.Int64,
        },
    )


def clean_dataset(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.drop_nulls(
            [
                "user_id",
                "item_id",
                "category_id",
                "event_type",
                "timestamp",
            ]
        )
        .with_columns(
            pl.from_epoch(
                pl.col("timestamp"),
                time_unit="s",
            )
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(TIMEZONE)
            .alias("event_time"),
            pl.col("event_type")
            .replace_strict(EVENT_TYPES)
            .cast(EVENT_TYPE_DTYPE)
            .alias("event_type"),
        )
        .filter(
            pl.col("event_time").is_between(
                START_TIME,
                END_TIME,
                closed="left",
            )
        )
        .select(
            "user_id",
            "item_id",
            "category_id",
            "event_type",
            "event_time",
        )
    )


def dataset_overview(df: pl.LazyFrame) -> pl.DataFrame:
    return df.select(
        pl.len().alias("events"),
        pl.col("user_id").approx_n_unique().alias("approx_users"),
        pl.col("item_id").approx_n_unique().alias("approx_items"),
        pl.col("category_id").approx_n_unique().alias("approx_categories"),
        pl.col("event_time").dt.date().min().alias("first_event"),
        pl.col("event_time").dt.date().max().alias("last_event"),
    ).collect(engine="streaming")


def event_distribution(df: pl.LazyFrame) -> pl.DataFrame:
    return df.group_by("event_type").len().sort("len", descending=True).collect(engine="streaming")


def item_support(df: pl.LazyFrame) -> pl.DataFrame:
    item_counts = df.group_by("item_id").len()

    return item_counts.select(
        pl.len().alias("items"),
        (pl.col("len") >= 2).sum().alias("items_ge_2"),
        (pl.col("len") >= 5).sum().alias("items_ge_5"),
        (pl.col("len") >= 10).sum().alias("items_ge_10"),
        (pl.col("len") >= 20).sum().alias("items_ge_20"),
        (pl.col("len") >= 50).sum().alias("items_ge_50"),
    ).collect(engine="streaming")


def purchase_support(df: pl.LazyFrame) -> pl.DataFrame:
    purchase_counts = df.filter(pl.col("event_type") == "purchase").group_by("user_id").len()

    return purchase_counts.select(
        pl.len().alias("users_with_purchase"),
        (pl.col("len") >= 2).sum().alias("users_ge_2"),
        (pl.col("len") >= 3).sum().alias("users_ge_3"),
        (pl.col("len") >= 5).sum().alias("users_ge_5"),
        (pl.col("len") >= 10).sum().alias("users_ge_10"),
    ).collect(engine="streaming")


def user_activity_summary(df: pl.LazyFrame) -> pl.DataFrame:
    return (
        df.group_by("user_id")
        .agg(
            pl.len().alias("events"),
            pl.col("item_id").n_unique().alias("unique_items"),
            (pl.col("event_type") == "purchase").sum().alias("purchases"),
        )
        .select(
            pl.col("events").median().alias("events_median"),
            pl.col("events").mean().alias("events_mean"),
            pl.col("events").quantile(0.90).alias("events_p90"),
            pl.col("unique_items").median().alias("items_median"),
            pl.col("unique_items").quantile(0.90).alias("items_p90"),
            pl.col("purchases").median().alias("purchases_median"),
            pl.col("purchases").mean().alias("purchases_mean"),
            pl.col("purchases").quantile(0.90).alias("purchases_p90"),
        )
    ).collect(engine="streaming")


def item_activity_summary(df: pl.LazyFrame) -> pl.DataFrame:
    return (
        df.group_by("item_id")
        .agg(
            pl.len().alias("events"),
            pl.col("user_id").n_unique().alias("users"),
        )
        .select(
            pl.col("events").median().alias("events_median"),
            pl.col("events").mean().alias("events_mean"),
            pl.col("events").quantile(0.90).alias("events_p90"),
            pl.col("events").quantile(0.99).alias("events_p99"),
            pl.col("events").max().alias("events_max"),
        )
    ).collect(engine="streaming")


def daily_activity(df: pl.LazyFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("event_time").dt.date().alias("date"))
        .group_by(["date", "event_type"])
        .len()
        .sort(["date", "event_type"])
        .collect(engine="streaming")
    )


def print_summary(df: pl.LazyFrame) -> None:
    print("=== OVERVIEW ===")
    print(dataset_overview(df))

    print("\n=== EVENT TYPES ===")
    print(event_distribution(df))

    print("\n=== USER ACTIVITY ===")
    print(user_activity_summary(df))

    print("\n=== ITEM ACTIVITY ===")
    print(item_activity_summary(df))

    print("\n=== DAILY ACTIVITY ===")
    print(daily_activity(df))

    print("\n=== ITEM SUPPORT ===")
    print(item_support(df))

    print("\n=== PURCHASE SUPPORT ===")
    print(purchase_support(df))


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to the raw dataset.")
    parser.add_argument("output", type=Path, help="Path to a cleaned parquet dataset.")
    parser.add_argument(
        "--skip-eda",
        default=False,
        action="store_true",
        help="Skip the EDA and final dataset summary.",
    )

    return parser.parse_args()


def main(args: Namespace):
    df = scan_dataset(args.input)
    df = clean_dataset(df)
    df.sink_parquet(args.output, engine="streaming", mkdir=True)

    if not args.skip_eda:
        df = pl.scan_parquet(args.output)
        print_summary(df)


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
