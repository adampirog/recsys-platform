from pathlib import Path

import polars as pl


class RecommenderDataset:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @property
    def data(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path)
