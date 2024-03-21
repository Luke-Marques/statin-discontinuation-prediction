from datetime import date, datetime

import polars as pl
from discontinuations import identify_discontinuations


def identify_switches(
    rx: pl.LazyFrame,
    min_switch_from_rx: int = 2,
    max_switch_from_rx: int = 6,
    min_switch_to_rx: int = 1,
    max_global_rx_issue_date: date | datetime | None = None,
) -> pl.LazyFrame:
    """
    Identify and label instances of within-class switching from prescription records.
    """
    if "discontinued" not in rx.columns:
        rx = rx.pipe(identify_discontinuations, max_global_rx_issue_date)

    # define polars expressions
    prev_rx_count = (pl.col("index") - pl.col("index").first() + 1).over(
        "eid", "generic_name"
    )
    consecutive_rx_count = (pl.col("index").last() - pl.col("index")).over(
        "eid", "generic_name"
    )
    is_switch = (
        (pl.col("generic_name") != pl.col("generic_name").shift(-1)).over("eid")
        & (pl.col("discontinued") is not True)
        & (prev_rx_count >= min_switch_from_rx)
        & (prev_rx_count <= max_switch_from_rx)
        & (consecutive_rx_count.shift(-1) >= min_switch_to_rx)
        & (
            pl.col("first_issue_date")
            >= pl.col("min_global_rx_issue_date") + pl.duration(days=365)
        )
        & (
            pl.col("first_issue_date")
            <= pl.col("max_global_rx_issue_date") - pl.duration(days=365)
        )
    )
    switch_to_drug = (
        pl.when(is_switch)
        .then(pl.col("generic_name").shift(-1).over("eid"))
        .otherwise(None)
    )

    # apply polars expressions
    rx = (
        rx.with_row_index()
        .with_columns(
            is_switch=is_switch,
            switch_to_drug=switch_to_drug,
        )
        .drop("index")
    )

    return rx
