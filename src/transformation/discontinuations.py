from datetime import date, datetime

import polars as pl


def get_drugs_first_issue_date_for_eid(rx: pl.LazyFrame) -> pl.LazyFrame:
    """"""
    return rx.with_columns(
        first_issue_date=pl.col("issue_date").first().over("eid", "generic_name")
    )


def identify_interruption(
    rx: pl.LazyFrame,
    missed_rx_count: int = 4,
) -> pl.LazyFrame:
    """
    Identify and label instances of treatment interruption from prescription records.
    """

    # define polars expression for the date threshold to determine interruption
    date_threshold: pl.Expr = pl.col("issue_date") + (
        pl.col("expected_rx_duration") * missed_rx_count
    ).cast(pl.Duration("ms"))

    # define polars expression for interruption logic
    interrupt = (
        (pl.col("next_issue_date").is_not_null())
        & (pl.col("next_issue_date") > date_threshold)
        & (pl.col("issue_date") != pl.col("next_issue_date"))
        & (
            pl.col("first_issue_date")
            <= (pl.col("max_global_rx_issue_date") - pl.duration(days=365))
        )
        & (pl.col("next_issue_date").dt.date != pl.col("issue_date").dt.date)
    ).alias("interrupt")

    # apply polars expression
    rx = rx.sort("eid", "issue_date").with_columns(interrupt).drop("index")

    return rx


def identify_discontinuations(
    rx: pl.LazyFrame,
    max_global_rx_issue_date: date | datetime,
    missed_rx_count: int = 4,
) -> pl.LazyFrame:
    """Identify and label instances of discontinuations from prescription records."""

    # define polars expression for the date threshold to determine interruption
    date_threshold: pl.Expr = pl.col("issue_date") + (
        pl.col("expected_rx_duration") * missed_rx_count
    ).cast(pl.Duration("ms"))

    # define polars expression for discontinuation logic
    discontinue = (
        pl.col("next_issue_date").is_null()
        & (
            pl.col("date_of_death").is_null()
            | (pl.col("date_of_death") > date_threshold)
        )
        & (pl.col("max_global_rx_issue_date") >= date_threshold)
        & (pl.col("expected_rx_end_date") < max_global_rx_issue_date)
        & (
            pl.col("first_issue_date")
            >= (pl.col("min_global_rx_issue_date") + pl.duration(days=365))
        )
        & (
            pl.col("first_issue_date")
            <= (pl.col("max_global_rx_issue_date") - pl.duration(days=365))
        )
    ).alias("discontinue")

    # apply polars expression
    rx = rx.sort("eid", "issue_date").with_columns(discontinue)

    return rx


def count_discontinuations(rx: pl.LazyFrame) -> pl.LazyFrame:
    """Count the number of discontinuation events per `eid`."""
    rx = rx.with_columns(discontinuation_count=pl.col("discontinue").sum().over("eid"))

    return rx


def identify_restarts(rx: pl.LazyFrame) -> pl.LazyFrame:
    """
    Identify and label instances of individuals restarting treatment, following a
    discontinuation event.
    """
    rx = rx.with_columns(
        (pl.col("discontinue") & pl.col("next_issue_date").is_not_null()).alias(
            "restart"
        )
    )

    return rx


def discontinuation_pipeline(
    rx: pl.LazyFrame,
    max_global_rx_issue_date: date | datetime,
    missed_rx_count: int = 4,
) -> pl.LazyFrame:
    rx = (
        rx.pipe(get_drugs_first_issue_date_for_eid)
        .pipe(identify_discontinuations, max_global_rx_issue_date, missed_rx_count)
        .pipe(count_discontinuations)
        .pipe(identify_restarts)
    )
    return rx
