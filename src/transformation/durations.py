import polars as pl


def get_issue_date_diff(rx: pl.LazyFrame) -> pl.LazyFrame:
    """
    Custom Polars function to get the time diferrence between consecutive prescription
    records. Returns the input dataframe with the additional columns `next_issue_date`
    and `issue_date_diff`.
    """
    rx = rx.sort("eid", "issue_date").with_columns(
        next_issue_date=pl.col("issue_date").shift(-1).over("eid"),
        issue_date_diff=(pl.col("issue_date").shift(-1) - pl.col("issue_date")).over(
            "eid"
        ),
    )

    return rx


def calculate_median_rx_issue_date_difference_per_eid(
    rx: pl.LazyFrame,
    date_diff_cutoff: int | pl.Duration = pl.duration(days=182),
) -> pl.LazyFrame:
    """
    Custom Polars function to calculate the median time difference between consecutive
    prescriptions of similar drugs, accounting for drug name, form, strength, and
    quantity prescribed. Optionally, can compute the mean instead of the median. Returns
    the input dataframe with an additional `issue_date_diff_median_per_eid` column.
    """
    if isinstance(date_diff_cutoff, int):
        date_diff_cutoff = pl.duration(days=date_diff_cutoff)

    if "issue_date_diff" not in rx.columns:
        rx = rx.pipe(get_issue_date_diff)

    group_by_cols = [
        "eid",
        "generic_name",
        "form",
        "strength_amt",
        "strength_unit",
        "quantity",
    ]

    rx = (
        rx.sort("eid", "issue_date")
        .group_by(group_by_cols)
        .agg(
            "*",
            median_issue_date_diff_per_eid=pl.col("issue_date_diff")
            .filter(
                pl.col("next_issue_date").is_not_null()
                & (pl.col("issue_date_diff") < date_diff_cutoff)
            )
            .median(),
        )
        .explode(pl.exclude(group_by_cols, "median_issue_date_diff_per_eid"))
    )

    return rx


def calculate_mean_rx_issue_date_difference(
    rx: pl.LazyFrame,
    date_diff_cutoff: int | pl.Duration = pl.duration(days=182),
) -> pl.LazyFrame:
    """
    Custom Polars function to calculate the mean, across all `eid`s, of the median time
    difference between consecutive prescriptions per `eid`.
    """
    if "median_issue_date_diff_per_eid" not in rx.columns:
        rx = calculate_median_rx_issue_date_difference_per_eid(rx, date_diff_cutoff)

    group_by_cols = [
        "generic_name",
        "form",
        "strength_amt",
        "strength_unit",
        "quantity",
    ]

    rx = (
        rx.group_by(group_by_cols)
        .agg(
            "*",
            mean_issue_date_diff=pl.col("median_issue_date_diff_per_eid").mean(),
        )
        .explode(pl.exclude(group_by_cols, "mean_issue_date_diff"))
        .drop("median_issue_date_diff_per_eid")
    )

    return rx


def calculate_rx_duration(
    rx: pl.LazyFrame,
    date_diff_cutoff: int | pl.Duration = pl.duration(days=182),
) -> pl.LazyFrame:
    """
    Calculate the expected duration of prescriptions using either the `time_supply`
    column, or the mean time difference between consecutive prescriptions for each drug.
    """
    if isinstance(date_diff_cutoff, int):
        date_diff_cutoff: pl.Duration = pl.duration(days=date_diff_cutoff)

    if "mean_issue_date_diff" not in rx.columns:
        rx = rx.pipe(calculate_mean_rx_issue_date_difference, date_diff_cutoff)

    rx = rx.with_columns(
        expected_rx_duration=pl.when(pl.col("time_supply").is_not_null())
        .then(pl.col("time_supply"))
        .otherwise(pl.col("mean_issue_date_diff"))
    )

    return rx


def calculate_rx_end_date(
    rx: pl.LazyFrame, date_diff_cutoff: int | pl.Duration = pl.duration(days=182)
) -> pl.LazyFrame:
    """Calculate the expected end date of prescriptions."""
    if "expected_rx_duration" not in rx.columns:
        rx = calculate_rx_duration(rx, date_diff_cutoff)

    rx = rx.with_columns(
        expected_rx_end_date=(pl.col("issue_date") + pl.col("expected_rx_duration"))
    )

    return rx
