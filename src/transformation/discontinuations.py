from datetime import date, datetime
from pathlib import Path
from typing import Tuple, List

import polars as pl
from durations import calculate_rx_duration


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
    rx = rx.with_columns(discontinuation_count=pl.col("discontinued").sum().over("eid"))

    return rx


def identify_restarts(rx: pl.LazyFrame) -> pl.LazyFrame:
    """
    Identify and label instances of individuals restarting treatment, following a
    discontinuation event.
    """
    rx = rx.with_columns(
        restarted=(pl.col("discontinued") & pl.col("next_issue_date").is_not_null())
    )

    return rx


def generate_discontinuation_summary(
    discontinuations: pl.DataFrame,
    out_path: Path = Path("data") / "statins" / "discontinuation_summary.csv",
) -> Tuple[pl.DataFrame, Path]:
    return


def generate_sample_size_summary(
    discontinuations: pl.DataFrame | pl.LazyFrame,
    out_path: Path = Path("data") / "statins" / "discontinuation_sample_size.csv",
    drugs_to_include: str | List[str] | None = None,
    drugs_to_exclude: str | List[str] | None = None,
) -> Tuple[pl.DataFrame, Path]:
    """Generate a summary table CSV file of discontinuation group sample sizes."""

    if isinstance(discontinuations, pl.DataFrame):
        discontinuations = discontinuations.lazy()

    if drugs_to_include and isinstance(drugs_to_include, str):
        drugs_to_include = [drugs_to_include]
    if drugs_to_exclude and isinstance(drugs_to_exclude, str):
        drugs_to_exclude = [drugs_to_exclude]

    if drugs_to_include:
        discontinuations = discontinuations.filter(
            pl.col("generic_name").is_in(drugs_to_include)
        )
        out_path: Path = out_path.parent / str(
            str(out_path.stem) + "_" + "_".join(drugs_to_include) + ".csv"
        )
    elif drugs_to_exclude:
        drugs_to_include = [
            drug
            for drug in (
                discontinuations.select("generic_name")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
            if drug not in drugs_to_exclude
        ]
        discontinuations = discontinuations.filter(
            pl.col("generic_name").is_in(drugs_to_include)
        )
        out_path: Path = out_path.parent / str(
            str(out_path.stem) + "_not_" + "_".join(drugs_to_exclude) + ".csv"
        )

    summary = (
        discontinuations.sort("eid", "issue_date")
        .with_row_index()
        .with_columns(
            rx_count=(pl.col("index") - pl.col("index").first() + 1).over("eid"),
        )
        .select(
            [
                (
                    pl.col("eid")
                    .filter(
                        (pl.col("discontinued").sum().over("eid") == 1)
                        & (pl.col("restarted").sum().over("eid") == 0)
                    )
                    .unique()
                    .count()
                    .alias("Real Stoppers")
                ),
                (
                    pl.col("eid")
                    .filter(
                        (pl.col("discontinued").sum().over("eid") == 1)
                        & (pl.col("restarted").sum().over("eid") == 0)
                        & pl.col("discontinued")
                        & (
                            pl.col("issue_date") - pl.col("first_issue_date")
                            <= pl.duration(days=365)
                        )
                    )
                    .unique()
                    .count()
                    .alias("Early Real Stoppers (1st Year)")
                ),
            ]
            + [
                pl.col("eid")
                .filter(
                    (pl.col("discontinued").sum().over("eid") == 1)
                    & (pl.col("restarted").sum().over("eid") == 0)
                    & pl.col("discontinued")
                    & (pl.col("rx_count") == i)
                )
                .unique()
                .count()
                .alias(f"Early Real Stoppers ({i} Prior Rx)")
                for i in range(1, 6)
            ]
            + [
                (
                    pl.col("eid")
                    .filter(pl.col("discontinuation_count") > 1)
                    .unique()
                    .count()
                    .alias("Multiple Discontinuers")
                ),
                (
                    pl.col("eid")
                    .filter(pl.col("restarted"))
                    .unique()
                    .count()
                    .alias("Restarters")
                ),
                (
                    pl.col("eid")
                    .filter(pl.col("discontinued"))
                    .unique()
                    .count()
                    .alias("Total Discontinuers")
                ),
                pl.col("eid").unique().count().alias("Total Statin Users"),
            ]
        )
        .collect()
        .transpose(include_header=True, header_name="Group", column_names=["Count"])
        .with_columns(
            (pl.col("Count") / (pl.col("Count").tail(1)))
            .round(2)
            .alias("Count / Total Statin Users"),
            pl.when(pl.col("Group") != "Total Statin Users")
            .then((pl.col("Count") / (pl.col("Count").tail(2).head(1))).round(2))
            .otherwise(None)
            .alias("Count / Total Discontinuers"),
        )
    )

    print(f"Writing discontinuation sample size summary to {out_path}...", end=" ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(out_path)
    print("done.")

    return summary, out_path


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
