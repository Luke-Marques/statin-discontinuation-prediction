from typing import List, Dict, Tuple

import polars as pl


STATIN_DOSAGE_GUIDELINES = {
    "atorvastatin": (None, (10, 10), (20, 80)),
    "fluvastatin": ((20, 40), (80, 80), None),
    "lovastatin": ((20, 20), (40, 80), None),
    "pitavastatin": (None, (1, 4), None),
    "pravastatin": ((10, 40), None, None),
    "simvastatin": ((10, 10), (20, 40), (80, 80)),
}


def convert_strength_mcg_to_mg(rx: pl.LazyFrame) -> pl.LazyFrame:
    """
    Custom Polars function to convert microgram strength amounts/units
    to milligram amounts/units.
    """
    rx = rx.with_columns(
        pl.col("strength_unit").cast(str)
    )  # FIXME: remove cat from data cleaning - creates more issues than it's worth
    strength_amt_standardised = (
        pl.when(pl.col("strength_unit") == "mcg")
        .then(pl.col("strength_amt") / 1000)
        .otherwise(pl.col("strength_amt"))
    )
    strength_unit_standardised = (
        pl.when(pl.col("strength_unit") == "mcg")
        .then(pl.lit("mg"))
        .otherwise(pl.col("strength_unit"))
    )
    rx = rx.with_columns(
        strength_amt=strength_amt_standardised, strength_unit=strength_unit_standardised
    )

    return rx


def calculate_dosage(rx: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate the total prescribed dosage and dosage per day of prescriptions."""
    # number of milliseconds in a day
    # MS_PER_DAY = 86_400_000

    # standardise strength amounts/units
    rx = rx.pipe(convert_strength_mcg_to_mg)

    # polars expressions
    dosage = pl.col("num_packs") * pl.col("pack_size") * pl.col("strength_amt")
    # dosage_per_day = (
    #     pl.when(pl.col("next_issue_date").is_not_null())
    #     .then(
    #         dosage
    #         / (
    #             (
    #                 pl.col("next_issue_date") - pl.col("issue_date")
    #             ).dt.total_milliseconds()
    #             / MS_PER_DAY
    #         )
    #     )
    #     .otherwise(pl.col("expected_rx_duration").dt.total_milliseconds() / MS_PER_DAY)
    # )

    rx = rx.with_columns(
        dosage=dosage,
        dosage_unit=pl.col("strength_unit"),
        # dosage_per_day=dosage_per_day,
    )

    return rx


def smooth_dosage(
    rx: pl.LazyFrame, window_size: int = 3, use_median: bool = False
) -> pl.LazyFrame:
    """
    Apply a rolling average (mean or median) to the `dosage_per_day` column, grouped
    over `eid`s.
    """
    if "dosage_per_day" not in rx.columns:
        rx = rx.pipe(calculate_dosage)

    rx = rx.sort("eid", "issue_date")

    if use_median:
        rx = rx.with_columns(
            dosage_per_day_smoothed=pl.col("dosage_per_day")
            .rolling_median(window_size=window_size, center=True)
            .over("eid")
        )
    else:
        rx = rx.with_columns(
            dosage_per_day_smoothed=pl.col("dosage_per_day")
            .rolling_mean(window_size=window_size, center=True)
            .over("eid")
        )

    return rx


# def calculate_dosage_per_day(
#     rx: pl.LazyFrame, round_to_nearest: int | None = 5
# ) -> pl.LazyFrame:
#     rx = rx.with_columns(
#         dosage_per_day=(pl.col("strength_amt") * pl.col("quantity_per_day"))
#     )
#     if round_to_nearest:
#         rx = rx.with_columns(
#             dosage_per_day=(pl.col("dosage_per_day") / round_to_nearest)
#             .round()
#             .mul(round_to_nearest)
#         )
#     return rx


def calculate_volume_prescribed(rx: pl.LazyFrame) -> pl.LazyFrame:
    rx = rx.with_columns(
        pl.when(pl.col("time_supply").is_null())
        .then(pl.col("num_packs") * pl.col("pack_size") * pl.col("strength_amt"))
        .otherwise(None)
        .alias("volume_prescribed")
    )
    return rx


def calculate_quantity_per_day(rx: pl.LazyFrame, round: bool = False) -> pl.LazyFrame:
    if round:
        rx = rx.with_columns(
            quantity_per_day=pl.when(pl.col("time_supply").is_not_null())
            .then(None)
            .when(pl.col("next_issue_date").is_not_null() & ~pl.col("discontinued"))
            .then(
                (
                    pl.col("quantity")
                    / (pl.col("next_issue_date") - pl.col("issue_date")).dt.total_days()
                ).round()
            )
            .otherwise(
                (
                    pl.col("quantity") / pl.col("expected_rx_duration").dt.total_days()
                ).round()
            )
        )
    else:
        rx = rx.with_columns(
            quantity_per_day=pl.when(pl.col("time_supply").is_not_null())
            .then(None)
            .when(pl.col("next_issue_date").is_not_null() & ~pl.col("discontinued"))
            .then(
                (
                    pl.col("quantity")
                    / (pl.col("next_issue_date") - pl.col("issue_date")).dt.total_days()
                )
            )
            .otherwise(
                (pl.col("quantity") / pl.col("expected_rx_duration").dt.total_days())
            )
        )
    return rx


def calculate_dosage_per_day(rx: pl.LazyFrame) -> pl.LazyFrame:
    rx = rx.with_columns(
        pl.when(pl.col("time_supply").is_not_null())
        .then(None)
        .when(pl.col("next_issue_date").is_not_null() & ~pl.col("discontinued"))
        .then(
            (
                pl.col("volume_prescribed")
                / (pl.col("next_issue_date") - pl.col("issue_date")).dt.total_days()
            )
        )
        .otherwise(
            (pl.col("quantity") / pl.col("expected_rx_duration").dt.total_days())
        )
        .alias("dosage_per_day")
    )
    return rx


def discretise_dosage(
    rx: pl.LazyFrame,
    discrete_dosages: List[int] = [5, 10, 20, 40, 80],
    upper_limit: int | float = 120,
) -> pl.LazyFrame:
    """
    Maps floats stored in `dosage` column to their nearest value in the
    `discrete_dosages` list.
    """
    rx = (
        rx.with_columns(
            pl.concat_list(
                [
                    (pl.col("dosage_per_day") - v).abs().alias(f"{i}_discrete")
                    for i, v in enumerate(discrete_dosages)
                ]
            )
            .list.arg_min()
            .alias("min_diff_index"),
            pl.lit(discrete_dosages).alias("discrete_dosages"),
        )
        .with_columns(
            pl.when(
                (pl.col("dosage_per_day") > 0)
                & (pl.col("dosage_per_day") < upper_limit)
            )
            .then(pl.col("discrete_dosages").list.get(pl.col("min_diff_index")))
            .otherwise(None)
            .alias("discrete_dosage_per_day")
        )
        .drop("min_diff_index", "discrete_dosages")
    )
    return rx


def map_discrete_dosage_to_intensity(
    rx: pl.LazyFrame,
    intensity_ranges: Dict[str, Tuple[Tuple[int] | None]] = {
        "simvastatin": ((5, 10), (20, 40), (80)),
        "fluvastatin": ((20, 40), (80), None),
        "pravastatin": ((10, 20, 40), None, None),
        "atorvastatin": (None, (5, 10), (20, 40, 80)),
        "rosuvastatin": (None, (5), (10, 20, 40)),
    },
) -> pl.LazyFrame:
    # Create dataframe from intensity dictionary
    rows = []
    for generic_name, ranges in intensity_ranges.items():
        row = {"generic_name": generic_name}
        for i, intensity in enumerate(["low", "medium", "high"]):
            row[intensity] = ranges[i] if i < len(ranges) else None
        rows.append(row)
    intensity = pl.DataFrame(rows)

    # Join intensity dataframe to prescription records
    rx = rx.join(intensity, on="generic_name", how="left")

    # Map dosage values to intensity categories
    rx = rx.with_columns(
        pl.when(pl.col("low").list.contains(pl.col("discrete_dosage_per_day")))
        .then(1)
        .when(pl.col("medium").list.contains(pl.col("discrete_dosage_per_day")))
        .then(2)
        .when(pl.col("high").list.contains(pl.col("discrete_dosage_per_day")))
        .then(3)
        .otherwise(None)
        .alias("dosage_intensity")
    ).drop("low", "medium", "high")

    return rx


def dosage_pipeline(rx: pl.LazyFrame) -> pl.LazyFrame:
    rx = (
        rx.sort("eid", "issue_date")
        .pipe(convert_strength_mcg_to_mg)
        .pipe(calculate_volume_prescribed)
        .pipe(calculate_quantity_per_day)
        .pipe(calculate_dosage_per_day)
        .pipe(discretise_dosage)
        .pipe(map_discrete_dosage_to_intensity)
    )
    return rx
