import polars as pl

from pathlib import Path


def generate_eid_rx_summary(
    rx: pl.LazyFrame,
    out_path: Path = Path("data") / "statins" / "rx_summary.parquet",
) -> pl.LazyFrame:
    """"""
    # transform prescription records to show continuous prescribing periods
    summary = (
        rx.sort("eid", "issue_date")
        .with_columns(
            period=(
                pl.col("discontinued")
                | pl.col("is_switch")
                | (
                    (pl.col("generic_name") != pl.col("generic_name").shift())
                    .over("eid")
                    .fill_null(False)
                )
            )
        )
        .with_columns(pl.col("period").cum_sum().over("eid"))
        .group_by("eid", "period")
        .agg(
            pl.col("generic_name").first(),
            pl.col("issue_date").min().alias("start_date"),
            pl.col("issue_date").max().alias("end_date"),
            pl.col("discontinued").last(),
            pl.col("is_switch").last(),
        )
        .sort("eid", "start_date")
        .select(
            "eid",
            "generic_name",
            "period",
            "start_date",
            "end_date",
            "discontinued",
            "is_switch",
        )
    )

    # write summary dataframe to local parquet file
    print(f"Writing Rx summary to {out_path}...", end=" ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_path)
    print("done.")

    return summary, out_path
