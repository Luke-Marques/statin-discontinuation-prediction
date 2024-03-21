import argparse
from pathlib import Path

import polars as pl
from discontinuations import discontinuation_pipeline, generate_sample_size_summary
from dosage import dosage_pipeline
from durations import calculate_rx_end_date
from switches import identify_switches
from rx_summary import generate_eid_rx_summary


def select_columns(rx: pl.DataFrame) -> pl.DataFrame:
    return rx.select(
        [
            "eid",
            "generic_name",
            "brand_name",
            "manufacturer_info",
            "atc_class_code",
            "atc_class_name",
            "form",
            "strength_amt",
            "strength_unit",
            "quantity",
            "num_packs",
            "pack_size",
            "time_supply",
            "volume_prescribed",
            "quantity_per_day",
            "dosage_per_day",
            "discrete_dosage_per_day",
            "dosage_intensity",
            "issue_date",
            "expected_rx_duration",
            "expected_rx_end_date",
            "first_issue_date",
            "interrupt",
            "discontinue",
            "discontinuation_count",
            "restart",
            "is_switch",
            "switch_to_drug",
        ]
    )


def pipeline(
    rx: pl.LazyFrame,
    ukb_demographics: pl.LazyFrame,
    global_rx_dates: pl.LazyFrame,
    return_lazy: bool = True,
) -> pl.LazyFrame:
    """Apply all transformations to prescription records."""
    # get maximum issue date in global_rx_dates
    max_global_rx_issue_date = (
        global_rx_dates.select("max_global_rx_issue_date").max().collect().item()
    )

    rx = (
        rx.collect()
        .join(ukb_demographics.collect(), on="eid", how="left")
        .join(global_rx_dates.collect(), on="eid", how="left")
        .pipe(calculate_rx_end_date)
        .pipe(discontinuation_pipeline, max_global_rx_issue_date)
        .pipe(dosage_pipeline)
        .pipe(identify_switches)
        .pipe(select_columns)
        .sort("eid", "issue_date")
    )

    if not return_lazy:
        return rx.collect()

    return rx


def check_file_exists(parser: argparse.ArgumentParser, arg: str) -> None:
    if not Path(arg).exists():
        parser.error(f"File {arg} does not exist.")
    else:
        return Path(arg)


def main() -> None:
    """Script entry point function."""
    # create the parser
    parser = argparse.ArgumentParser(
        description="""Apply transformations to cleaned prescription records; duration, 
        discontinuation, dosage, and switching."""
    )

    # add the arguments
    parser.add_argument(
        "-r",
        "--rx-records",
        type=lambda x: check_file_exists(parser, x),
        default="data/statins/statins_clean.parquet",
    )
    parser.add_argument(
        "-d",
        "--demographics",
        type=lambda x: check_file_exists(parser, x),
        default="data/ukb_demographics.parquet",
    )
    parser.add_argument(
        "-g",
        "--global-rx-dates",
        type=lambda x: check_file_exists(parser, x),
        default="data/global_rx_dates.parquet",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="/scratch/prj/premandm/usr/luke/rx_data/statins/statins_processed.parquet",
    )

    # parse the arguments
    args = parser.parse_args()

    # load data
    print("Reading datasets...", end=" ")
    rx: pl.LazyFrame = pl.scan_parquet(args.rx_records).rename({"f.eid": "eid"})
    ukb_demographics: pl.LazyFrame = (
        pl.scan_parquet(args.demographics)
        .rename({"f.eid": "eid", "date_of_death_first_visit": "date_of_death"})
        .select("eid", "date_of_death")
    )
    global_rx_dates: pl.LazyFrame = pl.scan_parquet(args.global_rx_dates).rename(
        {
            "f.eid": "eid",
            "min_issue_date": "min_global_rx_issue_date",
            "max_issue_date": "max_global_rx_issue_date",
        }
    )
    print("done.")

    # apply transformations
    print("Applying transformation pipeline...", end=" ")
    rx_processed: pl.LazyFrame = pipeline(
        rx, ukb_demographics, global_rx_dates, return_lazy=False
    )
    print(rx_processed.columns)
    print("done.")

    # write processed rx dataframe to local parquet file
    print(
        "Writing transformed prescription records to",
        args.output,
        "...",
        end=" ",
    )
    rx_processed.write_parquet(args.output)
    print("done.")

    # show first rows of processed rx dataframe
    print("Transformed prescription records glimpse:")
    print(rx_processed)


if __name__ == "__main__":
    main()
