from pathlib import Path

import polars as pl
from utils import (
    Drug,
    check_col_contains,
    extract_text_from_col,
)

# --------------------------------------------------------------------------------------
# PART 1: LOAD DATA
# --------------------------------------------------------------------------------------

# define paths
project_dir = Path("../../").absolute()  # relative path (do not change)
data_dir = project_dir / "data"
ukb_project_dir = Path("/scratch/prj/premandm/")  # absolute path (change as needed)
ukb_user_dir = ukb_project_dir / "usr" / "luke"

# define list of substrings to exclude irrelevant records
drug_name_remove = [
    "nystatin",
    "ecostatin",
    "sandostatin",
    "ostoguard",
    "sharpsguard",
    "lactose powder",
    "guardian opaque",
    "testing",
    "ileobag",
]

# load ldl lowering drug prescription records as LazyFrame
ldl_file = ukb_user_dir / "rx_data" / "ldl_rx_records.parquet"
ldl = (
    pl.scan_parquet(ldl_file)
    .rename({"drug_name": "prescription_text", "quantity": "quantity_text"})
    # remove rows with irrelevant drugs/devices
    .filter(~check_col_contains("prescription_text", "|".join(drug_name_remove)))
)


