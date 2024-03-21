## To initiate ukbkings container on CREATE, run:
## singularity run --bind /scratch/users/k1763489/Documents/Projects/StatinsRxRecords:/study,/scratch/prj/premandm/:/ukbiobank docker://onekenken/ukbkings:0.2.3
## Then, from within the contianer, run the following R code:

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------

# load packages
library(ukbkings)       # reading ukb datasets
library(arrow)          # writing/reading parquet files
library(dplyr)          # data manipulation
library(stringr)        # string manipulation
library(lubridate)      # date manipulation
library(purrr)
library(readr)
library(R.utils)

# enable max threads for disk.frame objects
library(disk.frame)
setup_disk.frame()

# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------

get_dob <- function(participant_id, demographic_df) {
    dob <- demographic_df |>
        filter(f.eid == participant_id) |>
        pull(date_of_birth) |>
        lubridate::ymd()
    return(dob)
}

# Function to read a text file into a vector, skipping the first line of the file
read_file_to_vector <- function(file_path) {
  read_lines(file_path)[2:n()]
}

# ------------------------------------------------------------------------------
# SPECIAL DATES
# ------------------------------------------------------------------------------

# define special dates which should map to missing values
missing_dates <- lubridate::ymd(c(
    "1900-01-01",
    "1901-01-01",
    "2037-07-07"
))

# define special dates which should map to participant's date of birth
dob_dates <- lubridate::ymd(c(
    "1902-02-02",
    "1903-03-03"
))

# ------------------------------------------------------------------------------
# LOAD AND CLEAN DATA
# ------------------------------------------------------------------------------

# read demograpgic data
demog_file <- "/ukbiobank/usr/luke/ukb_demog.parquet"
demog <- read_parquet(demog_file)

# read prescription data
ukb_project_dir <- "/ukbiobank/ukb23203_rga"
rx_diskf <- bio_record(ukb_project_dir, record = "gp_scripts") |>
    # reformat columns
    mutate(
        dmd_code = as.character(dmd_code),
        across(
            c(read_2, bnf_code, dmd_code, drug_name, quantity), 
            .fns = ~ na_if(.x, "")
        ),
        across(read_2:quantity, as.character),
        issue_date = lubridate::ymd(lubridate::dmy(issue_date))
    ) |>
    rename(f.eid = eid) |>
    mutate(
        issue_date = lubridate::as_date(ifelse(
            issue_date %in% dob_dates,
            get_dob(f.eid, demog),
            issue_date
        ))
    ) |>
    # remove records with missing issue dates
    filter(
        !is.na(issue_date),
        !(issue_date %in% missing_dates)
    )

# ------------------------------------------------------------------------------
# FILTER RX RECORDS
# ------------------------------------------------------------------------------

# define regex patterns for rx names/codes
bnf_pattern <- "^0212"  # bnf code prefix for lipid lowering agents
# read_pattern <- NA
name_pattern <- "/study/data/other_ldl_lowering_drugs/" |>
    list.files(pattern="\\.txt$", full.names=TRUE) |>
    map(read_csv, skip=1, col_names=FALSE) |>
    unlist() |>
    unique() |>
    str_sort() |>
    paste0(collapse="|")

# define regex pattern for terms to exclude from records
name_excludes <- c("strip", "test")
name_excludes_pattern <- paste0(name_excludes, collapse="|")

rx_ldl <- rx_diskf |>
    # clean bnf_code column by removing periods
    mutate(bnf_code_clean = str_replace_all(bnf_code, "\\.", "")) |>
    # filter prescription dataset for diabetes medications
    filter(
        str_detect(tolower(drug_name), name_pattern) |
        str_detect(bnf_code_clean, bnf_pattern) &
        # str_detect(read_2, read_pattern),
        !str_detect(drug_name, name_excludes_pattern)
    ) |>
    # drop trailing 00 on some read2 codes
    mutate(read_2 = str_replace(read_2, "00$", "")) |> 
    collect() |>
    as_tibble()

# ------------------------------------------------------------------------------
# SAVE FILTERED RX RECORDS TO PARQUET FILE
# ------------------------------------------------------------------------------

# write rx_dm to file
# NOTE: change path as needed
out_path = "/ukbiobank/usr/luke/rx_data/ldl_rx_records.parquet"
write_parquet(rx_ldl, out_path)

# create symlink to parquet file in study directory
link_path = "/study/data/ldl_lowering_drugs/ldl_raw.parquet"
createLink(
    link=link_path,
    target=out_path
)
