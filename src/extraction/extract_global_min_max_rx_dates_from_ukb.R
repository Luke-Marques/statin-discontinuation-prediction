## To initiate ukbkings container on CREATE, run:
## singularity run --bind /scratch/users/k1763489/Documents/Projects/RxRecords:/RxRecords,/scratch/prj/premandm/:/ukbiobank docker://onekenken/ukbkings:0.2.3
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
# GET MAXIMUM ISSUE DATE PER PARTICIPANT
# ------------------------------------------------------------------------------

# identify maximum issue date for each participant in prescription record dataset
min_max_issue_date_table <- rx_diskf |>
    select(f.eid, issue_date) |>
    collect() |>
    group_by(f.eid) |>
    summarise(
        min_issue_date = min(issue_date),
        max_issue_date = max(issue_date)
    )

# ------------------------------------------------------------------------------
# SAVE TO PARQUET FILE
# ------------------------------------------------------------------------------

# write rx_diskf to file
# NOTE: change path as needed
write_parquet(
    min_max_issue_date_table, 
    "/ukbiobank/usr/luke/rx_data/min_max_issue_date_table.parquet"
)