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

# load statin prescription records as LazyFrame
statins_file = ukb_user_dir / "rx_data" / "statins" / "statins_raw.parquet"
statins = (
    pl.scan_parquet(statins_file)
    .rename({"drug_name": "prescription_text", "quantity": "quantity_text"})
    # remove rows with irrelevant drugs/devices
    .filter(~check_col_contains("prescription_text", "|".join(drug_name_remove)))
)


# --------------------------------------------------------------------------------------
# PART 2: CREATE DRUG OBJECTS
# --------------------------------------------------------------------------------------


# define mappings of drug names
statin_generic_to_brand_names_map = {
    "atorvastatin": ["lipitor"],
    "rosuvastatin": ["crestor", "ezallor"],
    "simvastatin": ["zocor", "flolipid", "inegy", "simvador"],
    "pitavastatin": ["livalo", "zypitamag", "nikita"],
    "pravastatin": ["pravachol", "lipostat"],
    "lovastatin": ["mevacor", "altroprev", "altocor"],
    "fluvastatin": ["lescol"],
    "cerivastatin": ["baycol", "lipobay"],
}
non_statin_generic_to_brand_names_map = {"ezetimibe": ["inegy"]}

# create drug objects and store in dictionary
drugs = {}
for generic_name, brand_names in statin_generic_to_brand_names_map.items():
    drug = Drug(generic_name=generic_name, brand_names=brand_names)
    drugs[generic_name] = drug
for generic_name, brand_names in non_statin_generic_to_brand_names_map.items():
    drug = Drug(generic_name=generic_name, brand_names=brand_names)
    drugs[generic_name] = drug

# define brand to generic drug name mapping for later standardisation
brand_to_generic_map = {}
for drug in drugs.values():
    for brand_name in drug.brand_names:
        if brand_name not in brand_to_generic_map:
            brand_to_generic_map[brand_name] = []
        brand_to_generic_map[brand_name].append(drug.generic_name)

# --------------------------------------------------------------------------------------
# PART 3: REGEX PATTERNS FOR DRUG INFORMATION EXTRACTION
# --------------------------------------------------------------------------------------

# generic number pattern (includes spaces inbetween numbers!)
number_pattern = r"\d+(?:\.|,|\s)?\d*"

# strength patterns
strength_unit_numerators = [
    "gram",
    "g",
    "milligram",
    "mg",
    "microgram",
    "mcg",
    "unit",
    "pct",
    r"\%",
]
strength_unit_denominators = [
    "g",
    "gram",
    "mg",
    "milligram",
    "mcg",
    "microgram",
    "litre",
    "l",
    "millilitre",
    "ml",
    "dose",
]
strength_unit_pattern = (
    rf"(?:(?:{'|'.join(strength_unit_numerators)})s?)"
    rf"(?:/(?:{'|'.join(strength_unit_denominators)})s?)?"
)
strength_pattern = (
    rf"(?:{number_pattern})\s*"
    rf"(?:(?:{'|'.join(strength_unit_numerators)})s?)"
    rf"(?:/(?:{'|'.join(strength_unit_denominators)})s?)?"
)

# strength unit map for later standardisation of units
strength_unit_map = {
    "g": ["g", "gram"],
    "mg": ["mg", r"millig(?:ram)?"],
    "mcg": ["mcg", r"microg(?:ram)?"],
    "unit": ["unit"],
    "l": ["l", "litre"],
    "ml": ["ml", r"millil(?:itre)?"],
    "dose": ["dose"],
    "pct": [r"\%", "pct", "percent"],
}
for key, values in strength_unit_map.items():
    strength_unit_map[key] = "|".join([value + r"s?" for value in values])

# form mapping for standardisation and pattern
form_map = {
    "tablet": [r"tab[let]?", r"cap[sule]?", r"pastil[le]?"],
    "suspension": ["sus", "mixture", "powder"],
    "oral_solution": [r"oral sol[ution]?", r"oral liq[uid]?", r"sach[et]?"],
    "injection": [
        r"inj[ection]?",
        "pen",
        r"syr[inge]?",
        "vial",
        "needle",
        "cartridge",
        "powder and solvent",  # specific case for cutoff "suspension" string
    ],
    "drops": ["drop"],
    "pessary": [r"pes[sary]?"],
    "cream_gel_ointment": ["cream", "crm", "oin", "gel"],
    "spray": ["spray"],
}
for key, values in form_map.items():
    form_map[key] = "|".join([r"\b" + value + r"s?" for value in values])
form_pattern = "|".join(form_map.values())

# --------------------------------------------------------------------------------------
# PART 4: POLARS EXPRESSIONS FOR DRUG INFORMATION EXTRACTION
# --------------------------------------------------------------------------------------

# expression to extract generic drug names
generic_names = set([drug.generic_name for drug in drugs.values()])
extract_generic_name_expression = (
    pl.col("prescription_text")
    .str.to_lowercase()
    .str.extract_all(rf"(?i)({'|'.join(generic_names)})")
)

# expression to extract brand drug names
brand_names = set(
    [brand_name for drug in drugs.values() for brand_name in drug.brand_names]
)
extract_brand_name_expression = (
    pl.col("prescription_text")
    .str.to_lowercase()
    .str.extract_all(rf"(?i)({'|'.join(brand_names)})")
    .flatten()  # assuming no records with > 1 brand name
)

# expression to extract strength text
extract_strength_expression = (
    pl.col("prescription_text")
    # remove P42 to prevent incorrect strength extraction
    .str.replace_all(r"(?i)p42", "")
    .str.extract_all(rf"(?i)({strength_pattern})")
    .list.eval(pl.element().str.to_lowercase())
)

# expression to extract strength amounts from strength text
extract_strength_amt_expression = (
    extract_strength_expression.list.join(separator=";")
    .str.extract_all(rf"(?i)({number_pattern})")
    .list.eval(pl.element().str.replace_all(r"\s+|,", "").cast(pl.Float64))
)

# expression to extract strength units from strength text
extract_strength_unit_expression = extract_strength_expression.list.join(
    separator=";"
).str.extract_all(rf"(?i)({strength_unit_pattern})")

# expression to extract drug form
extract_form_expression = (
    pl.col("prescription_text")
    .str.extract(rf"(?i)({form_pattern})")
    .str.strip_chars()
    .str.to_lowercase()
    .map_dict(form_map)
    .cast(pl.Categorical)
)

# expression to extract drug manufacturer/brand information
manufacturer_info_pattern = r"(?:\(|\[)([^)]*)(?:\)|\])\s*$"
extract_manufacturer_info_expression = (
    pl.col("prescription_text")
    .str.extract(rf"(?i)({manufacturer_info_pattern})", group_index=1)
    .str.to_lowercase()
)

# expression to extract drug class information
extract_atc_class_code_expression = pl.col("generic_name").list.eval(
    pl.element()
    .str.replace("atorvastatin", drugs["atorvastatin"].atc_class_code)
    .str.replace("rosuvastatin", drugs["rosuvastatin"].atc_class_code)
    .str.replace("simvastatin", drugs["simvastatin"].atc_class_code)
    .str.replace("pitavastatin", drugs["pitavastatin"].atc_class_code)
    .str.replace("pravastatin", drugs["pravastatin"].atc_class_code)
    .str.replace("lovastatin", drugs["lovastatin"].atc_class_code)
    .str.replace("fluvastatin", drugs["fluvastatin"].atc_class_code)
    .str.replace("cerivastatin", drugs["cerivastatin"].atc_class_code)
    .str.replace("ezetimibe", drugs["ezetimibe"].atc_class_code)
)
extract_atc_class_name_expression = pl.col("generic_name").list.eval(
    pl.element()
    .str.replace("atorvastatin", drugs["atorvastatin"].atc_class_name)
    .str.replace("rosuvastatin", drugs["rosuvastatin"].atc_class_name)
    .str.replace("simvastatin", drugs["simvastatin"].atc_class_name)
    .str.replace("pitavastatin", drugs["pitavastatin"].atc_class_name)
    .str.replace("pravastatin", drugs["pravastatin"].atc_class_name)
    .str.replace("lovastatin", drugs["lovastatin"].atc_class_name)
    .str.replace("fluvastatin", drugs["fluvastatin"].atc_class_name)
    .str.replace("cerivastatin", drugs["cerivastatin"].atc_class_name)
    .str.replace("ezetimibe", drugs["ezetimibe"].atc_class_name)
)

# --------------------------------------------------------------------------------------
# PART 5: DRUG INFORMATION TEXT EXTRACTION QUERY
# --------------------------------------------------------------------------------------

statins = (
    statins.with_columns(
        generic_name=extract_generic_name_expression,
        brand_name=extract_brand_name_expression,
        generic_name_from_brand_name=extract_brand_name_expression.map_dict(
            brand_to_generic_map, default=[]
        ),
        strength_text=extract_strength_expression,
        strength_amt=extract_strength_amt_expression,
        strength_unit=extract_strength_unit_expression,
        form=extract_form_expression,
        manufacturer_info=extract_manufacturer_info_expression,
    )
    # concatenate generic name columns
    .with_columns(
        generic_name=pl.concat_list(
            ["generic_name", "generic_name_from_brand_name"]
        ).list.unique(),
    )
    .drop("generic_name_from_brand_name")
    .with_columns(
        # handle instances of missing strength units
        strength_amt=pl.when(
            pl.col("prescription_text").str.contains(
                rf"(?i)(?:(?:tab)|(?:cap)) ({number_pattern})$"
            )
        )
        .then(
            pl.col("prescription_text")
            .str.extract(rf"(?i)(?:(?:tab)|(?:cap)) ({number_pattern})$", group_index=1)
            .cast(pl.Float64),
        )
        .otherwise(pl.col("strength_amt")),
        strength_unit=pl.when(
            pl.col("prescription_text").str.contains(
                rf"(?i)(?:(?:tab)|(?:cap)) ({number_pattern})$"
            )
        )
        .then(pl.lit("mg"))
        .otherwise(pl.col("strength_unit")),
        # assume tablet form for all records missing form
        form=pl.when(pl.col("form").is_null())
        .then(pl.lit("tablet"))
        .otherwise(pl.col("form")),
    )
    .with_columns(
        # handle instances of duplicate strength
        strength_amt=pl.when(pl.col("prescription_text") == "SIMVASTATIN 40MG 40MG")
        .then(pl.lit(40))
        .otherwise(pl.col("strength_amt")),
        strength_unit=pl.when(pl.col("prescription_text") == "SIMVASTATIN 40MG 40MG")
        .then(pl.lit("mg"))
        .otherwise(pl.col("strength_unit")),
    )
    .with_columns(
        strength_amt=pl.when(pl.col("strength_amt") == [])
        .then(pl.lit([None]))
        .otherwise(pl.col("strength_amt")),
        strength_unit=pl.when(pl.col("strength_unit") == [])
        .then(pl.lit([None]))
        .otherwise(pl.col("strength_unit")),
    )
    .drop("strength_text")
    # add class columns
    .with_columns(
        atc_class_code=extract_atc_class_code_expression,
        atc_class_name=extract_atc_class_name_expression,
    )
    .explode(
        [
            "generic_name",
            "atc_class_code",
            "atc_class_name",
            "strength_amt",
            "strength_unit",
        ]
    )
    # standardise strength units and convert to categorical dtype
    .with_columns(
        strength_unit=pl.col("strength_unit")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace(strength_unit_map["g"], "g")
        .str.replace(strength_unit_map["mg"], "mg")
        .str.replace(strength_unit_map["mcg"], "mcg")
        .str.replace(strength_unit_map["unit"], "unit")
        .str.replace(strength_unit_map["l"], "l")
        .str.replace(strength_unit_map["ml"], "ml")
        .str.replace(strength_unit_map["dose"], "dose")
        .str.replace(strength_unit_map["pct"], "pct")
        .cast(pl.Categorical)
    )
    # remove non statin drug records
    .filter(pl.col("generic_name") != "ezetimibe")
)

# --------------------------------------------------------------------------------------
# PART 6: REGEX PATTERNS FOR QUANTITY INFORMATION EXTRACTION
# --------------------------------------------------------------------------------------

# generic number pattern (does not allow spaces inside numbers!)
quantity_number_pattern = r"\d+(?:\.|,)?\d*"  # without accepting spaces in number

# pattern to match forms (e.g. tablets) within quantity text
quantity_form_pattern = (
    r"\[?(?:(?:tab(?:let)?)|(?:tbl?\.?)|(?:cap(?:sule)?)|"
    r"(?:millilitre)|(?:ml)|(?:dose)|(?:unit))\(?s?\)?\]?"
)

# quantity patterns (pack size and number of packs)
quantity_pattern1 = rf"^\s*\(?({quantity_number_pattern})\)?\s*$"
quantity_pattern2 = (
    rf"^\s*\(?({quantity_number_pattern})"
    rf"\s*(?:\-|x|\s)?\s*(?:{quantity_form_pattern})\)?"
)
quantity_pattern3 = (
    rf"^\s*({quantity_number_pattern})"
    r"\s*(?:(?:\-?\s*packs?\s+of\s+)|(?:\-?\s*o\.?p\s+of\s+)|x|\*|\s|\-)\s*"
    rf"({quantity_number_pattern})"
    r"(?:\s*(?:\-|x)?\s*"
    rf"(?:{quantity_form_pattern}))?"
)
quantity_pattern4 = rf"^\s*x?\s*({quantity_number_pattern})\s*(?:'|\-|<|\[|b|\.|\()"

# pattern to match time units
time_units = ["day", "week", "month", "year"]
time_unit_pattern = rf"(?:{'|'.join(time_units)})s?"

# time supply and unit patterns
time_pattern1 = rf"({quantity_number_pattern})\s*(?:\-|x|\s)?\s*({time_unit_pattern})"
time_pattern2 = rf"({quantity_number_pattern})\s*/\s*12"
time_pattern3 = rf"({quantity_number_pattern})\s*/\s*52"
time_pattern4 = rf"\(?({quantity_number_pattern})\s*d\)?"
time_pattern5 = rf"number of days\s*=\s*({quantity_number_pattern})"

# --------------------------------------------------------------------------------------
# PART 7: POLARS EXPRESSIONS FOR QUANTITY INFORMATION EXTRACTION
# --------------------------------------------------------------------------------------

# expression to clean quantity_text column
clean_quantity_text_expression = (
    pl.col("quantity_text").str.replace_all(strength_pattern, "").str.strip_chars()
)

# expression to extract pack size as float
extract_pack_size_expression = (
    pl.when(check_col_contains("quantity_text", quantity_pattern1))
    .then(extract_text_from_col("quantity_text", quantity_pattern1).cast(pl.Float64))
    .when(check_col_contains("quantity_text", quantity_pattern2))
    .then(extract_text_from_col("quantity_text", quantity_pattern2).cast(pl.Float64))
    .when(check_col_contains("quantity_text", quantity_pattern3))
    .then(extract_text_from_col("quantity_text", quantity_pattern3, 2).cast(pl.Float64))
    .when(check_col_contains("quantity_text", quantity_pattern4))
    .then(extract_text_from_col("quantity_text", quantity_pattern4).cast(pl.Float64))
    .otherwise(None)
)

# expression to extract number of packs as float
extract_num_packs_expression = (
    pl.when(check_col_contains("quantity_text", quantity_pattern3))
    .then(extract_text_from_col("quantity_text", quantity_pattern3).cast(pl.Float64))
    .otherwise(None)
    .fill_null(1)
    .replace({0: 1})
)

# expression to extract time supply as float
extract_time_supply_expression = (
    pl.when(check_col_contains("quantity_text", time_pattern1))
    .then(extract_text_from_col("quantity_text", time_pattern1))
    .when(check_col_contains("quantity_text", time_pattern2))
    .then(extract_text_from_col("quantity_text", time_pattern2))
    .when(check_col_contains("quantity_text", time_pattern3))
    .then(extract_text_from_col("quantity_text", time_pattern3))
    .when(check_col_contains("quantity_text", time_pattern4))
    .then(extract_text_from_col("quantity_text", time_pattern4))
    .when(check_col_contains("quantity_text", time_pattern5))
    .then(extract_text_from_col("quantity_text", time_pattern5))
    .otherwise(None)
    .cast(pl.Int64)
)

# expression to extract or fill time units
extract_time_unit_expression = (
    pl.when(check_col_contains("quantity_text", time_pattern1))
    .then(extract_text_from_col("quantity_text", time_pattern1, group_index=2))
    .when(check_col_contains("quantity_text", time_pattern2))
    .then(pl.lit("month"))
    .when(check_col_contains("quantity_text", time_pattern3))
    .then(pl.lit("week"))
    .when(check_col_contains("quantity_text", time_pattern4))
    .then(pl.lit("day"))
    .when(check_col_contains("quantity_text", time_pattern5))
    .then(pl.lit("day"))
    .otherwise(None)
    .str.to_lowercase()
    .str.strip_chars_end("s")
    .cast(pl.Categorical)
)

# --------------------------------------------------------------------------------------
# PART 8: QUANTITY INFORMATION TEXT EXTRACTION QUERY
# --------------------------------------------------------------------------------------

# extract quantity information
statins = (
    statins.with_columns(
        quantity_text_original=pl.col("quantity_text"),
        quantity_text=clean_quantity_text_expression,
    )
    .with_columns(
        pack_size=extract_pack_size_expression,
        num_packs=extract_num_packs_expression,
        time_supply=extract_time_supply_expression,
        time_unit=extract_time_unit_expression,
    )
    # set num_packs to 1 where pack_size = num_packs
    # (e.g. "60 60 TABLETS" should be 60 tablets, not 60 x 60 = 3600)
    .with_columns(
        num_packs=pl.when(
            (pl.col("pack_size") == pl.col("num_packs"))
            & (pl.col("pack_size") != 1)
            & (pl.col("pack_size") != 0)
            & (pl.col("pack_size").is_not_null())
        )
        .then(1)
        .otherwise(pl.col("num_packs"))
    )
    .with_columns(
        # calculate final quantity from number of packs and pack size
        quantity=pl.col("pack_size") * pl.col("num_packs"),
        # clean time supply column into polars duration dtype
        time_supply=pl.when(pl.col("time_unit") == "day")
        .then(pl.duration(days=pl.col("time_supply")))
        .when(pl.col("time_unit") == "week")
        .then(pl.duration(weeks=pl.col("time_supply")))
        .when(pl.col("time_unit") == "month")
        .then(pl.duration(days=30 * pl.col("time_supply")))
        .when(pl.col("time_unit") == "year")
        .then(pl.duration(days=365 * pl.col("time_supply")))
        .otherwise(None),
    )
    .with_columns(quantity_text=pl.col("quantity_text_original"))
    .drop("quantity_text_original")
    # remove redundant time supply unit column
    .drop("time_unit")
)

# --------------------------------------------------------------------------------------
# PART 9: WRITE TO LOCAL FILE
# --------------------------------------------------------------------------------------

print(statins.collect())

out_file = ukb_user_dir / "rx_data" / "statins" / "statins_clean.parquet"
statins.collect().write_parquet(out_file)
