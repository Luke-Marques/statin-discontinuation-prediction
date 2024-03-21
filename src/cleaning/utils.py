from dataclasses import dataclass
from typing import Iterable

import polars as pl
import requests


@dataclass
class Drug:
    """Class for keeping track of drugs in datasets."""

    generic_name: str
    brand_names: Iterable[str] | None
    rxcui: int | None = None
    atc_class_code: str | None = None
    atc_class_name: str | None = None

    def __post_init__(self):
        self.brand_names = sorted(self.brand_names)
        rxclass_results = self._find_rxcui_and_class_using_rxclass_api()
        if rxclass_results:
            self.rxcui = rxclass_results["rxcui"]
            self.atc_class_code = rxclass_results["atc_class_code"]
            self.atc_class_name = rxclass_results["atc_class_name"]

    def _find_rxcui_and_class_using_rxclass_api(
        self, source: str = "ATC", relationships: str = "ALL"
    ):
        service_domain = "https://rxnav.nlm.nih.gov"
        request_string = (
            f"{service_domain}/REST/rxclass/class/byDrugName.json?"
            f"drugName={self.generic_name}&relaSource={source}&relas={relationships}"
        )
        response = requests.get(request_string).json()

        for result in response["rxclassDrugInfoList"]["rxclassDrugInfo"]:
            if result["minConcept"]["name"] == self.generic_name:
                return {
                    "rxcui": result["minConcept"]["rxcui"],
                    "atc_class_code": result["rxclassMinConceptItem"]["classId"],
                    "atc_class_name": result["rxclassMinConceptItem"]["className"],
                }
        return None


def get_polars_col(col: str | pl.Series) -> pl.Series:
    if isinstance(col, str):
        col = pl.col(col)
    return col


def add_ignore_case_flag_to_regex(pattern: str) -> str:
    return r"(?i)" + pattern


def check_col_contains(
    col: str | pl.Series, pattern: str, ignore_case: bool = True
) -> pl.Expr:
    col = get_polars_col(col)
    if ignore_case:
        pattern = add_ignore_case_flag_to_regex(pattern)
    return col.str.contains(pattern)


def extract_text_from_col(
    col: str | pl.Series, pattern: str, group_index: int = 1, ignore_case: bool = True
) -> pl.Expr:
    col = get_polars_col(col)
    if ignore_case:
        pattern = add_ignore_case_flag_to_regex(pattern)
    return col.str.extract(pattern, group_index=group_index).str.strip_chars()
