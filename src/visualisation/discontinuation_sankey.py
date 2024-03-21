from pathlib import Path
from typing import Tuple

import polars as pl
import plotly
import plotly.graph_objects as go


def get_col(col: str | pl.Expr) -> pl.Expr:
    if isinstance(col, str):
        return pl.col(col)
    return col


def get_participant_count(
    rx: pl.LazyFrame | pl.DataFrame,
    filter_expr: pl.Expr | None = None,
    participant_col: str = "eid",
) -> pl.Expr | int:
    """
    Syntactic sugar to identify number of participants meeting some criteria specified
    in the `filter_expr` polars expression.
    """
    participant_col = get_col(participant_col)
    if filter_expr is not None:
        count = rx.select(participant_col.filter(filter_expr)).unique().count()
    else:
        count = rx.select(participant_col).unique().count()
    if isinstance(rx, pl.LazyFrame):
        count = count.collect().item()
    else:
        count = count.item()
    return count


def get_rx_user_count(rx: pl.LazyFrame) -> int:
    return get_participant_count(rx)


def get_rx_discontinuer_count(rx: pl.LazyFrame) -> int:
    filter_expr = pl.col("discontinued").sum().over("eid") > 0
    return get_participant_count(rx, filter_expr)


def get_rx_restarter_count(rx: pl.LazyFrame, max_restarts: int | None = None) -> int:
    filter_expr = pl.col("restarted").sum().over("eid") > 0
    if max_restarts:
        filter_expr = filter_expr & (
            pl.col("restarted").sum().over("eid") <= max_restarts
        )
    return get_participant_count(rx, filter_expr)


def get_rx_final_discontinuation_count(
    rx: pl.LazyFrame,
    min_prior_discontinuations: int | None = None,
    max_prior_discontinuations: int | None = None,
    max_time_to_final_discontinuation: pl.Duration | None = None,
) -> int:
    filter_expr = pl.col("discontinued") & (
        pl.col("issue_date").shift(-1).over("eid").is_null()
    )
    if min_prior_discontinuations is not None:
        filter_expr = filter_expr & (
            pl.col("discontinued").cum_sum().over("eid").shift()
            >= min_prior_discontinuations
        )
    if max_prior_discontinuations is not None:
        filter_expr = filter_expr & (
            pl.col("discontinued").cum_sum().over("eid").shift()
            <= max_prior_discontinuations
        )
    if max_time_to_final_discontinuation is not None:
        filter_expr = filter_expr & (
            (pl.col("issue_date") - pl.col("issue_date").first()).over("eid")
            <= max_time_to_final_discontinuation
        )

    return get_participant_count(rx, filter_expr)


def generate_discontinuation_sankey_diagram(
    rx: pl.LazyFrame, out_path: Path | None = None, num_years: int = 2
) -> plotly.graph_objs._figure.Figure:
    """
    Generates a plotly sankey diagram showing the sample sizes and progression of
    prescription discontinuation/restarts.
    """

    DAYS_IN_YEAR = 365

    # get node/ribbon values (participant counts)
    user_count = get_rx_user_count(rx)
    discontinuer_count = get_rx_discontinuer_count(rx)
    restarter_count = get_rx_restarter_count(rx)
    single_restarter_count = get_rx_restarter_count(rx, max_restarts=1)
    multiple_restarter_count = restarter_count - single_restarter_count
    final_discontinuation_count = get_rx_final_discontinuation_count(rx)
    final_discontinuation_no_prior_discontinuations_count = (
        get_rx_final_discontinuation_count(rx, max_prior_discontinuations=0)
    )
    final_discontinuation_single_prior_discontinuation_count = (
        get_rx_final_discontinuation_count(
            rx, min_prior_discontinuations=1, max_prior_discontinuations=1
        )
    )
    final_discontinuation_multiple_prior_discontinuations_count = (
        get_rx_final_discontinuation_count(rx, min_prior_discontinuations=2)
    )
    final_discontinuation_within_first_year_count = get_rx_final_discontinuation_count(
        rx, max_time_to_final_discontinuation=pl.duration(days=DAYS_IN_YEAR * num_years)
    )
    final_discontinuation_after_first_year_count = (
        final_discontinuation_count - final_discontinuation_within_first_year_count
    )
    continuer_count = (user_count - discontinuer_count) + (
        restarter_count
        - final_discontinuation_single_prior_discontinuation_count
        - final_discontinuation_multiple_prior_discontinuations_count
    )

    # define node labels
    node_labels = [
        f"Users ({user_count:,})",  # node 0
        f"Continuers ({continuer_count:,})",  # node 1
        f"Discontinuers ({discontinuer_count:,})",  # node 2
        f"Restarters ({restarter_count:,})",  # node 3
        ("Single Restart " f"({single_restarter_count:,})",),  # node 4
        ("Multiple Restarts " f"({multiple_restarter_count:,})",),  # node 5
        f"Final Discontinuation ({final_discontinuation_count:,})",  # node 6
        f"After Year {num_years} ({final_discontinuation_after_first_year_count:,})",  # node 7
        f"Within Year {num_years} ({final_discontinuation_within_first_year_count:,})",  # node 8
    ]

    [
        ("Users", 0),
        ("Continuers", 1),
        ("Discontinuers", 2),
        ("Restarters", 3),
        ("Single Restart", 4),
        ("Multiple Restarts", 5),
        ("Final Discontinuation", 6),
        ("After 1st Year", 7),
        ("Within First Year", 8),
    ]

    # define node colors
    node_colors = [
        "slategrey",
        "royalblue",
        "cornflowerblue",
        "skyblue",
        "orange",
        "orangered",
        "lightgreen",
        "skyblue",
        "orange",
    ]

    # define node positions
    # TODO: ^

    # define node links
    node_links = [
        {"source_label_index": 0, "target_label_index": 1, "value": continuer_count},
        {"source_label_index": 0, "target_label_index": 2, "value": discontinuer_count},
        {"source_label_index": 2, "target_label_index": 3, "value": restarter_count},
        {
            "source_label_index": 2,
            "target_label_index": 6,
            "value": final_discontinuation_no_prior_discontinuations_count,
        },
        {
            "source_label_index": 3,
            "target_label_index": 4,
            "value": single_restarter_count,
        },
        {
            "source_label_index": 3,
            "target_label_index": 5,
            "value": multiple_restarter_count,
        },
        {
            "source_label_index": 4,
            "target_label_index": 6,
            "value": final_discontinuation_single_prior_discontinuation_count,
        },
        {
            "source_label_index": 4,
            "target_label_index": 1,
            "value": (
                single_restarter_count
                - final_discontinuation_single_prior_discontinuation_count
            ),
        },
        {
            "source_label_index": 5,
            "target_label_index": 6,
            "value": final_discontinuation_multiple_prior_discontinuations_count,
        },
        {
            "source_label_index": 5,
            "target_label_index": 1,
            "value": (
                multiple_restarter_count
                - final_discontinuation_multiple_prior_discontinuations_count
            ),
        },
        {
            "source_label_index": 6,
            "target_label_index": 7,
            "value": final_discontinuation_after_first_year_count,
        },
        {
            "source_label_index": 6,
            "target_label_index": 8,
            "value": final_discontinuation_within_first_year_count,
        },
    ]

    # plot sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=15,
                    # line=dict(color="black", width=0.5),
                    label=node_labels,
                    # x=xpos,
                    # y=ypos,
                    color=node_colors,
                ),
                link=dict(
                    source=[node["source_label_index"] for node in node_links],
                    target=[node["target_label_index"] for node in node_links],
                    value=[node["value"] for node in node_links],
                ),
            )
        ]
    )

    # optionally, save figure to out_path
    if out_path:
        fig.write_image(out_path)

    return fig


def main() -> None:
    return


if __name__ == "__main__":
    main()
