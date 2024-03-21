from pathlib import Path

import polars as pl
import plotly.graph_objects as go


# Read discontinuation summary files to polars dataframes
data_dir = Path("data") / "statins"
ds_all = pl.read_csv(data_dir / "discontinuation_sample_size.csv").with_columns(
    drug=pl.lit("all")
)
ds_sim_ator = pl.read_csv(
    data_dir / "discontinuation_sample_size_simvastatin_atorvastatin.csv"
).with_columns(drug=pl.lit("simvastatin_atorvastatin"))
ds_sim = pl.read_csv(
    data_dir / "discontinuation_sample_size_simvastatin.csv"
).with_columns(drug=pl.lit("simvastatin"))
ds_ator = pl.read_csv(
    data_dir / "discontinuation_sample_size_atorvastatin.csv"
).with_columns(drug=pl.lit("atorvastatin"))

# Concatenate summary dataframes to single dataframe and clean column names
summary = (
    pl.concat(
        [
            ds_all,
            ds_sim_ator,
            ds_sim,
            ds_ator,
        ]
    )
    .sort("drug")
    .with_columns(pl.col("Group").replace("Total Statin Users", "Total Users"))
).rename(
    lambda col_name: col_name.lower()
    .replace("total", "")
    .replace("count / ", "prop")
    .replace(" ", "_")
)


# Initialize plotly figure
fig = go.Figure()

# Add traces to figure
fig.add_trace(
    go.Scatter(
        x=summary.filter(pl.col("drug") == "all").select("group").to_series().to_list(),
        y=summary.filter(pl.col("drug") == "all").select("count").to_series().to_list(),
        name="count_all",
    )
).add_trace(
    go.Bar(
        x=summary.filter(pl.col("drug") == "simvastatin_atorvastatin")
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(pl.col("drug") == "simvastatin_atorvastatin")
        .select("count")
        .to_series()
        .to_list(),
        name="count_simvastatin_atorvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(pl.col("drug") == "simvastatin")
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(pl.col("drug") == "simvastatin")
        .select("count")
        .to_series()
        .to_list(),
        name="count_simvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(pl.col("drug") == "atorvastatin")
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(pl.col("drug") == "atorvastatin")
        .select("count")
        .to_series()
        .to_list(),
        name="count_atorvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(pl.col("drug") == "all").select("group").to_series().to_list(),
        y=summary.filter(pl.col("drug") == "all")
        .select("prop_statin_users")
        .to_series()
        .to_list(),
        name="prop_statin_users_all",
    )
).add_trace(
    go.Bar(
        x=summary.filter(pl.col("drug") == "simvastatin_atorvastatin")
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(pl.col("drug") == "simvastatin_atorvastatin")
        .select("prop_statin_users")
        .to_series()
        .to_list(),
        name="prop_statin_users_simvastatin_atorvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(pl.col("drug") == "simvastatin")
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(pl.col("drug") == "simvastatin")
        .select("prop_statin_users")
        .to_series()
        .to_list(),
        name="prop_statin_users_simvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(pl.col("drug") == "atorvastatin")
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(pl.col("drug") == "atorvastatin")
        .select("prop_statin_users")
        .to_series()
        .to_list(),
        name="prop_statin_users_atorvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(pl.col("drug") == "all", pl.col("group") != "Total Users")
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(pl.col("drug") == "all", pl.col("group") != "Total Users")
        .select("prop_discontinuers")
        .to_series()
        .to_list(),
        name="prop_discontinuers_all",
    )
).add_trace(
    go.Bar(
        x=summary.filter(
            pl.col("drug") == "simvastatin_atorvastatin",
            pl.col("group") != "Total Users",
        )
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(
            pl.col("drug") == "simvastatin_atorvastatin",
            pl.col("group") != "Total Users",
        )
        .select("prop_discontinuers")
        .to_series()
        .to_list(),
        name="prop_discontinuers_simvastatin_atorvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(
            pl.col("drug") == "simvastatin", pl.col("group") != "Total Users"
        )
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(
            pl.col("drug") == "simvastatin", pl.col("group") != "Total Users"
        )
        .select("prop_discontinuers")
        .to_series()
        .to_list(),
        name="prop_discontinuers_simvastatin",
    )
).add_trace(
    go.Scatter(
        x=summary.filter(
            pl.col("drug") == "atorvastatin", pl.col("group") != "Total Users"
        )
        .select("group")
        .to_series()
        .to_list(),
        y=summary.filter(
            pl.col("drug") == "atorvastatin", pl.col("group") != "Total Users"
        )
        .select("prop_discontinuers")
        .to_series()
        .to_list(),
        name="prop_discontinuers_atorvastatin",
    )
)

# Add buttons
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list(
                [
                    dict(
                        label=""
                    )
                ]
            )
        )
    ]
)