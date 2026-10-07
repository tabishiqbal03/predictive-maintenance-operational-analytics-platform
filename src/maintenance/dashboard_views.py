"""Presentation helpers without fitting, prediction or lifecycle side effects."""

import altair as alt
import pandas as pd

PRIORITY = ["Critical", "Schedule Maintenance", "Monitor", "Healthy"]


def sensor_chart(trend):
    values = trend.reset_index().melt("cycle", var_name="series", value_name="reading")
    return (
        alt.Chart(values)
        .mark_line()
        .encode(
            x=alt.X("cycle:Q", title="Observed cycle"),
            y=alt.Y(
                "reading:Q",
                scale=alt.Scale(zero=False),
                title="Sensor reading (raw scale; axis does not start at zero)",
            ),
            color="series:N",
            tooltip=["cycle:Q", "series:N", "reading:Q"],
        )
    )


def monitoring_tables(records):
    """Latest valid summary and one latest labelled row per scenario AND model version."""
    table = pd.json_normalize(records).sort_values("timestamp", ascending=False)
    valid = table[table.status == "valid"]
    invalid = table[table.status != "valid"]
    comparison = valid.drop_duplicates(["scenario", "model_version"])
    return valid, invalid, comparison
