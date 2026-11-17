"""NYC Taxi Demand Platform: public beta (Streamlit Community Cloud).

User task: a fleet or dispatch planner looks up the hourly taxi-demand
forecast for one NYC zone and one week, and sees how accurate it was
compared with the simple "same hour last week" rule.

Reads the read-only snapshot in deliverable/data (no credentials).
"""

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import core

st.set_page_config(page_title="NYC Taxi Demand Platform", page_icon="🚕", layout="wide")


@st.cache_data(show_spinner="Loading snapshot…")
def get_snapshot(folder: str):
    return core.load_snapshot(folder)


st.title("NYC Taxi Demand Platform")
st.caption("Hourly yellow-taxi demand forecasts per NYC zone · built on a Databricks lakehouse · "
           "public beta reading a read-only snapshot")

try:
    snap = get_snapshot(str(core.data_dir()))
except FileNotFoundError as error:
    st.error(f"The data snapshot is not available: {error}")
    st.stop()

forecast_tab, model_tab, pipeline_tab, about_tab = st.tabs(["Forecast lookup", "Model accuracy", "Pipeline", "About"])

# ---------------------------------------------------------------------------
# Forecast lookup (the user task)
# ---------------------------------------------------------------------------
with forecast_tab:
    st.subheader("Look up the forecast for one zone and one week")
    st.write(f"Forecasts cover the **{snap.predictions.zone_id.nunique()} busiest taxi zones** for the 2026 hold-out "
             f"period, **{snap.first_day:%b %d, %Y} – {snap.last_day:%b %d, %Y}**. Enter a zone ID or name.")

    with st.form("lookup"):
        c1, c2 = st.columns([2, 1])
        zone_text = c1.text_input("Zone ID or name", value="", placeholder="e.g. 161 or Midtown Center")
        week_start = c2.date_input("Week starting", value=date(2026, 3, 2),
                                   min_value=date(2020, 1, 1), max_value=date(2030, 12, 31))
        submitted = st.form_submit_button("Show forecast", type="primary")

    if submitted:
        result = core.lookup(zone_text, week_start, snap)
        if result.message:
            getattr(st, result.level)(result.message)
        if result.ok:
            s = result.summary
            st.markdown(f"#### {result.zone_label}, week of {result.week_start:%b %d, %Y}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Forecast trips (week)", f"{s['forecast_trips']:,.0f}",
                      delta=f"{s['forecast_trips'] - s['actual_trips']:+,.0f} vs actual", delta_color="off")
            m2.metric("Busiest forecast hour", f"{s['peak_hour']:%a %H:00}", f"{s['peak_forecast']:.0f} trips",
                      delta_color="off")
            m3.metric("Model error (MAE)", f"{s['model_mae']:.1f} trips/hour")
            imp = s["improvement"]
            verdict = None if imp != imp else (f"model {imp:.0%} better" if imp >= 0 else f"model {-imp:.0%} worse")
            m4.metric("Last-week rule error", f"{s['baseline_mae']:.1f} trips/hour", delta=verdict, delta_color="off")

            h = result.hourly
            fig = go.Figure()
            fig.add_scatter(x=h.hour_ts, y=h.actual_trips, name="Actual", line=dict(color="#143D4A"))
            fig.add_scatter(x=h.hour_ts, y=h.predicted_trips, name="Model forecast", line=dict(color="#E8553E"))
            fig.add_scatter(x=h.hour_ts, y=h.baseline_trips, name="Same hour last week",
                            line=dict(color="#9AA5AB", dash="dot"))
            fig.update_layout(height=380, yaxis_title="Trips per hour", legend=dict(orientation="h"),
                              margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            st.download_button("Download these hours (CSV)", h.to_csv(index=False).encode(),
                               file_name=f"zone{result.zone_id}_{result.week_start}.csv", mime="text/csv")

    with st.expander("Zones you can look up"):
        z = snap.modeled_zones[["zone_id", "zone_name", "borough", "total_trips"]]
        st.dataframe(z.rename(columns={"total_trips": "trips 2023–2026"}), hide_index=True, width="stretch")

    st.subheader("Where demand is")
    map_data = snap.zones.dropna(subset=["latitude", "longitude"]).copy()
    map_data["radius"] = 150 + 1500 * (map_data.total_trips / map_data.total_trips.max()) ** 0.5
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.94, zoom=10),
        layers=[pdk.Layer("ScatterplotLayer", map_data, get_position="[longitude, latitude]",
                          get_radius="radius", get_fill_color=[232, 85, 62, 150], pickable=True)],
        tooltip={"text": "{zone_name} ({borough}) · zone {zone_id}\n{total_trips} trips"},
    ))

# ---------------------------------------------------------------------------
# Model accuracy
# ---------------------------------------------------------------------------
with model_tab:
    test = snap.metrics[snap.metrics.split == "test"].sort_values("mae")
    st.subheader("2026 hold-out: model vs baselines")
    model_mae, base_mae = core.overall_test_mae(snap)
    st.write(f"Mean absolute error over **{len(snap.predictions):,} zone-hours** "
             f"({snap.predictions.zone_id.nunique()} zones × hours from {snap.first_day} to {snap.last_day}). "
             f"Lower is better. Unit: trips per zone per hour.")
    st.plotly_chart(px.bar(test, x="model", y="mae", text_auto=".2f",
                           labels={"mae": "MAE (trips per zone-hour)", "model": ""}), width="stretch")
    st.dataframe(snap.metrics, hide_index=True, width="stretch")
    st.subheader("Error by zone (largest first)")
    st.dataframe(snap.zone_errors.drop(columns=["zone_id"]), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
with pipeline_tab:
    st.subheader("Rows per layer")
    cols = st.columns(len(snap.layer_counts))
    for col, (_, row) in zip(cols, snap.layer_counts.iterrows()):
        col.metric(row.table_name, f"{row.row_count:,.0f}")

    st.subheader("Citywide trips per day")
    st.plotly_chart(px.line(snap.daily, x="pickup_date", y="trips", labels={"pickup_date": ""}),
                    width="stretch")

    st.subheader("Quarantined trips by failed rule")
    st.plotly_chart(px.bar(snap.quarantine, x="trips", y="rule", orientation="h"), width="stretch")

    st.subheader("Benchmarks")
    experiment = st.selectbox("Experiment", sorted(snap.benchmarks.experiment.unique()))
    subset = snap.benchmarks[snap.benchmarks.experiment == experiment]
    st.plotly_chart(px.bar(subset, x="variant", y="avg_seconds", text_auto=".2f",
                           labels={"avg_seconds": "seconds", "variant": ""}), width="stretch")
    st.dataframe(snap.benchmarks, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
with about_tab:
    m = snap.manifest
    st.markdown(f"""
**What this is.** The public beta of a CS 163 project. The full system is an automated Databricks lakehouse
(Bronze → Silver → Gold, one Lakeflow pipeline and one Lakeflow Job) that ingests NYC TLC trip records, Open-Meteo
weather and NYC permitted events, and forecasts trips per zone per hour.

**Data in this app.** A read-only snapshot exported on **{m.get('exported_at_utc', 'unknown')}**.
Model `{m.get('model', {}).get('name', '?')}` version **{m.get('model', {}).get('version', '?')}**
(alias `@{m.get('model', {}).get('alias', '?')}`). Sources are public: NYC TLC, Open-Meteo, NYC Open Data.

**Limitations.**
- Forecasts exist only for the 60 busiest zones (≈98% of trips) and the 2026 hold-out period.
- The model uses *actual* weather for the forecast hours; a production system would use a weather forecast.
- It under-predicts the highest evening peaks.
- It is about 4% better than a 4-week same-hour average and about 15% better than "same hour last week".
- This beta is a snapshot. It does not refresh; the Databricks Job refreshes the full system.

Source code: see the project README.
""")
