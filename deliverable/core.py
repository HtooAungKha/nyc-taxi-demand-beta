"""Data access and request validation for the public beta.

This module has no Streamlit code, so it can be unit-tested directly.
The app reads a read-only snapshot exported by
`05_publish/01_export_snapshot` (see deliverable/data/manifest.json).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def data_dir() -> Path:
    """Snapshot folder; SNAPSHOT_DIR overrides it (used by the tests)."""
    return Path(os.environ.get("SNAPSHOT_DIR", DEFAULT_DATA_DIR))

REQUIRED_FILES = [
    "zones.csv",
    "citywide_daily.csv",
    "zone_hourly_predictions.parquet",
    "model_metrics.csv",
    "zone_error_summary.csv",
    "benchmarks.csv",
    "layer_counts.csv",
    "quarantine_rules.csv",
    "manifest.json",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    zones: pd.DataFrame
    daily: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    zone_errors: pd.DataFrame
    benchmarks: pd.DataFrame
    layer_counts: pd.DataFrame
    quarantine: pd.DataFrame
    manifest: dict

    @property
    def modeled_zones(self) -> pd.DataFrame:
        ids = set(self.predictions.zone_id.unique())
        return self.zones[self.zones.zone_id.isin(ids)].sort_values("total_trips", ascending=False)

    @property
    def first_day(self) -> date:
        return self.predictions.hour_ts.min().date()

    @property
    def last_day(self) -> date:
        return self.predictions.hour_ts.max().date()


def missing_files(folder: Path | str | None = None) -> list[str]:
    folder = Path(folder) if folder else data_dir()
    return [f for f in REQUIRED_FILES if not (folder / f).exists()]


def load_snapshot(folder: Path | str | None = None) -> Snapshot:
    d = Path(folder) if folder else data_dir()
    missing = missing_files(d)
    if missing:
        raise FileNotFoundError(f"Snapshot files missing in {d}: {', '.join(missing)}")

    predictions = pd.read_parquet(d / "zone_hourly_predictions.parquet")
    predictions["hour_ts"] = pd.to_datetime(predictions["hour_ts"])
    daily = pd.read_csv(d / "citywide_daily.csv", parse_dates=["pickup_date"])
    return Snapshot(
        zones=pd.read_csv(d / "zones.csv"),
        daily=daily,
        predictions=predictions,
        metrics=pd.read_csv(d / "model_metrics.csv"),
        zone_errors=pd.read_csv(d / "zone_error_summary.csv"),
        benchmarks=pd.read_csv(d / "benchmarks.csv"),
        layer_counts=pd.read_csv(d / "layer_counts.csv"),
        quarantine=pd.read_csv(d / "quarantine_rules.csv"),
        manifest=json.loads((d / "manifest.json").read_text()),
    )


# ---------------------------------------------------------------------------
# Request validation: the user task is "show the forecast for zone Z,
# week starting D, and how much to trust it".
# ---------------------------------------------------------------------------
@dataclass
class LookupResult:
    ok: bool
    message: str = ""
    level: str = "info"            # info | warning | error
    zone_id: int | None = None
    zone_label: str = ""
    week_start: date | None = None
    hourly: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)


def resolve_zone(text: str, snap: Snapshot) -> tuple[int | None, str, str]:
    """Return (zone_id, label, problem). `problem` is empty when the zone is usable."""
    text = (text or "").strip()
    if not text:
        return None, "", "Enter a zone ID (for example 161) or a zone name (for example Midtown Center)."

    zones = snap.zones
    if text.isdigit():
        match = zones[zones.zone_id == int(text)]
    else:
        lowered = text.lower()
        match = zones[zones.zone_name.str.lower() == lowered]
        if match.empty:
            match = zones[zones.zone_name.str.lower().str.contains(lowered, regex=False)]
            if len(match) > 1:
                names = ", ".join(match.zone_name.head(5))
                return None, "", f"'{text}' matches {len(match)} zones ({names}…). Type the full name or the zone ID."

    if match.empty:
        return None, "", f"No NYC taxi zone matches '{text}'. Valid zone IDs are 1–263; see the zone list below."

    row = match.iloc[0]
    label = f"{row.zone_name} · {row.borough} (zone {int(row.zone_id)})"
    if int(row.zone_id) not in set(snap.predictions.zone_id.unique()):
        return None, label, (f"{label} is a real taxi zone, but it is not one of the "
                             f"{snap.predictions.zone_id.nunique()} busiest zones the model forecasts. "
                             "Pick a zone from the list below.")
    return int(row.zone_id), label, ""


def lookup(zone_text: str, week_start: date | None, snap: Snapshot) -> LookupResult:
    zone_id, label, problem = resolve_zone(zone_text, snap)
    if problem:
        return LookupResult(ok=False, message=problem, level="warning" if label else "error", zone_label=label)

    if week_start is None:
        return LookupResult(ok=False, message="Choose the first day of the week.", level="error")

    first, last = snap.first_day, snap.last_day
    if week_start < first or week_start > last:
        return LookupResult(
            ok=False, level="error", zone_id=zone_id, zone_label=label,
            message=(f"{week_start:%Y-%m-%d} is outside the forecast period. "
                     f"Forecasts are scored on the 2026 hold-out data only: {first:%Y-%m-%d} to {last:%Y-%m-%d}."),
        )

    end = week_start + timedelta(days=7)
    p = snap.predictions
    hourly = p[(p.zone_id == zone_id) & (p.hour_ts >= pd.Timestamp(week_start)) & (p.hour_ts < pd.Timestamp(end))]
    hourly = hourly.sort_values("hour_ts").reset_index(drop=True)
    if hourly.empty:
        return LookupResult(ok=False, level="error", zone_id=zone_id, zone_label=label,
                            message="No forecast rows for that zone and week.")

    days = hourly.hour_ts.dt.date.nunique()
    summary = week_summary(hourly)
    message = ""
    level = "info"
    if days < 7:
        level = "warning"
        message = (f"Only {days} day(s) of data are available from {week_start:%Y-%m-%d} "
                   f"(the hold-out period ends {last:%Y-%m-%d}). Results cover those days only.")
    return LookupResult(ok=True, message=message, level=level, zone_id=zone_id, zone_label=label,
                        week_start=week_start, hourly=hourly, summary=summary)


def week_summary(hourly: pd.DataFrame) -> dict:
    """Error of the model and the last-week baseline over the selected hours (trips per zone-hour)."""
    model_mae = (hourly.actual_trips - hourly.predicted_trips).abs().mean()
    base_mae = (hourly.actual_trips - hourly.baseline_trips).abs().mean()
    peak = hourly.loc[hourly.predicted_trips.idxmax()]
    return {
        "hours": int(len(hourly)),
        "actual_trips": int(hourly.actual_trips.sum()),
        "forecast_trips": float(hourly.predicted_trips.sum()),
        "model_mae": float(model_mae),
        "baseline_mae": float(base_mae),
        "improvement": float(1 - model_mae / base_mae) if base_mae else float("nan"),
        "peak_hour": peak.hour_ts,
        "peak_forecast": float(peak.predicted_trips),
    }


def overall_test_mae(snap: Snapshot) -> tuple[float, float]:
    """MAE of model and last-week baseline over the whole hold-out snapshot."""
    p = snap.predictions
    return (float((p.actual_trips - p.predicted_trips).abs().mean()),
            float((p.actual_trips - p.baseline_trips).abs().mean()))
