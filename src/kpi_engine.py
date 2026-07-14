from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("data/processed/uac_features.csv")


@dataclass
class KPIResult:
    name: str
    value: float
    unit: str
    previous_value: float | None
    delta_pct: float | None
    trend: str
    description: str


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _slice_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].dropna(subset=["total_system_load"])


def _previous_period(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    span = end - start
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - span
    return prev_start, prev_end


def _trend(current: float, previous: float | None) -> tuple[float | None, str]:
    if previous is None or previous == 0 or pd.isna(previous):
        return None, "n/a"
    delta_pct = ((current - previous) / abs(previous)) * 100
    if abs(delta_pct) < 1:
        return delta_pct, "flat"
    return delta_pct, "up" if delta_pct > 0 else "down"


def kpi_total_children_under_care(cur: pd.DataFrame, prev: pd.DataFrame) -> KPIResult:
    current_value = cur["total_system_load"].iloc[-1] if not cur.empty else float("nan")
    previous_value = prev["total_system_load"].iloc[-1] if not prev.empty else None
    delta_pct, trend = _trend(current_value, previous_value)
    return KPIResult(
        name="Total Children Under Care",
        value=round(current_value, 0),
        unit="children",
        previous_value=previous_value,
        delta_pct=delta_pct,
        trend=trend,
        description="System-wide responsibility: CBP + HHS active load on the most recent reported day in range.",
    )


def kpi_net_intake_pressure(cur: pd.DataFrame, prev: pd.DataFrame) -> KPIResult:
    current_value = cur["net_daily_intake"].mean() if not cur.empty else float("nan")
    previous_value = prev["net_daily_intake"].mean() if not prev.empty else None
    delta_pct, trend = _trend(current_value, previous_value)
    return KPIResult(
        name="Net Intake Pressure",
        value=round(current_value, 1),
        unit="children/day (avg)",
        previous_value=previous_value,
        delta_pct=delta_pct,
        trend=trend,
        description="Average daily imbalance between HHS transfers-in and discharges. Positive = system filling faster than it empties.",
    )


def kpi_volatility_index(cur: pd.DataFrame, prev: pd.DataFrame) -> KPIResult:
    def cov(frame: pd.DataFrame) -> float:
        if frame.empty or frame["total_system_load"].mean() == 0:
            return float("nan")
        return (frame["total_system_load"].std() / frame["total_system_load"].mean()) * 100

    current_value = cov(cur)
    previous_value = cov(prev) if not prev.empty else None
    delta_pct, trend = _trend(current_value, previous_value)
    return KPIResult(
        name="Care Load Volatility Index",
        value=round(current_value, 2),
        unit="% (coefficient of variation)",
        previous_value=previous_value,
        delta_pct=delta_pct,
        trend=trend,
        description="Day-to-day variability of Total System Load relative to its average. Lower = more stable, predictable staffing needs.",
    )


def kpi_backlog_accumulation_rate(cur: pd.DataFrame, prev: pd.DataFrame) -> KPIResult:
    def rate(frame: pd.DataFrame) -> float:
        if frame.empty:
            return float("nan")
        return (frame["is_backlog_accumulating"].sum() / len(frame)) * 100

    current_value = rate(cur)
    previous_value = rate(prev) if not prev.empty else None
    delta_pct, trend = _trend(current_value, previous_value)
    return KPIResult(
        name="Backlog Accumulation Rate",
        value=round(current_value, 1),
        unit="% of days under sustained pressure",
        previous_value=previous_value,
        delta_pct=delta_pct,
        trend=trend,
        description="Share of days where the trailing 14-day net intake was positive, i.e. backlog was actively building rather than clearing.",
    )


def kpi_discharge_offset_ratio(cur: pd.DataFrame, prev: pd.DataFrame) -> KPIResult:
    current_value = cur["discharge_offset_ratio"].mean() if not cur.empty else float("nan")
    previous_value = prev["discharge_offset_ratio"].mean() if not prev.empty else None
    delta_pct, trend = _trend(current_value, previous_value)
    return KPIResult(
        name="Discharge Offset Ratio",
        value=round(current_value, 2),
        unit="ratio (discharges / transfers)",
        previous_value=previous_value,
        delta_pct=delta_pct,
        trend=trend,
        description="How well HHS discharge rate keeps pace with CBP transfer-in volume. 1.0 = perfectly matched; below 1.0 = load is accumulating.",
    )


KPI_FUNCTIONS = [
    kpi_total_children_under_care,
    kpi_net_intake_pressure,
    kpi_volatility_index,
    kpi_backlog_accumulation_rate,
    kpi_discharge_offset_ratio,
]


def compute_all_kpis(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> list[KPIResult]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    prev_start, prev_end = _previous_period(start, end)

    current_slice = _slice_range(df, start, end)
    previous_slice = _slice_range(df, prev_start, prev_end)

    return [fn(current_slice, previous_slice) for fn in KPI_FUNCTIONS]


if __name__ == "__main__":
    df = load_features()
    full_start, full_end = df["date"].min(), df["date"].max()

    print("=" * 60)
    print(f"KPI ENGINE - full range {full_start.date()} to {full_end.date()}")
    print("=" * 60)
    for kpi in compute_all_kpis(df, full_start, full_end):
        arrow = {"up": "^", "down": "v", "flat": "~", "n/a": "-"}[kpi.trend]
        print(f"[{arrow}] {kpi.name}: {kpi.value} {kpi.unit}")
        print(f"     {kpi.description}")

    print("\n" + "=" * 60)
    print("KPI ENGINE - last 30 reported days vs prior 30")
    print("=" * 60)
    last_30_end = full_end
    last_30_start = full_end - pd.Timedelta(days=29)
    for kpi in compute_all_kpis(df, last_30_start, last_30_end):
        arrow = {"up": "^", "down": "v", "flat": "~", "n/a": "-"}[kpi.trend]
        delta_str = f"{kpi.delta_pct:+.1f}%" if kpi.delta_pct is not None else "n/a"
        print(f"[{arrow}] {kpi.name}: {kpi.value} {kpi.unit}  (vs prior period: {delta_str})")