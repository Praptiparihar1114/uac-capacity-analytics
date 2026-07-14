from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/processed/uac_cleaned.csv")
FEATURES_PATH = Path("data/processed/uac_features.csv")

ROLLING_WINDOWS = (7, 14)

STRESS_Z_THRESHOLD = 1.0
RELIEF_Z_THRESHOLD = -1.0


def load_cleaned(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def add_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["total_system_load"] = df["cbp_active_load"] + df["hhs_active_load"]
    df["net_daily_intake"] = df["cbp_to_hhs_transfers"] - df["hhs_discharges"]
    df["care_load_growth_rate_pct"] = df["total_system_load"].pct_change() * 100
    df["discharge_offset_ratio"] = np.where(
        df["cbp_to_hhs_transfers"] > 0,
        df["hhs_discharges"] / df["cbp_to_hhs_transfers"],
        np.nan,
    )
    return df


def add_rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    for window in ROLLING_WINDOWS:
        df[f"total_load_roll_avg_{window}d"] = (
            df["total_system_load"].rolling(window, min_periods=1).mean()
        )
        df[f"total_load_roll_std_{window}d"] = (
            df["total_system_load"].rolling(window, min_periods=2).std()
        )
        df[f"net_intake_roll_avg_{window}d"] = (
            df["net_daily_intake"].rolling(window, min_periods=1).mean()
        )
    return df


def add_backlog_indicator(df: pd.DataFrame) -> pd.DataFrame:
    is_positive = df["net_daily_intake"] > 0
    streak = (is_positive.groupby((~is_positive).cumsum()).cumcount() + 1) * is_positive
    df["backlog_streak_days"] = streak.astype(int)
    df["backlog_rolling_sum_14d"] = df["net_daily_intake"].rolling(14, min_periods=1).sum()
    df["is_backlog_accumulating"] = df["backlog_rolling_sum_14d"] > 0
    return df


def add_stress_relief_classification(df: pd.DataFrame) -> pd.DataFrame:
    baseline_mean = df["total_load_roll_avg_7d"].rolling(90, min_periods=14).mean()
    baseline_std = df["total_load_roll_avg_7d"].rolling(90, min_periods=14).std()
    df["load_zscore"] = (df["total_load_roll_avg_7d"] - baseline_mean) / baseline_std

    conditions = [
        df["load_zscore"] >= STRESS_Z_THRESHOLD,
        df["load_zscore"] <= RELIEF_Z_THRESHOLD,
    ]
    choices = ["Stress", "Relief"]
    df["system_regime"] = np.select(conditions, choices, default="Normal")
    return df


def run_pipeline(clean_path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = load_cleaned(clean_path)
    df = add_core_metrics(df)
    df = add_rolling_metrics(df)
    df = add_backlog_indicator(df)
    df = add_stress_relief_classification(df)
    return df


def save_output(df: pd.DataFrame) -> None:
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)


if __name__ == "__main__":
    features_df = run_pipeline()
    save_output(features_df)

    reported = features_df.dropna(subset=["total_system_load"])
    regime_counts = reported["system_regime"].value_counts().to_dict()

    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Rows processed         : {len(features_df)}")
    print(f"Avg Total System Load  : {reported['total_system_load'].mean():,.0f}")
    print(f"Max Total System Load  : {reported['total_system_load'].max():,.0f}")
    print(f"Days in Stress regime  : {regime_counts.get('Stress', 0)}")
    print(f"Days in Relief regime  : {regime_counts.get('Relief', 0)}")
    print(f"Days in Normal regime  : {regime_counts.get('Normal', 0)}")
    print(f"Longest backlog streak : {int(reported['backlog_streak_days'].max())} days")
    print(f"\nSaved feature set -> {FEATURES_PATH}")