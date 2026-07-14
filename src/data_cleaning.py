from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/hhs_uac_program.csv")
CLEAN_PATH = Path("data/processed/uac_cleaned.csv")
QUALITY_REPORT_PATH = Path("data/processed/data_quality_report.json")

COLUMN_RENAME_MAP = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "cbp_intake",
    "Children in CBP custody": "cbp_active_load",
    "Children transferred out of CBP custody": "cbp_to_hhs_transfers",
    "Children in HHS Care": "hhs_active_load",
    "Children discharged from HHS Care": "hhs_discharges",
}

NUMERIC_COLS = [
    "cbp_intake",
    "cbp_active_load",
    "cbp_to_hhs_transfers",
    "hhs_active_load",
    "hhs_discharges",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna(how="all").reset_index(drop=True)


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"nan": None, "": None})
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("Int64")


def structure_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    for col in NUMERIC_COLS:
        df[col] = clean_numeric(df[col])
    return df.sort_values("date").reset_index(drop=True)


def build_complete_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_dates = sorted(set(full_range) - set(df["date"]))
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"
    df = df.reset_index()
    return df, [d.strftime("%Y-%m-%d") for d in missing_dates]


def flag_duplicates(raw_df: pd.DataFrame) -> list[str]:
    dupes = raw_df.loc[raw_df.duplicated(subset="date", keep=False), "date"]
    return sorted({d.strftime("%Y-%m-%d") for d in dupes.dropna()})


def validate_logical_constraints(df: pd.DataFrame) -> dict:
    df["flag_transfers_exceed_cbp_load"] = (
        df["cbp_to_hhs_transfers"] > df["cbp_active_load"]
    ).fillna(False)
    df["flag_discharges_exceed_hhs_load"] = (
        df["hhs_discharges"] > df["hhs_active_load"]
    ).fillna(False)
    df["is_anomalous"] = df["flag_transfers_exceed_cbp_load"] | df["flag_discharges_exceed_hhs_load"]

    return {
        "transfers_exceed_cbp_load": df.loc[
            df["flag_transfers_exceed_cbp_load"], "date"
        ].dt.strftime("%Y-%m-%d").tolist(),
        "discharges_exceed_hhs_load": df.loc[
            df["flag_discharges_exceed_hhs_load"], "date"
        ].dt.strftime("%Y-%m-%d").tolist(),
    }


def run_pipeline(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, dict]:
    raw_df = load_raw(raw_path)
    structured = structure_data(raw_df)

    duplicate_dates = flag_duplicates(structured)
    complete_df, missing_dates = build_complete_calendar(structured)
    violations = validate_logical_constraints(complete_df)

    complete_df["is_missing_report"] = (
        complete_df["cbp_intake"].isna() & complete_df["cbp_active_load"].isna()
    )

    quality_report = {
        "date_range": {
            "start": complete_df["date"].min().strftime("%Y-%m-%d"),
            "end": complete_df["date"].max().strftime("%Y-%m-%d"),
        },
        "total_calendar_days": len(complete_df),
        "reported_days": int((~complete_df["is_missing_report"]).sum()),
        "missing_report_days": int(complete_df["is_missing_report"].sum()),
        "missing_dates_sample": missing_dates[:25],
        "missing_dates_total": len(missing_dates),
        "duplicate_dates_found": duplicate_dates,
        "logical_constraint_violations": violations,
        "anomalous_rows_total": int(complete_df["is_anomalous"].sum()),
    }

    return complete_df, quality_report


def save_outputs(df: pd.DataFrame, report: dict) -> None:
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    cleaned_df, quality_report = run_pipeline()
    save_outputs(cleaned_df, quality_report)

    print("=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)
    print(f"Date range        : {quality_report['date_range']['start']} -> "
          f"{quality_report['date_range']['end']}")
    print(f"Total calendar days: {quality_report['total_calendar_days']}")
    print(f"Reported days      : {quality_report['reported_days']}")
    print(f"Missing report days: {quality_report['missing_report_days']}")
    print(f"Duplicate dates    : {len(quality_report['duplicate_dates_found'])}")
    print(f"Anomalous rows     : {quality_report['anomalous_rows_total']}")
    print(f"\nSaved cleaned data -> {CLEAN_PATH}")
    print(f"Saved quality report -> {QUALITY_REPORT_PATH}")