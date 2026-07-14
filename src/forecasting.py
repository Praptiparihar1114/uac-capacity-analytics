from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_PATH = Path("data/processed/uac_features.csv")
FORECAST_PATH = Path("data/processed/uac_forecast.csv")
METRICS_PATH = Path("data/processed/forecast_metrics.json")

HOLDOUT_DAYS = 30
FORECAST_HORIZON_DAYS = 14
LAGS = [1, 2, 3, 7, 14]
RANDOM_STATE = 42


def load_series() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.dropna(subset=["total_system_load"]).sort_values("date").reset_index(drop=True)
    return df[["date", "total_system_load"]].copy()


def build_feature_frame(series_df: pd.DataFrame) -> pd.DataFrame:
    df = series_df.copy()
    for lag in LAGS:
        df["lag_" + str(lag)] = df["total_system_load"].shift(lag)
    df["rolling_avg_7"] = df["total_system_load"].shift(1).rolling(7).mean()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df.dropna().reset_index(drop=True)


def get_feature_columns() -> list[str]:
    return ["lag_" + str(lag) for lag in LAGS] + ["rolling_avg_7", "day_of_week", "month"]


def evaluate_holdout(feature_df: pd.DataFrame) -> dict:
    feature_cols = get_feature_columns()
    train_df = feature_df.iloc[:-HOLDOUT_DAYS]
    test_df = feature_df.iloc[-HOLDOUT_DAYS:]

    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    model.fit(train_df[feature_cols], train_df["total_system_load"])
    predictions = model.predict(test_df[feature_cols])

    actuals = test_df["total_system_load"].values
    baseline_predictions = test_df["lag_7"].values

    def compute_metrics(y_true, y_pred) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape_pct": round(mape, 2)}

    model_metrics = compute_metrics(actuals, predictions)
    baseline_metrics = compute_metrics(actuals, baseline_predictions)

    return {
        "holdout_days": HOLDOUT_DAYS,
        "model": "RandomForestRegressor",
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_description": "Seasonal-naive: predicts the value from 7 days prior",
        "test_dates": test_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "test_actuals": [round(v, 1) for v in actuals.tolist()],
        "test_predictions": [round(v, 1) for v in predictions.tolist()],
    }


def train_full_model(feature_df: pd.DataFrame) -> RandomForestRegressor:
    feature_cols = get_feature_columns()
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    model.fit(feature_df[feature_cols], feature_df["total_system_load"])
    return model


def forecast_forward(series_df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    history = series_df["total_system_load"].tolist()
    last_date = series_df["date"].max()
    feature_cols = get_feature_columns()

    forecast_rows = []
    for step in range(1, FORECAST_HORIZON_DAYS + 1):
        forecast_date = last_date + pd.Timedelta(days=step)
        row = {}
        for lag in LAGS:
            row["lag_" + str(lag)] = history[-lag]
        row["rolling_avg_7"] = float(np.mean(history[-7:]))
        row["day_of_week"] = forecast_date.dayofweek
        row["month"] = forecast_date.month

        x_input = pd.DataFrame([row])[feature_cols]
        predicted_value = float(model.predict(x_input)[0])

        forecast_rows.append({"date": forecast_date, "predicted_total_system_load": round(predicted_value, 1)})
        history.append(predicted_value)

    return pd.DataFrame(forecast_rows)


def run_pipeline() -> tuple[pd.DataFrame, dict]:
    series_df = load_series()
    feature_df = build_feature_frame(series_df)

    metrics = evaluate_holdout(feature_df)
    full_model = train_full_model(feature_df)
    forecast_df = forecast_forward(series_df, full_model)

    return forecast_df, metrics


def save_outputs(forecast_df: pd.DataFrame, metrics: dict) -> None:
    FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(FORECAST_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    forecast_df, metrics = run_pipeline()
    save_outputs(forecast_df, metrics)

    print("=" * 60)
    print("FORECASTING COMPLETE")
    print("=" * 60)
    print("Holdout evaluation (last " + str(metrics["holdout_days"]) + " days):")
    print("  Model (" + metrics["model"] + "):")
    print("    MAE : " + str(metrics["model_metrics"]["mae"]))
    print("    RMSE: " + str(metrics["model_metrics"]["rmse"]))
    print("    MAPE: " + str(metrics["model_metrics"]["mape_pct"]) + "%")
    print("  Baseline (" + metrics["baseline_description"] + "):")
    print("    MAE : " + str(metrics["baseline_metrics"]["mae"]))
    print("    RMSE: " + str(metrics["baseline_metrics"]["rmse"]))
    print("    MAPE: " + str(metrics["baseline_metrics"]["mape_pct"]) + "%")
    print("")
    print("Forward forecast (next " + str(FORECAST_HORIZON_DAYS) + " days):")
    print(forecast_df.to_string(index=False))
    print("")
    print("Saved forecast -> " + str(FORECAST_PATH))
    print("Saved metrics -> " + str(METRICS_PATH))