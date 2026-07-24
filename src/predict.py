"""
predict.py — Produce the current per-region weekly risk forecast (Week 4, Day 6).

The inference step of the pipeline: build TODAY's feature row per region from the clean catalog
(in a live run, the fresh last-30-days pull), run the tuned model, turn its probability into a
calibrated risk + a Low/Med/High tier + a yes/no alert (using the frozen config/final_model.json),
and save the forecast to a `forecasts` table in SQLite.

Minimal/self-contained: refits the model on train + calibrators on validation each run (fast, and
avoids loading the registered artifact + separate calibrators). A production version would load
models:/earthquake-risk-model/1 and cache the calibrators.

Structure:
  - build_predictor      : fit tuned model on train + per-region isotonic calibrators on validation
  - build_today_features : latest-day (as-of-today) feature row per region, no label
  - predict_current      : proba -> calibrated risk -> tier + alert, per region
  - save_forecast        : write the forecast (with run date) to SQLite
  - main                 : run it and print the forecast
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, TARGET, REGIONS,
)
from train_xgboost import build_xgb_model
from tune_thresholds import BEST_VARIANT, SCALE_POS_WEIGHT
from risk_tiers import calibrate_per_region
from build_features import load_clean_data, build_daily_grid, compute_window_features


CONFIG_PATH = Path(__file__).parent.parent / "config" / "final_model.json"
DB_PATH = Path(__file__).parent.parent / "data" / "earthquakes.db"
FORECAST_TABLE = "forecasts"


def build_predictor():
    """
    Fit the tuned model on TRAIN and per-region isotonic calibrators on VALIDATION.

    Minimal/self-contained: refits deterministically (fast) rather than loading the registered
    MLflow model, and re-fits the calibrators (which aren't part of the registered artifact).
    A production version would load models:/earthquake-risk-model/1 and cache the calibrators.

    Returns:
        tuple[Pipeline, dict, list[str]]: (fitted model, {region: isotonic calibrator}, feature_cols).
    """
    parts = split_data(load_features())
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, BEST_VARIANT)
    feature_cols = numeric + categorical

    model = build_xgb_model(build_preprocessor(numeric, categorical), SCALE_POS_WEIGHT)
    model.fit(parts["train"][feature_cols], parts["train"][TARGET])

    val = parts["validate"]
    _, calibrators = calibrate_per_region(val, model.predict_proba(val[feature_cols])[:, 1])
    return model, calibrators, feature_cols


def build_today_features():
    """
    Compute the latest-day ("as-of-today") feature row per region from the clean catalog.

    Reuses build_features' window logic on whatever clean data is present — in a live run that's
    the fresh last-30-days pull, so the latest row per region is TODAY's features. No label is
    computed (we're predicting the future, not scoring the past).

    Returns:
        pd.DataFrame: one row per region — the most recent (region, date) with the 6 features.
    """
    df = load_clean_data()
    grid = build_daily_grid(df)
    feats = compute_window_features(df, grid)
    # Keep only the most recent available day for each region.
    return feats.sort_values("date").groupby("region", as_index=False).tail(1)


def predict_current(model, calibrators, config, today):
    """
    Turn each region's today-features into a forecast: calibrated risk + tier + yes/no alert.

    Args:
        model (Pipeline): fitted model from build_predictor.
        calibrators (dict): {region: isotonic calibrator} from build_predictor.
        config (dict): the frozen final_model.json (features, thresholds, base_rates, high_mult).
        today (pd.DataFrame): one row per region from build_today_features.

    Returns:
        pd.DataFrame: [region, date, risk (calibrated), tier, alert, threshold, raw_proba].
    """
    feature_cols = config["features"]
    thresholds, base_rates, high_mult = config["thresholds"], config["base_rates"], config["high_mult"]

    rows = []
    for _, row in today.iterrows():
        region = row["region"]
        X = pd.DataFrame([row[feature_cols]])                 # one-row frame for the model
        raw = float(model.predict_proba(X)[:, 1][0])          # model's raw probability
        risk = float(calibrators[region].predict([raw])[0])   # calibrated risk (0-1)

        # Tier from the region's base-rate bands, on the CALIBRATED risk.
        b = base_rates[region]
        tier = "High" if risk >= high_mult * b else ("Medium" if risk >= b else "Low")
        # Yes/no alert from the region's F2 threshold, on the RAW proba (as tuned).
        alert = bool(raw >= thresholds[region])

        rows.append({
            "region": region,
            "date": row["date"].date().isoformat(),
            "risk": round(risk, 3),
            "tier": tier,
            "alert": alert,
            "threshold": thresholds[region],
            "raw_proba": round(raw, 3),
        })
    return pd.DataFrame(rows)


def save_forecast(forecast, db_path=DB_PATH, table=FORECAST_TABLE):
    """
    Append the forecast to the SQLite `forecasts` table, stamped with the run time.

    Appends (doesn't replace) so the table builds a track record of forecasts over time —
    which stakeholders can later check for calibration.
    """
    stamped = forecast.copy()
    stamped.insert(0, "run_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with sqlite3.connect(db_path) as conn:
        stamped.to_sql(table, conn, if_exists="append", index=False)
    return stamped


def main():
    """Generate and save the current per-region forecast."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    model, calibrators, _ = build_predictor()
    today = build_today_features()
    forecast = predict_current(model, calibrators, config, today)
    saved = save_forecast(forecast)

    print("Per-region weekly risk forecast:\n")
    print(forecast.to_string(index=False))
    print(f"\nSaved {len(forecast)} rows to '{FORECAST_TABLE}' in {DB_PATH.name} "
          f"(run_at {saved['run_at'].iloc[0]}).")


# Only run when launched directly (e.g. "python src/predict.py").
if __name__ == "__main__":
    main()