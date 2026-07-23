"""
evaluate_test.py — The one-shot honest evaluation on the sealed TEST set
(Week 4, Day 5, step 3 of finalize).

Rebuilds the EXACT locked pipeline from the frozen config (config/final_model.json): the tuned
model fit on TRAIN, per-region isotonic calibrators fit on VALIDATION. Applies it ONCE to the
TEST split (2022-26) and reports the honest final numbers — ranking (PR-AUC/ROC-AUC), alerts at
the frozen F2 thresholds, calibrated Low/Med/High tiers, and calibration (Brier) — alongside the
validation numbers so we can see how well it generalized. Results are logged to MLflow.

The model is fit on TRAIN only (calibrators on VALIDATION) to match the frozen config exactly and
stay comparable to our validation numbers. Deployment (Day 6) may refit on train+validation.

Structure:
  - load_config          : read the frozen final-model config
  - fit_and_predict_all  : fit tuned model on train, return val + test probabilities
  - calibrate_test       : fit per-region isotonic on validation, apply to test
  - (metrics helpers)    : ranking / threshold / tier scoring
  - main                 : evaluate on test, print val-vs-test, log to MLflow
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, fbeta_score, brier_score_loss)

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, TARGET, REGIONS, TRACKING_URI,
)
from train_xgboost import build_xgb_model
from tune_thresholds import BEST_VARIANT, SCALE_POS_WEIGHT, BETA
from risk_tiers import calibrate_per_region, assign_tiers, tier_summary


CONFIG_PATH = Path(__file__).parent.parent / "config" / "final_model.json"


def load_config(path=CONFIG_PATH):
    """Read the frozen final-model config (thresholds, base rates, high_mult, ...)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fit_and_predict_all():
    """Fit the tuned model on TRAIN; return (val, val_proba, test, test_proba)."""
    parts = split_data(load_features())
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, BEST_VARIANT)
    fc = numeric + categorical

    model = build_xgb_model(build_preprocessor(numeric, categorical), SCALE_POS_WEIGHT)
    model.fit(parts["train"][fc], parts["train"][TARGET])

    val, test = parts["validate"], parts["test"]
    return val, model.predict_proba(val[fc])[:, 1], test, model.predict_proba(test[fc])[:, 1]


def calibrate_test(val, val_proba, test, test_proba):
    """Fit per-region isotonic calibrators on VALIDATION, apply them to TEST probabilities."""
    _, calibrators = calibrate_per_region(val, val_proba)   # fitted per region on validation
    cal = np.empty(len(test))
    reg = test["region"].to_numpy()
    for r in REGIONS:
        m = reg == r
        cal[m] = calibrators[r].predict(test_proba[m])
    return cal


def ranking_metrics(df, proba):
    """PR-AUC + ROC-AUC overall and per region (ranking is identical for raw vs calibrated)."""
    y = df[TARGET].to_numpy()
    reg = df["region"].to_numpy()
    out = {"pr_auc": average_precision_score(y, proba), "roc_auc": roc_auc_score(y, proba)}
    for r in REGIONS:
        m = reg == r
        if m.any() and len(set(y[m])) > 1:
            out[f"pr_auc_{r}"] = average_precision_score(y[m], proba[m])
    return out


def alert_metrics(df, proba, thresholds):
    """Per-region precision/recall/F2 at the frozen F2 thresholds (applied to RAW proba)."""
    y = df[TARGET].to_numpy()
    reg = df["region"].to_numpy()
    rows = []
    for r in REGIONS:
        m = reg == r
        pred = (proba[m] >= thresholds[r]).astype(int)
        rows.append({"region": r, "threshold": thresholds[r],
                     "precision": round(precision_score(y[m], pred, zero_division=0), 3),
                     "recall": round(recall_score(y[m], pred, zero_division=0), 3),
                     "f2": round(fbeta_score(y[m], pred, beta=BETA, zero_division=0), 3)})
    return pd.DataFrame(rows)


def main():
    """Evaluate the frozen model once on TEST, print val-vs-test, and log to MLflow."""
    config = load_config()
    thresholds, base_rates, high_mult = config["thresholds"], config["base_rates"], config["high_mult"]

    val, val_proba, test, test_proba = fit_and_predict_all()
    cal_test = calibrate_test(val, val_proba, test, test_proba)

    val_rank = ranking_metrics(val, val_proba)
    test_rank = ranking_metrics(test, test_proba)

    print("=== FINAL TEST EVALUATION (sealed 2022-26) ===\n")
    print("Ranking — validation vs test:")
    rank_rows = [{"metric": k, "validation": round(val_rank.get(k, float("nan")), 4),
                  "test": round(test_rank.get(k, float("nan")), 4),
                  "diff": round(test_rank.get(k, float("nan")) - val_rank.get(k, float("nan")), 4)}
                 for k in ["pr_auc", "roc_auc"] + [f"pr_auc_{r}" for r in REGIONS]]
    print(pd.DataFrame(rank_rows).to_string(index=False))

    print("\nAlerts at frozen F2 thresholds (test):")
    alerts = alert_metrics(test, test_proba, thresholds)
    print(alerts.to_string(index=False))

    print("\nCalibrated risk tiers (test):")
    test_tiers = assign_tiers(test, cal_test, base_rates, high_mult)
    tiers = tier_summary(test_tiers)
    print(tiers.to_string(index=False))

    test_brier = brier_score_loss(test[TARGET].to_numpy(), cal_test)
    print(f"\nCalibrated Brier (test): {test_brier:.4f}")

    # --- Log the honest final numbers to MLflow, tied to the registered model ---
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("earthquake-xgboost")
    with mlflow.start_run(run_name="final-test-eval"):
        mlflow.log_params({"registered_model": "earthquake-risk-model", "version": 1,
                           "config": "config/final_model.json", "eval_split": "test(2022-26)"})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_rank.items()})
        mlflow.log_metric("test_brier", test_brier)
        for _, row in alerts.iterrows():
            mlflow.log_metrics({f"test_precision_{row['region']}": row["precision"],
                                f"test_recall_{row['region']}": row["recall"],
                                f"test_f2_{row['region']}": row["f2"]})
    print("\nLogged final test metrics to MLflow run 'final-test-eval'.")


# Only run when launched directly (e.g. "python src/evaluate_test.py").
if __name__ == "__main__":
    main()