"""
tune_thresholds.py — Per-region decision thresholds for the region-blind model
(Week 4, Day 3, experiment #4).

The model outputs a probability of a region-significant quake in the next 7 days. Turning
that into a yes/no alert needs a cutoff — and one global 0.5 cutoff isn't fair across regions
with very different base rates (Japan 40% vs California 15%). This script fits the chosen model
on TRAIN, then for each region separately sweeps the cutoff over the VALIDATION probabilities and
picks the one that maximizes F2 (favor recall ~2x precision — a missed quake is the costly error
for first responders, while F2 still penalizes false-alarm 'alarm fatigue').

Thresholds are fitted on VALIDATION only; TEST stays sealed until the final model is locked (Day 5).

Structure:
  - fit_and_predict : fit the region-blind model on train, return validation probabilities
  - sweep_region    : sweep cutoffs for one region, return the metric curve + best-F2 threshold
  - tune_all        : do every region, build the summary vs the default 0.5 cutoff
  - main            : run and print the per-region threshold table
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, TARGET, REGIONS,
)
from train_xgboost import build_xgb_model


# The Day-2 winning feature set: 6 recent-activity features, no region, no depth.
BEST_VARIANT = "pruned_nodepth_noreg"

# Favor recall 2x precision (a missed quake costs more than a false alarm).
BETA = 2

# No class weighting: Day-3 found it inflates the probabilities (validation Brier 0.257 vs
# 0.204) without improving ranking (PR-AUC ~equal), so unweighted gives honest, far better-
# calibrated risk probabilities — the right basis for per-region thresholds and risk tiers.
SCALE_POS_WEIGHT = 1.0


def fit_and_predict(variant=BEST_VARIANT):
    """
    Fit the chosen model on TRAIN and return validation rows + predicted probabilities.

    Fits both preprocessing and the trees on train only (no leakage), using the same
    balanced positive-class weight as train_xgboost. Returns the validation split so we
    can slice it by region for per-region threshold tuning.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: (validation_df, proba) where proba is
        P(region-significant quake in next 7 days) for each validation row.
    """
    parts = split_data(load_features())
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, variant)
    feature_cols = numeric + categorical

    # No class weighting (see SCALE_POS_WEIGHT note) — honest, well-calibrated probabilities.
    y_train = parts["train"][TARGET]
    model = build_xgb_model(build_preprocessor(numeric, categorical),
                            scale_pos_weight=SCALE_POS_WEIGHT)
    model.fit(parts["train"][feature_cols], y_train)

    val = parts["validate"]
    proba = model.predict_proba(val[feature_cols])[:, 1]
    return val, proba


def sweep_region(y, proba, beta=BETA, grid=None):
    """
    Sweep decision thresholds for ONE region and score each with precision/recall/F1/F2.

    For every candidate cutoff, label a day "alert" if proba >= cutoff, then compute the
    metrics against the true labels. The best threshold is the one with the highest F2
    (ties broken toward the higher cutoff, i.e. fewer false alarms).

    Args:
        y (array-like): true 0/1 labels for this region's validation rows.
        proba (array-like): predicted P(quake) for the same rows.
        beta (float): recall weighting for F-beta (2 = recall twice as important).
        grid (np.ndarray | None): candidate thresholds. Defaults to 0.005..0.995 step 0.005.

    Returns:
        tuple[float, pd.DataFrame]: (best_threshold, curve) where curve has one row per
        candidate threshold with columns [threshold, precision, recall, f1, f2].
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    if grid is None:
        grid = np.arange(0.005, 1.0, 0.005)   # fine sweep of possible cutoffs

    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "f2": fbeta_score(y, pred, beta=beta, zero_division=0),
        })
    curve = pd.DataFrame(rows)

    # Best cutoff = highest F2; among ties pick the HIGHER threshold (fewer false alarms).
    best_f2 = curve["f2"].max()
    best_threshold = curve.loc[curve["f2"] >= best_f2 - 1e-12, "threshold"].max()
    return float(best_threshold), curve


def pick_threshold(curve, kind="fbeta", value=BETA):
    """
    Choose a threshold from a metric curve under a given strategy.

    Strategies:
      - kind="fixed": use `value` as-is (e.g. the default 0.5 baseline).
      - kind="fbeta": maximize F-beta (value = beta; 2 = favor recall 2x, 1.5 = milder).
      - kind="precision_floor": among cutoffs with precision >= value, take the highest
        recall (ties -> higher threshold, fewer alarms). Falls back to the max-precision
        cutoff if no threshold reaches the floor.

    Args:
        curve (pd.DataFrame): from sweep_region — columns threshold/precision/recall/f1/f2.
        kind (str): "fixed", "fbeta", or "precision_floor".
        value (float): the cutoff (fixed), beta (fbeta), or precision floor (precision_floor).

    Returns:
        float: the chosen threshold.
    """
    c = curve
    if kind == "fixed":
        return float(value)
    if kind == "fbeta":
        b2 = value * value
        # F-beta straight from precision & recall (works for any beta, no re-scoring).
        denom = (b2 * c["precision"] + c["recall"]).replace(0, np.nan)
        fbeta = (1 + b2) * c["precision"] * c["recall"] / denom
        best = fbeta.max()
        return float(c.loc[fbeta >= best - 1e-12, "threshold"].max())
    if kind == "precision_floor":
        ok = c[c["precision"] >= value]
        if ok.empty:                          # no cutoff meets the floor -> best precision available
            return float(c.loc[c["precision"].idxmax(), "threshold"])
        best_recall = ok["recall"].max()
        return float(ok.loc[ok["recall"] >= best_recall - 1e-12, "threshold"].max())
    raise ValueError(f"unknown kind: {kind}")


# The threshold strategies we compare: the 0.5 baseline plus the three candidates.
STRATEGIES = [
    ("default(0.5)", "fixed",           0.5),
    ("F2",           "fbeta",           2),
    ("F1.5",         "fbeta",           1.5),
    ("prec>=0.35",   "precision_floor", 0.35),
]


def tune_all(strategies=STRATEGIES, variant=BEST_VARIANT):
    """
    Fit once, then for each region + strategy pick a threshold and score it on validation.

    Returns:
        tuple[pd.DataFrame, dict]: (summary, thresholds) where thresholds[strategy][region]
        = chosen cutoff, and summary has one row per (region, strategy) with the metrics.
    """
    val, proba = fit_and_predict(variant)

    rows = []
    thresholds = {name: {} for name, _, _ in strategies}
    for region in REGIONS:
        m = (val["region"] == region).to_numpy()
        y = val.loc[m, TARGET].to_numpy()
        p = proba[m]
        _, curve = sweep_region(y, p)                      # one curve per region, reused by all strategies
        for name, kind, value in strategies:
            t = pick_threshold(curve, kind, value)
            thresholds[name][region] = t
            pred = (p >= t).astype(int)
            rows.append({
                "region": region, "strategy": name, "threshold": round(t, 3),
                "precision": round(precision_score(y, pred, zero_division=0), 3),
                "recall": round(recall_score(y, pred, zero_division=0), 3),
                "f1": round(f1_score(y, pred, zero_division=0), 3),
                "f2": round(fbeta_score(y, pred, beta=2, zero_division=0), 3),
            })
    return pd.DataFrame(rows), thresholds


def main():
    """Compare per-region threshold strategies on validation and print the table."""
    print(f"Per-region threshold strategies for the region-blind model ('{BEST_VARIANT}')\n")

    summary, thresholds = tune_all()

    # Grouped by region, strategies in the order defined above (baseline first).
    print("Per-region validation metrics by strategy:")
    print(summary.to_string(index=False))

    print("\nChosen thresholds by strategy:")
    for name in thresholds:
        vals = "  ".join(f"{r}={t:.3f}" for r, t in thresholds[name].items())
        print(f"  {name:<12} {vals}")


# Only run when launched directly (e.g. "python src/tune_thresholds.py").
if __name__ == "__main__":
    main()