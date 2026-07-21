"""
tune_weighting.py — Per-region class weighting for the region-blind model
(Week 4, Day 3, experiment #5).

Class weighting tells the model to pay extra attention to the rare "quake-coming" days.
We currently use one GLOBAL balanced weight (train negatives/positives ≈ 3.04). This script
asks whether a PER-REGION weight helps: California's quake-weeks are rare (~14%) and Japan's
common (~40%), so a single global weight lets Japan's frequent positives dominate training and
may drown out California's patterns. Per-region weighting up-weights each region's positives by
its OWN imbalance.

We compare three schemes on the region-blind model and judge by validation PR-AUC (overall +
per region) — the whole precision-recall curve, so we see if weighting actually LIFTS the model
rather than just sliding along the curve (which the per-region thresholds already do).
All fitting is on TRAIN only; TEST stays sealed.

Structure:
  - region_sample_weights : per-region class-balancing weight for each train row
  - fit_eval              : fit under one weighting scheme, score on validation
  - main                  : run all three schemes and print the comparison
"""

import numpy as np
import pandas as pd

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, evaluate, TARGET, REGIONS,
)
from train_xgboost import build_xgb_model


# The Day-2 winning feature set: 6 recent-activity features, no region, no depth.
BEST_VARIANT = "pruned_nodepth_noreg"


def region_sample_weights(train):
    """
    Per-region class-balancing weights for each TRAIN row.

    A positive ("quake coming") row is weighted by its OWN region's negative/positive
    ratio, so within each region positives and negatives carry equal total weight;
    negative rows stay at weight 1. This makes rare-positive regions (California) count
    as much as common-positive ones (Japan) during training. Computed on train only.

    Args:
        train (pd.DataFrame): the training split (needs 'region' and the target).

    Returns:
        np.ndarray: a weight per training row, aligned to train's order.
    """
    w = np.ones(len(train))
    y = train[TARGET].to_numpy()
    reg = train["region"].to_numpy()
    for r in REGIONS:
        m = reg == r
        n_pos = int((y[m] == 1).sum())
        n_neg = int((y[m] == 0).sum())
        ratio = n_neg / n_pos if n_pos else 1.0     # how many negatives per positive here
        w[m & (y == 1)] = ratio                      # up-weight this region's positives to match
    return w


def fit_eval(scheme, variant=BEST_VARIANT):
    """
    Fit the region-blind model under one weighting scheme and score it on validation.

    Args:
        scheme (str): 'none' (scale_pos_weight=1), 'global' (one balanced weight from
            train), or 'region' (per-region sample weights from region_sample_weights).
        variant (str): the feature-set variant. Defaults to the Day-2 winner.

    Returns:
        dict: validation metrics from train_baseline.evaluate (PR-AUC overall + per region).
    """
    parts = split_data(load_features())
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, variant)
    feature_cols = numeric + categorical
    y_train = parts["train"][TARGET]

    # One global balanced weight = train negatives / positives (matches train_xgboost).
    n_pos = int((y_train == 1).sum())
    global_spw = ((y_train == 0).sum() / n_pos) if n_pos else 1.0

    # Map each scheme to (scale_pos_weight, per-row sample_weight). 'region' uses sample
    # weights instead of scale_pos_weight, so scale_pos_weight stays 1 (no double-counting).
    if scheme == "none":
        spw, sample_weight = 1.0, None
    elif scheme == "global":
        spw, sample_weight = global_spw, None
    elif scheme == "region":
        spw, sample_weight = 1.0, region_sample_weights(parts["train"])
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    model = build_xgb_model(build_preprocessor(numeric, categorical), scale_pos_weight=spw)
    # Pipeline routes 'clf__sample_weight' to the XGBoost step's fit.
    fit_kwargs = {} if sample_weight is None else {"clf__sample_weight": sample_weight}
    model.fit(parts["train"][feature_cols], y_train, **fit_kwargs)

    return evaluate(model, parts["validate"], feature_cols)


def main():
    """Compare no / global / per-region class weighting by validation PR-AUC."""
    print(f"Class-weighting comparison for the region-blind model ('{BEST_VARIANT}')\n")

    rows = []
    for scheme in ["none", "global", "region"]:
        m = fit_eval(scheme)
        rows.append({
            "scheme": scheme,
            "pr_auc": round(m["pr_auc"], 4),
            "pr_auc_ca": round(m.get("pr_auc_california", float("nan")), 4),
            "pr_auc_jp": round(m.get("pr_auc_japan", float("nan")), 4),
            "pr_auc_gr": round(m.get("pr_auc_greece", float("nan")), 4),
        })

    print("Validation PR-AUC by weighting scheme (overall + per region):")
    print(pd.DataFrame(rows).to_string(index=False))


# Only run when launched directly (e.g. "python src/tune_weighting.py").
if __name__ == "__main__":
    main()