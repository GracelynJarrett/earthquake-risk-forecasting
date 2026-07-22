"""
leakage_demo.py — Cautionary demo: what random (non-temporal) splitting does to the metrics
(Week 4, experiment #2).

Our honest pipeline splits BY DATE (train <=2018, validate 2019-21) so the model is always
tested on the future. This shows what happens if we split RANDOMLY instead — the shortcut that
inflates earthquake models. Same features, same model, same split sizes; ONLY the split changes.
Random splitting scatters near-duplicate neighboring days and clustered aftershock sequences
across train and validation, so the model is effectively tested on data it already saw.

TEST (2022-26) is NOT touched — the demo re-partitions only the train+validate pool.

Structure:
  - fit_and_score : fit the chosen model on one train, score PR-AUC on one validation
  - main          : build the random re-split and print temporal-vs-random comparison
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, TARGET, REGIONS,
)
from train_xgboost import build_xgb_model
from tune_thresholds import BEST_VARIANT, SCALE_POS_WEIGHT   # our chosen model: lean, unweighted


def fit_and_score(train_df, val_df, variant=BEST_VARIANT):
    """
    Fit the chosen model on train_df and return validation PR-AUC (overall + per region).

    Identical model to the rest of Week 4 — only the rows in train_df/val_df differ between
    the temporal and random calls, so any metric gap is caused purely by the split method.

    Returns:
        dict: {'pr_auc': overall, '<region>': per-region PR-AUC}.
    """
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, variant)
    fc = numeric + categorical

    model = build_xgb_model(build_preprocessor(numeric, categorical),
                            scale_pos_weight=SCALE_POS_WEIGHT)
    model.fit(train_df[fc], train_df[TARGET])

    proba = model.predict_proba(val_df[fc])[:, 1]
    y = val_df[TARGET].to_numpy()
    reg = val_df["region"].to_numpy()

    out = {"pr_auc": average_precision_score(y, proba)}
    for r in REGIONS:
        m = reg == r
        if m.any() and len(set(y[m])) > 1:
            out[r] = average_precision_score(y[m], proba[m])
    return out