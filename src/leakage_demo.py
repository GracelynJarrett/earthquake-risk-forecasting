"""
leakage_demo.py — Cautionary demo: what random (non-temporal) splitting does to the metrics
(Week 4, experiment #2).

Our honest pipeline splits BY DATE (train <=2018, validate 2019-21) so the model is always
tested on the future. This shows what happens if we split RANDOMLY instead — the shortcut that
inflates earthquake models. Same features, same model, same split sizes; ONLY the split changes.
Random splitting scatters near-duplicate neighboring days and clustered aftershock sequences
across train and validation, so the model is effectively tested on data it already saw.

This shuffles ALL the data (train+validate+test, 2000-2026) before re-splitting, mirroring a
fully leaky pipeline from the start. The demo model is THROWAWAY: our real model is never trained
on test rows, so our honest Day-5 test evaluation is unaffected — we only read the inflated number.

Structure:
  - fit_and_score : fit the chosen model on one train, score PR-AUC on one validation
  - main          : shuffle ALL data, re-split randomly, print temporal-vs-random comparison
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, log_loss, brier_score_loss)

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, TARGET, REGIONS, TRACKING_URI,
)
from train_xgboost import build_xgb_model
from tune_thresholds import BEST_VARIANT, SCALE_POS_WEIGHT   # our chosen model: lean, unweighted


def fit_and_score(train_df, val_df, variant=BEST_VARIANT):
    """
    Fit the chosen model on train_df and return validation metrics.

    Identical model to the rest of Week 4 — only the rows in train_df/val_df differ between
    the temporal and random calls, so any metric gap is caused purely by the split method.

    Returns:
        tuple[Pipeline, dict, dict]: (fitted model, overall metrics, per-region PR-AUC).
        Overall includes ranking metrics (pr_auc, roc_auc), threshold metrics at 0.5
        (f1, precision, recall), and probabilistic metrics (log_loss, brier).
    """
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, variant)
    fc = numeric + categorical

    model = build_xgb_model(build_preprocessor(numeric, categorical),
                            scale_pos_weight=SCALE_POS_WEIGHT)
    model.fit(train_df[fc], train_df[TARGET])

    proba = model.predict_proba(val_df[fc])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = val_df[TARGET].to_numpy()
    reg = val_df["region"].to_numpy()

    overall = {
        "pr_auc": average_precision_score(y, proba),            # higher better
        "roc_auc": roc_auc_score(y, proba),                     # higher better
        "f1": f1_score(y, pred, zero_division=0),               # higher better
        "precision": precision_score(y, pred, zero_division=0), # higher better
        "recall": recall_score(y, pred, zero_division=0),       # higher better
        "log_loss": log_loss(y, proba, labels=[0, 1]),          # LOWER better
        "brier": brier_score_loss(y, proba),                    # LOWER better
    }
    region_pr = {}
    for r in REGIONS:
        m = reg == r
        if m.any() and len(set(y[m])) > 1:
            region_pr[r] = average_precision_score(y[m], proba[m])
    return model, overall, region_pr


def main():
    """Compare the honest temporal split vs a random split (leakage demo)."""
    parts = split_data(load_features())
    temporal_train, temporal_val = parts["train"], parts["validate"]

    # Shuffle ALL the data (train+validate+test, 2000-2026), then RANDOMLY re-split into the
    # SAME train/validate sizes as the temporal split. This is the fully-leaky scenario.
    pool = pd.concat([parts["train"], parts["validate"], parts["test"]], ignore_index=True)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(pool))
    n_train, n_val = len(temporal_train), len(temporal_val)
    rand_train = pool.iloc[order[:n_train]]
    rand_val = pool.iloc[order[n_train:n_train + n_val]]

    print("Experiment #2 — honest temporal split vs. RANDOM split over ALL data (leakage demo)")
    print(f"(same model + features + sizes: train={n_train:,}, validate={n_val:,}; "
          f"shuffled pool={len(pool):,})\n")

    temp_model, temp_overall, temp_region = fit_and_score(temporal_train, temporal_val)
    rand_model, rand_overall, rand_region = fit_and_score(rand_train, rand_val)

    # Which direction is "better" for each metric (so the diff reads correctly).
    better = {"pr_auc": "higher", "roc_auc": "higher", "f1": "higher", "precision": "higher",
              "recall": "higher", "log_loss": "lower", "brier": "lower"}
    rows = [{"metric": k, "better": better[k],
             "temporal": round(temp_overall[k], 4),
             "random": round(rand_overall[k], 4),
             "diff": round(rand_overall[k] - temp_overall[k], 4)} for k in temp_overall]
    print("Overall metrics (temporal honest vs random leaky):")
    print(pd.DataFrame(rows).to_string(index=False))

    rrows = [{"region": r, "temporal": round(temp_region.get(r, float("nan")), 4),
              "random": round(rand_region.get(r, float("nan")), 4),
              "inflation": round(rand_region.get(r, float("nan")) - temp_region.get(r, float("nan")), 4)}
             for r in REGIONS]
    print("\nPer-region PR-AUC (higher=better):")
    print(pd.DataFrame(rrows).to_string(index=False))

    # Log both split models to a clearly-labeled demo experiment (NOT the real experiments).
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("earthquake-leakage-demo")
    for split_type, model, overall, region in [
        ("temporal_honest", temp_model, temp_overall, temp_region),
        ("random_leaky", rand_model, rand_overall, rand_region),
    ]:
        with mlflow.start_run(run_name=split_type):
            mlflow.log_params({"split_type": split_type, "variant": BEST_VARIANT,
                               "scale_pos_weight": SCALE_POS_WEIGHT,
                               "n_train": n_train, "n_val": n_val})
            mlflow.log_metrics(overall)
            mlflow.log_metrics({f"pr_auc_{r}": v for r, v in region.items()})
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    print("\nLogged both split models to MLflow experiment 'earthquake-leakage-demo'.")


# Only run when launched directly (e.g. "python src/leakage_demo.py").
if __name__ == "__main__":
    main()