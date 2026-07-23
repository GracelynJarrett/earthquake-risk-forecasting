"""
tune_hyperparams.py — Regularization + learning-rate search for the region-blind model
(Week 4, Day 5).

Reads the fixed settings and search grid from config/gxboost_tunning.yaml, then tries every
combination: fit on TRAIN, score PR-AUC on VALIDATION (no CV shuffling — same honest temporal
split, so no leakage). Goal: shrink the train-vs-validation overfitting gap and lift honest
PR-AUC, with a close eye on whether Japan's ranking improves.

Uses the model we actually adopted: the lean region-blind feature set, UNWEIGHTED (scale_pos_weight=1).

Structure:
  - load_tuning_config : read base_params + search_grid from the YAML
  - fit_score          : fit one param combo on train, score PR-AUC (overall + per region)
  - search             : try every combo, return results sorted by validation PR-AUC
  - main               : run the search and print the leaderboard + best combo
"""

import itertools
from pathlib import Path

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, evaluate, TARGET, REGIONS, TRACKING_URI,
)
from tune_thresholds import BEST_VARIANT, SCALE_POS_WEIGHT   # lean feature set, unweighted


CONFIG_PATH = Path(__file__).parent.parent / "config" / "gxboost_tunning.yaml"

# Where the runs land: all combos in the tuning experiment; the winner in the main one.
MAIN_EXPERIMENT = "earthquake-xgboost"            # the winning tuned model (registerable Day 5)
TUNING_EXPERIMENT = "earthquake-xgboost-tuning"   # the full 108-combo search record


def load_tuning_config(path=CONFIG_PATH):
    """Read base_params (fixed) + search_grid (knobs to sweep) from the tuning YAML."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["base_params"], cfg["search_grid"]


def fit_score(params, base_params, parts, numeric, categorical):
    """
    Fit XGBoost (base_params + this combo) on TRAIN; return train/validation PR-AUC.

    Uses SCALE_POS_WEIGHT=1 (our adopted unweighted model) so tuning matches the model we
    carry forward. Returns per-region validation PR-AUC too, so we can watch Japan.

    Returns:
        dict: {train_pr_auc, val_pr_auc, val_california, val_japan, val_greece}.
    """
    fc = numeric + categorical
    clf = XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, **base_params, **params)
    model = Pipeline([("pre", build_preprocessor(numeric, categorical)), ("clf", clf)])
    model.fit(parts["train"][fc], parts["train"][TARGET])

    res = {
        "train_pr_auc": average_precision_score(
            parts["train"][TARGET], model.predict_proba(parts["train"][fc])[:, 1]),
        "val_pr_auc": average_precision_score(
            parts["validate"][TARGET], model.predict_proba(parts["validate"][fc])[:, 1]),
    }
    val = parts["validate"]
    proba = model.predict_proba(val[fc])[:, 1]
    y = val[TARGET].to_numpy()
    reg = val["region"].to_numpy()
    for r in REGIONS:
        m = reg == r
        if m.any() and len(set(y[m])) > 1:
            res[f"val_{r}"] = average_precision_score(y[m], proba[m])
    return res


def search(variant=BEST_VARIANT):
    """
    Try every combination in the grid; log each to MLflow; return (results, best_params).

    Loads data + config once, iterates the Cartesian product of the search grid, logs each
    combo's params + metrics to the TUNING_EXPERIMENT, and records validation PR-AUC, the
    overfit gap (train - val), and per-region validation PR-AUC (so Japan is visible).

    Returns:
        tuple[pd.DataFrame, dict]: (results sorted by val PR-AUC, best combo's raw params).
    """
    base_params, grid = load_tuning_config()
    parts = split_data(load_features())
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, variant)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(TUNING_EXPERIMENT)

    keys = list(grid)
    rows, combos = [], []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        res = fit_score(params, base_params, parts, numeric, categorical)

        # Log this combo (params + metrics only — no model artifact, to keep the search light).
        run_name = f"lr{params['learning_rate']}_d{params['max_depth']}_mcw{params['min_child_weight']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({**base_params, **params, "variant": variant,
                               "scale_pos_weight": SCALE_POS_WEIGHT})
            mlflow.log_metrics({"val_pr_auc": res["val_pr_auc"], "train_pr_auc": res["train_pr_auc"],
                                "gap": res["train_pr_auc"] - res["val_pr_auc"],
                                **{f"val_pr_auc_{r}": res[f"val_{r}"] for r in REGIONS if f"val_{r}" in res}})

        combos.append((params, res))
        rows.append({
            **params,
            "val_pr_auc": round(res["val_pr_auc"], 4),
            "gap": round(res["train_pr_auc"] - res["val_pr_auc"], 4),   # overfitting gauge
            "val_ca": round(res.get("val_california", float("nan")), 4),
            "val_jp": round(res.get("val_japan", float("nan")), 4),
            "val_gr": round(res.get("val_greece", float("nan")), 4),
        })

    results = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best_params = max(combos, key=lambda cr: cr[1]["val_pr_auc"])[0]   # raw types, for refit
    return results, best_params


def log_best_model(best_params, variant=BEST_VARIANT):
    """Refit the winning combo and log it (with the fitted model) to the main experiment."""
    base_params, _ = load_tuning_config()
    parts = split_data(load_features())
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, variant)
    fc = numeric + categorical

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MAIN_EXPERIMENT)
    clf = XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, **base_params, **best_params)
    model = Pipeline([("pre", build_preprocessor(numeric, categorical)), ("clf", clf)])

    with mlflow.start_run(run_name="tuned-xgboost"):
        model.fit(parts["train"][fc], parts["train"][TARGET])
        train_m = evaluate(model, parts["train"], fc)
        val_m = evaluate(model, parts["validate"], fc)
        mlflow.log_params({**base_params, **best_params, "variant": variant, "model": "xgboost_tuned",
                           "scale_pos_weight": SCALE_POS_WEIGHT, "features": ", ".join(fc)})
        mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
        # cloudpickle serialization avoids MLflow 3.x's skops "untrusted types" check.
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")


def main():
    """Run the hyperparameter search and print the leaderboard + best combo."""
    print(f"Hyperparameter search (region-blind, unweighted) — reading {CONFIG_PATH.name}")
    print("Baseline to beat: val PR-AUC 0.366 (train ~0.61, gap ~0.24); Japan 0.29\n")

    results, best_params = search()
    print(f"Ran {len(results)} combos. Top 12 by validation PR-AUC:")
    print(results.head(12).to_string(index=False))

    print("\nBest combo:")
    print(results.iloc[0].to_string())

    log_best_model(best_params)
    print(f"\nLogged all combos to '{TUNING_EXPERIMENT}' and the winner to '{MAIN_EXPERIMENT}'.")


# Only run when launched directly (e.g. "python src/tune_hyperparams.py").
if __name__ == "__main__":
    main()