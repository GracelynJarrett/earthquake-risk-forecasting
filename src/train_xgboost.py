"""
train_xgboost.py — Train the Week 4 XGBoost model (Day 1).

Same honest scaffolding as train_baseline.py — read the leakage-safe feature table,
split BY DATE, fit on TRAIN only, score on VALIDATION, log to MLflow, leave TEST
sealed — but swaps the straight-line logistic regression for gradient-boosted
decision trees (XGBoost), which capture non-linear patterns and feature interactions
the baseline cannot.

To avoid copy-paste drift, this script IMPORTS the loaders, date-splitter, feature
resolver, preprocessor, and metric function straight from train_baseline.py. The
only real change is the model engine.

Structure:
  - build_xgb_model : the preprocess -> XGBoost classifier pipeline
  - train_and_log   : fit on train, score train + validation, log to MLflow
  - main            : run it on the chosen feature set and print the comparison
"""

# pandas: assemble the small validation-results table printed at the end.
import pandas as pd
# sys: read an optional variant name from the command line (defaults to the winner).
import sys

# scikit-learn Pipeline: chains preprocessing + model so fitting on train fits BOTH
# on train only (the same leakage guard the baseline uses).
from sklearn.pipeline import Pipeline

# MLflow: experiment tracking — log this run so it sits beside the baseline runs.
import mlflow
import mlflow.sklearn

# XGBoost's scikit-learn-style classifier — behaves like any sklearn model
# (.fit / .predict_proba), so it drops straight into the existing pipeline.
from xgboost import XGBClassifier

# Reuse the baseline's tested building blocks — one source of truth, no drift:
# read the feature table, split by date, resolve a variant's columns from the YAML,
# build the impute + one-hot preprocessor, score metrics, and shared constants.
from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, evaluate, TARGET, REGIONS, TRACKING_URI,
)


# The Week 3 winning feature set we carry into Week 4 (base recent-activity + depth).
VARIANT = "depth"

# A separate MLflow experiment so the XGBoost runs group on their own, yet still sit
# in the same dashboard as the "earthquake-baseline" logistic runs for comparison.
EXPERIMENT_NAME = "earthquake-xgboost"


def build_xgb_model(preprocessor, scale_pos_weight=1.0):
    """
    Full model pipeline: preprocess -> XGBoost classifier.

    Fitting this on the train split fits BOTH the preprocessing (impute medians +
    one-hot region) and the trees on train only — the same no-leakage guarantee as
    the baseline. Trees don't need feature scaling, but we reuse the baseline's
    preprocessor unchanged so the XGBoost-vs-logistic comparison stays apples-to-
    apples (the scaling step is simply harmless for trees).

    Args:
        preprocessor (ColumnTransformer): from build_preprocessor() in train_baseline.
        scale_pos_weight (float): imbalance knob — extra weight for the rare
            "a big quake is coming" class. 1.0 = no adjustment; setting it to
            (#negatives / #positives) on train mirrors the baseline's
            class_weight="balanced". Defaults to 1.0; train_and_log() passes the
            balanced value.

    Returns:
        Pipeline: an unfitted preprocess + XGBoost pipeline.
    """
    return Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300,        # how many trees to build in sequence
            max_depth=4,             # keep each tree shallow -> less overfitting
            learning_rate=0.05,      # small steps -> steadier, more accurate learning
            subsample=0.8,           # each tree sees 80% of rows -> regularization
            colsample_bytree=0.8,    # each tree sees 80% of columns -> regularization
            eval_metric="aucpr",     # score by area under precision-recall (imbalance-aware)
            scale_pos_weight=scale_pos_weight,  # up-weight the rare positive class
            random_state=42,         # reproducible runs
            n_jobs=-1,               # use all CPU cores
        )),
    ])


def train_and_log(parts, variant, cfg):
    """
    Fit XGBoost on train, evaluate on train + validation, and log to MLflow.

    Mirrors train_baseline.train_and_log so the two models are logged the same way
    and compare cleanly in the dashboard. The one addition: scale_pos_weight is
    computed from the TRAIN split (negatives / positives) so the rare positive class
    is balanced using train information only — no peeking at validation or test.

    Args:
        parts (dict): the train/validate/test split from split_data().
        variant (str): which feature-set variant to run (a key in the YAML).
        cfg (dict): the loaded variants config from load_variants().

    Returns:
        tuple[dict, dict]: (train_metrics, validation_metrics).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Resolve this variant's feature columns straight from the YAML config.
    numeric, categorical = feature_columns(cfg, variant)
    feature_cols = numeric + categorical

    # Balance the rare positive class the XGBoost way: weight it by how many more
    # negatives than positives there are IN TRAIN (e.g. 3 negatives per positive -> 3).
    y_train = parts["train"][TARGET]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / n_pos if n_pos else 1.0

    model = build_xgb_model(build_preprocessor(numeric, categorical), scale_pos_weight=spw)

    with mlflow.start_run(run_name=f"{variant}-xgboost"):
        # Fit on TRAIN ONLY, then score train (overfit check) and validation.
        model.fit(parts["train"][feature_cols], parts["train"][TARGET])
        train_metrics = evaluate(model, parts["train"], feature_cols)
        val_metrics = evaluate(model, parts["validate"], feature_cols)

        # Record exactly which model, variant, features, and settings this run used,
        # so the dashboard answers "what was in this run?" at a glance.
        clf = model.named_steps["clf"]
        mlflow.log_params({
            "model": "xgboost",
            "variant": variant,
            "description": cfg["variants"][variant]["description"],
            "features": ", ".join(feature_cols),
            "n_features": len(feature_cols),
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "learning_rate": clf.learning_rate,
            "subsample": clf.subsample,
            "colsample_bytree": clf.colsample_bytree,
            "scale_pos_weight": round(spw, 3),
        })
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        # cloudpickle serialization avoids MLflow 3.x's skops "untrusted types" check.
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    return train_metrics, val_metrics


def main():
    """
    Train XGBoost on the chosen feature set and print a train-vs-validation summary.

    Usage:
        python src/train_xgboost.py           # run the winning feature set ('depth')
        python src/train_xgboost.py base      # run a different variant by name
    """
    parts = split_data(load_features())
    cfg = load_variants()

    # Which variant to run: the one named on the command line, else the winner.
    variant = sys.argv[1] if len(sys.argv) > 1 else VARIANT

    print(f"Training XGBoost on variant '{variant}' "
          f"({cfg['variants'][variant]['description']})\n")
    train_metrics, val_metrics = train_and_log(parts, variant, cfg)

    # Print train vs validation side by side — a quick read on performance AND
    # overfitting (a big train >> validation gap = the trees memorized the past).
    summary = pd.DataFrame({"train": train_metrics, "validation": val_metrics})
    print("Train vs. validation metrics:")
    print(summary.to_string())


# Only run when launched directly (e.g. "python src/train_xgboost.py").
if __name__ == "__main__":
    main()
