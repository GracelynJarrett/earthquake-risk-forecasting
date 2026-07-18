"""
train_baseline.py — Train the baseline logistic-regression model (Week 3, Day 3).

Loads the leakage-safe feature table (built by build_features.py, stored in
earthquakes.db) and trains a logistic regression for a chosen VARIANT — the feature
set is read from config/model_variants.yaml, so every run records exactly which
features it used. All preprocessing and model fitting use the TRAIN split only; the
model is evaluated on the VALIDATION split; PR-AUC, F1, and log-loss are logged to
MLflow. The TEST split is left untouched until every modeling decision is final.

Structure:
  - load_features / split_data       : read the table, split by date
  - load_variants / feature_columns  : resolve a variant's features from the YAML
  - build_preprocessor / build_model : train-only-fit preprocessing + logistic reg
  - evaluate / train_and_log         : score (PR-AUC, F1, log-loss) and log to MLflow
"""

# pandas: hold and slice the feature table.
import pandas as pd
# sqlite3: read the feature table back out of the database.
import sqlite3
# Path: OS-independent file paths.
from pathlib import Path
# yaml: read config/model_variants.yaml (the feature set for each ablation run).
import yaml
# sys: read which variant(s) to run from the command line.
import sys

# scikit-learn: preprocessing + the pipeline that guarantees train-only fitting.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# MLflow: experiment tracking. sklearn flavor to log the fitted pipeline.
import mlflow
import mlflow.sklearn
# The model + imbalance-aware metrics (log_loss = the loss LogisticRegression minimizes).
# precision = of the alarms we raise, how many are real ("don't cry wolf");
# recall = of the real events, how many we catch.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, log_loss,
                             precision_score, recall_score)


# The database + table where build_features.py stored the modeling-ready features.
DB_PATH = Path(__file__).parent.parent / "data" / "earthquakes.db"
FEATURES_TABLE = "features"

# The feature set for each ablation variant lives in this YAML (single source of
# truth), read by load_variants(). This is our FEATURE_COLUMNS allow-list — the
# model only ever sees columns named here, never the leakage-prone ones.
VARIANTS_CONFIG = Path(__file__).parent.parent / "config" / "model_variants.yaml"

# The prediction target column.
TARGET = "label_7d"

# The three regions (used for per-region metric breakdowns).
REGIONS = ["california", "japan", "greece"]

# Where MLflow stores runs (matches the existing mlflow.db backend). Launch the
# dashboard the same way:  mlflow ui --backend-store-uri sqlite:///mlflow.db
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "earthquake-baseline"


def load_features(db_path=DB_PATH, table=FEATURES_TABLE):
    """
    Read the feature table from SQLite into a DataFrame.

    Args:
        db_path (Path): the database file. Defaults to data/earthquakes.db.
        table (str): the table name. Defaults to 'features'.

    Returns:
        pd.DataFrame: the full feature table, with 'date' parsed as real dates.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["date"])
    return df


def split_data(df):
    """
    Split the feature table into train / validate / test by the 'split' column.

    Args:
        df (pd.DataFrame): the full feature table from load_features().

    Returns:
        dict[str, pd.DataFrame]: {'train': ..., 'validate': ..., 'test': ...},
                                 each a copy with a fresh row index.
    """
    return {name: df[df["split"] == name].reset_index(drop=True)
            for name in ["train", "validate", "test"]}


def load_variants(path=VARIANTS_CONFIG):
    """
    Load the base feature set + ablation variant definitions from the YAML config.

    Args:
        path (Path): the YAML file. Defaults to config/model_variants.yaml.

    Returns:
        dict: {'base_features': {'numeric': [...], 'categorical': [...]},
               'variants': {name: {'description': str, 'add': [...]}, ...}}.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_columns(cfg, variant):
    """
    Resolve the numeric + categorical feature columns for one variant.

    Args:
        cfg (dict): the loaded variants config from load_variants().
        variant (str): the variant name (e.g. 'base', 'depth', 'latlon').

    Returns:
        tuple[list[str], list[str]]: (numeric_columns, categorical_columns). A variant
        either ADDS to the base numeric features (`add:`) or fully OVERRIDES the numeric
        set (`numeric:`) — the override is used by the "remove-temporal" location-only
        experiments. Categorical defaults to the base (region) unless overridden.
    """
    base = cfg["base_features"]
    spec = cfg["variants"][variant]
    if "numeric" in spec:                       # explicit override (location-only variants)
        numeric = list(spec["numeric"])
    else:                                        # default: base numeric + the variant's extras
        numeric = list(base["numeric"]) + list(spec.get("add", []))
    categorical = list(spec.get("categorical", base["categorical"]))
    return numeric, categorical


def build_preprocessor(numeric, categorical):
    """
    Build the preprocessing step: impute + scale numerics, one-hot the region.

    Wrapped in a ColumnTransformer so that when the full model pipeline is fit on
    the TRAIN split, the medians, scaling stats, and category list are all learned
    from train only — then merely APPLIED to validation/test (no leakage).

    Args:
        numeric (list[str]): numeric feature column names.
        categorical (list[str]): categorical feature column names. Pass an empty
            list for a per-region model — a single region has no category to encode,
            so the one-hot step is skipped entirely.

    Returns:
        ColumnTransformer: the preprocessing transformer (unfitted).
    """
    transformers = []

    # Numeric: fill quiet-day blanks with the train median, then standardize —
    # added only when there are numeric columns.
    if numeric:
        numeric_steps = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_steps, numeric))

    # Categorical one-hot — added ONLY when there is a categorical column. Pooled
    # models one-hot 'region'; per-region models pass [] and skip this step.
    if categorical:
        categorical_steps = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_steps, categorical))

    return ColumnTransformer(transformers)


def build_model(preprocessor):
    """
    Full model pipeline: preprocess -> logistic regression. Fitting this on the
    train split fits BOTH the preprocessing and the coefficients on train only.

    Args:
        preprocessor (ColumnTransformer): from build_preprocessor().

    Returns:
        Pipeline: an unfitted preprocess+classify pipeline.
    """
    return Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            class_weight="balanced",  # counter the class imbalance (~25% positive)
            max_iter=1000,            # room for lbfgs to converge on scaled features
            C=1.0,
            random_state=42,
        )),
    ])


def evaluate(model, part, feature_cols):
    """
    Compute PR-AUC and F1 overall and per region for one split.

    Args:
        model (Pipeline): the fitted model.
        part (pd.DataFrame): one split (train/validate/test).
        feature_cols (list[str]): the feature columns to feed the model.

    Returns:
        dict[str, float]: {'pr_auc', 'f1', 'pr_auc_<region>', 'f1_<region>', ...}.
    """
    y = part[TARGET].to_numpy()
    proba = model.predict_proba(part[feature_cols])[:, 1]   # P(a big quake in 7d)
    pred = (proba >= 0.5).astype(int)
    regions = part["region"].to_numpy()

    metrics = {"pr_auc": average_precision_score(y, proba),
               "f1": f1_score(y, pred),
               "precision": precision_score(y, pred, zero_division=0),
               "recall": recall_score(y, pred, zero_division=0),
               "log_loss": log_loss(y, proba, labels=[0, 1])}
    # Per-region: only when that region has both classes present (else undefined).
    for region in REGIONS:
        m = regions == region
        if m.any() and len(set(y[m])) > 1:
            metrics[f"pr_auc_{region}"] = average_precision_score(y[m], proba[m])
            metrics[f"f1_{region}"] = f1_score(y[m], pred[m])
            metrics[f"precision_{region}"] = precision_score(y[m], pred[m], zero_division=0)
            metrics[f"recall_{region}"] = recall_score(y[m], pred[m], zero_division=0)
            metrics[f"log_loss_{region}"] = log_loss(y[m], proba[m], labels=[0, 1])
    return metrics


def train_and_log(parts, variant, cfg):
    """
    Fit one variant on train, evaluate on train + validation, and log to MLflow.

    Args:
        parts (dict): the train/validate/test split from split_data().
        variant (str): which variant to run (a key under 'variants' in the YAML).
        cfg (dict): the loaded variants config from load_variants().

    Returns:
        tuple[dict, dict]: (train_metrics, validation_metrics).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Resolve this variant's feature columns straight from the YAML config.
    numeric, categorical = feature_columns(cfg, variant)
    feature_cols = numeric + categorical
    model = build_model(build_preprocessor(numeric, categorical))

    with mlflow.start_run(run_name=f"{variant}-logreg"):
        # Fit on TRAIN ONLY, then score train (overfit check) and validation.
        model.fit(parts["train"][feature_cols], parts["train"][TARGET])
        train_metrics = evaluate(model, parts["train"], feature_cols)
        val_metrics = evaluate(model, parts["validate"], feature_cols)

        # Log WHICH variant and exactly which features it used, so the dashboard
        # answers "which run had what features" at a glance.
        mlflow.log_params({
            "model": "logistic_regression",
            "variant": variant,
            "description": cfg["variants"][variant]["description"],
            "features": ", ".join(feature_cols),
            "n_features": len(feature_cols),
            "class_weight": "balanced", "C": 1.0, "max_iter": 1000,
            "imputation": "median",
        })
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        # cloudpickle serialization avoids MLflow 3.x's skops "untrusted types" check.
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    return train_metrics, val_metrics


def run_ablation(parts, cfg, variants):
    """
    Train + log each requested variant, returning a validation results table.

    Args:
        parts (dict): the train/validate/test split.
        cfg (dict): the loaded variants config.
        variants (list[str]): which variant names to run, in order.

    Returns:
        pd.DataFrame: one row per variant with its validation PR-AUC / F1 / log-loss
                      (overall) plus per-region PR-AUC.
    """
    rows = []
    for variant in variants:
        _, val_metrics = train_and_log(parts, variant, cfg)
        print(f"  ran '{variant}':  val PR-AUC {val_metrics['pr_auc']:.3f}  "
              f"F1 {val_metrics['f1']:.3f}  log-loss {val_metrics['log_loss']:.3f}")
        rows.append({
            "variant": variant,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1"],
            "val_log_loss": val_metrics["log_loss"],
            **{f"pr_auc_{r}": val_metrics.get(f"pr_auc_{r}") for r in REGIONS},
        })
    return pd.DataFrame(rows)


def main():
    """
    Run one or more ablation variants and print a validation comparison.

    Usage:
        python src/train_baseline.py             # run every variant in the config
        python src/train_baseline.py latlon      # run just the 'latlon' variant
        python src/train_baseline.py base depth  # run a chosen subset, in order
    """
    parts = split_data(load_features())
    cfg = load_variants()

    # Variants to run: those named on the command line, else all in the config.
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(cfg["variants"].keys())

    print(f"Running variants: {requested}")
    results = run_ablation(parts, cfg, requested)

    print("\nValidation comparison (sorted by PR-AUC):")
    print(results.sort_values("val_pr_auc", ascending=False).to_string(index=False))


# Only run the smoke test when launched directly (e.g. "python src/train_baseline.py").
if __name__ == "__main__":
    main()