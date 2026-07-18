"""
train_ransomforest.py — Random Forest experiment (Week 3 preview of the tree family).

Runs the 5 POOLED feature variants with a Random Forest instead of logistic
regression, to test whether a tree-based model — which captures nonlinearity and
feature interactions the linear baseline cannot — beats logistic regression (and the
naive large_30d heuristic), especially on Japan/Greece.

Reuses the pipeline from train_baseline.py (same preprocessing, for a fair
comparison) and swaps in RandomForestClassifier. Logged to the SAME
'earthquake-baseline' MLflow experiment so RF sits next to the logreg + reference
runs. NOT tuned — hyperparameter tuning is Week 4; we watch the train-vs-validation
gap to see if the untuned forest overfits.

(Note: the filename keeps its original spelling, 'ransomforest' — it is Random Forest.)
"""

# MLflow: log the RF runs next to the logistic-regression runs.
import mlflow
import mlflow.sklearn
# The tree-ensemble model + the pipeline wrapper.
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Reuse the loaders, feature config, preprocessing, and metrics from the baseline.
from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, evaluate, TARGET, REGIONS, TRACKING_URI, EXPERIMENT_NAME,
)


# Sensible, UNTUNED Random Forest settings (hyperparameter tuning is Week 4).
# min_samples_leaf is a mild guardrail against the forest memorizing training rows.
RF_PARAMS = dict(
    n_estimators=300,
    class_weight="balanced",
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
)


def build_rf_model(numeric, categorical):
    """
    Same preprocessing as the logistic baseline, with a Random Forest classifier.

    Args:
        numeric (list[str]): numeric feature columns.
        categorical (list[str]): categorical feature columns.

    Returns:
        Pipeline: an unfitted preprocess + random-forest pipeline.
    """
    return Pipeline([
        ("pre", build_preprocessor(numeric, categorical)),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])


def train_and_log(parts, variant, cfg):
    """
    Fit a Random Forest for one variant on train, evaluate, and log to MLflow.

    Args:
        parts (dict): the train/validate/test split.
        variant (str): the feature variant (base / depth / faultline / both / latlon).
        cfg (dict): the loaded variants config.

    Returns:
        tuple[dict, dict]: (train_metrics, validation_metrics).
    """
    numeric, categorical = feature_columns(cfg, variant)
    feature_cols = numeric + categorical
    model = build_rf_model(numeric, categorical)

    with mlflow.start_run(run_name=f"rf-{variant}"):
        # Fit on TRAIN only, then score train (overfit check) and validation.
        model.fit(parts["train"][feature_cols], parts["train"][TARGET])
        train_metrics = evaluate(model, parts["train"], feature_cols)
        val_metrics = evaluate(model, parts["validate"], feature_cols)

        mlflow.log_params({
            "model": "random_forest",
            "variant": variant,
            "features": ", ".join(feature_cols),
            "n_features": len(feature_cols),
            # Log the RF settings (skip n_jobs — it's just a speed knob).
            **{k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
            "imputation": "median",
        })
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        # cloudpickle serialization avoids MLflow 3.x's skops "untrusted types" check.
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    return train_metrics, val_metrics


def main():
    """Run the 5 pooled variants with a Random Forest and print a comparison."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    parts = split_data(load_features())
    cfg = load_variants()

    print("Random Forest — pooled ablation (train vs. validation):")
    for variant in cfg["variants"]:
        tr, val = train_and_log(parts, variant, cfg)
        gap = tr["pr_auc"] - val["pr_auc"]
        print(f"  {variant:<10} train {tr['pr_auc']:.3f}  val {val['pr_auc']:.3f}  "
              f"F1 {val['f1']:.3f}  gap {gap:.3f}  |  "
              f"CA {val.get('pr_auc_california', float('nan')):.3f}  "
              f"JP {val.get('pr_auc_japan', float('nan')):.3f}  "
              f"GR {val.get('pr_auc_greece', float('nan')):.3f}")


# Only run when launched directly (e.g. "python src/train_ransomforest.py").
if __name__ == "__main__":
    main()