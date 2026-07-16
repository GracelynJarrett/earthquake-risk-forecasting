"""
train_interactions.py — Pooled logistic regression WITH region×feature interactions.

Tests the idea that different features matter in different regions (e.g. fault-line
in Greece) WITHOUT training separate, data-starved per-region models. We add
region-specific interaction columns to the ONE pooled model, so it can learn a
different fault-distance slope per region while keeping ALL the training data.

An interaction column `feat__<region>` = the feature's value in that region, 0
elsewhere. With the base features + the region one-hot (per-region intercepts) +
these interaction columns (per-region slopes), a single linear model can say
"fault-distance matters *in Greece*" — something the plain pooled model can't.

Logged to the 'earthquake-baseline' MLflow experiment with precision/recall.
"""

# MLflow: log next to the other earthquake-baseline runs.
import mlflow
import mlflow.sklearn

# Reuse the loaders, config, preprocessing, model, and metrics from the baseline.
from train_baseline import (
    load_features, split_data, load_variants, build_preprocessor, build_model,
    evaluate, TARGET, REGIONS, TRACKING_URI, EXPERIMENT_NAME,
)


def add_interactions(parts, feature):
    """
    Add region-specific interaction columns for `feature` to every split, in place.

    For each region r, creates `<feature>__<r>` = the feature value where region == r,
    and 0 otherwise. Returns the list of new column names.

    Args:
        parts (dict): the train/validate/test split (DataFrames, modified in place).
        feature (str): the numeric feature to interact with region.

    Returns:
        list[str]: the new interaction column names.
    """
    cols = []
    for region in REGIONS:
        col = f"{feature}__{region}"
        for part in parts.values():
            part[col] = part[feature] * (part["region"] == region).astype(float)
        cols.append(col)
    return cols


def run(parts, cfg, interact_features, run_name):
    """
    Fit a pooled logistic regression with the given region×feature interactions.

    Args:
        parts (dict): the train/validate/test split.
        cfg (dict): the loaded variants config (for the base feature list).
        interact_features (list[str]): features to interact with region.
        run_name (str): MLflow run name.

    Returns:
        tuple[dict, dict]: (train_metrics, validation_metrics).
    """
    # Base recent-activity features (keeps California's signal) + the interaction
    # columns (adds per-region slopes for the interacted features).
    numeric = list(cfg["base_features"]["numeric"])
    for feature in interact_features:
        numeric += add_interactions(parts, feature)
    categorical = list(cfg["base_features"]["categorical"])
    feature_cols = numeric + categorical

    model = build_model(build_preprocessor(numeric, categorical))

    with mlflow.start_run(run_name=run_name):
        model.fit(parts["train"][feature_cols], parts["train"][TARGET])
        train_metrics = evaluate(model, parts["train"], feature_cols)
        val_metrics = evaluate(model, parts["validate"], feature_cols)

        mlflow.log_params({
            "model": "logistic_regression",
            "variant": run_name,
            "interactions": ", ".join(interact_features),
            "n_features": len(feature_cols),
            "class_weight": "balanced", "C": 1.0, "max_iter": 1000, "imputation": "median",
        })
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    return train_metrics, val_metrics


def main():
    """Run the interaction experiments and print Greece + overall vs. the base bar."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    cfg = load_variants()

    experiments = [
        (["avg_dist_30d"], "base+dist_x_region"),
        (["avg_dist_30d", "large_30d"], "base+dist+activity_x_region"),
    ]

    print("Interaction experiments (validation) — bar to beat: base PR-AUC 0.360, Greece 0.296")
    for interact_features, run_name in experiments:
        parts = split_data(load_features())   # fresh split so interaction cols don't accumulate
        _, val = run(parts, cfg, interact_features, run_name)
        print(f"  {run_name:<30} PR-AUC {val['pr_auc']:.3f}  prec {val['precision']:.3f}  "
              f"rec {val['recall']:.3f}  |  CA {val.get('pr_auc_california', float('nan')):.3f}  "
              f"JP {val.get('pr_auc_japan', float('nan')):.3f}  "
              f"GR {val.get('pr_auc_greece', float('nan')):.3f}")


# Only run when launched directly (e.g. "python src/train_interactions.py").
if __name__ == "__main__":
    main()