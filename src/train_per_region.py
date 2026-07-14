"""
train_per_region.py — Per-region logistic-regression ablation (Week 3 experiment).

Trains ONE model per region (California / Japan / Greece) instead of a single
pooled model, running the same 5 feature variants for each region -> 15 models.
This tests whether region-specific models beat the pooled baseline — especially for
Japan and Greece — and whether distance-to-fault finally helps Greece once
California is no longer sharing its coefficients.

Reuses the building blocks from train_baseline.py (loading, splitting, the variant
config, the preprocessing/model pipeline, and the metrics). The only differences:
  - each model trains on ONE region's rows, and
  - the 'region' one-hot is dropped (a single region is a constant column).
Runs are logged to a SEPARATE MLflow experiment so they stay distinct from the
pooled 5-run ablation.
"""

# pandas: assemble the results comparison table.
import pandas as pd
# MLflow: log the per-region runs (separate experiment from the pooled ablation).
import mlflow
import mlflow.sklearn

# Reuse everything we already built for the pooled baseline — one source of truth.
from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, build_model, evaluate,
    TARGET, REGIONS, TRACKING_URI,
)


# Keep these 15 runs in their own experiment, separate from the pooled 5.
EXPERIMENT_NAME = "earthquake-per-region"


def region_split(parts, region):
    """
    Narrow each split (train/validate/test) down to a single region's rows.

    Args:
        parts (dict): the full train/validate/test split from split_data().
        region (str): the region to keep.

    Returns:
        dict[str, pd.DataFrame]: the same splits, filtered to that region.
    """
    return {name: part[part["region"] == region].reset_index(drop=True)
            for name, part in parts.items()}


def train_region_variant(region, variant, rparts, cfg):
    """
    Train one variant on ONE region (no region one-hot) and log it to MLflow.

    Args:
        region (str): the region this model is for.
        variant (str): the feature variant (base / depth / faultline / both / latlon).
        rparts (dict): that region's train/validate/test split.
        cfg (dict): the loaded variants config.

    Returns:
        dict: the model's validation metrics.
    """
    # Base numeric + the variant's extras; drop the categorical (region is constant).
    numeric, _ = feature_columns(cfg, variant)
    model = build_model(build_preprocessor(numeric, []))

    with mlflow.start_run(run_name=f"{region}-{variant}"):
        # Fit on this region's TRAIN split only, then score train + validation.
        model.fit(rparts["train"][numeric], rparts["train"][TARGET])
        train_metrics = evaluate(model, rparts["train"], numeric)
        val_metrics = evaluate(model, rparts["validate"], numeric)

        mlflow.log_params({
            "model": "logistic_regression",
            "region": region,
            "variant": variant,
            "features": ", ".join(numeric),
            "n_features": len(numeric),
            "class_weight": "balanced", "C": 1.0, "max_iter": 1000,
            "imputation": "median",
        })
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        # cloudpickle serialization avoids MLflow 3.x's skops "untrusted types" check.
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    return val_metrics


def main():
    """Train all 5 variants for each region (15 models) and print a comparison."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    parts = split_data(load_features())
    cfg = load_variants()
    variants = list(cfg["variants"].keys())

    rows = []
    for region in REGIONS:
        rparts = region_split(parts, region)
        for variant in variants:
            val = train_region_variant(region, variant, rparts, cfg)
            print(f"  {region:<11} {variant:<10} PR-AUC {val['pr_auc']:.3f}  "
                  f"F1 {val['f1']:.3f}  log-loss {val['log_loss']:.3f}")
            rows.append({"region": region, "variant": variant,
                         "val_pr_auc": val["pr_auc"], "val_f1": val["f1"],
                         "val_log_loss": val["log_loss"]})

    results = pd.DataFrame(rows)
    print("\nBest variant per region (by validation PR-AUC):")
    for region in REGIONS:
        best = (results[results["region"] == region]
                .sort_values("val_pr_auc", ascending=False).iloc[0])
        print(f"  {region:<11} -> {best['variant']:<10} PR-AUC {best['val_pr_auc']:.3f}")


# Only run when launched directly (e.g. "python src/train_per_region.py").
if __name__ == "__main__":
    main()