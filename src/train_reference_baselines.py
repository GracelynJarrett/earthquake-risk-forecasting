"""
train_reference_baselines.py — Reference baselines for Week 3 (Day 3 analysis).

Two simple, NON-learned reference points that frame how good the logistic
regression really is:
  1. Random "no-skill" floor  — DummyClassifier(strategy="prior"): always predicts
     the base rate, so PR-AUC = base rate and F1 = 0. The true floor.
  2. Naive persistence heuristic — "expect a big quake next week if there's been
     recent significant activity": rank by large_30d, predict positive if large_30d >= 1.

Beating BOTH is what shows the logistic regression actually earns its keep. Both
are logged to the same 'earthquake-baseline' MLflow experiment so they line up
next to the 5 logistic-regression runs in one comparison.
"""

# MLflow: log the two reference runs alongside the logistic-regression runs.
import mlflow
# The random no-skill baseline + the metrics we report.
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# Reuse the loaders, feature config, and constants from the pooled baseline.
from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    TARGET, REGIONS, TRACKING_URI, EXPERIMENT_NAME,
)


# The persistence signal: count of region-significant quakes in the last 30 days.
HEURISTIC_SIGNAL = "large_30d"


def per_region_metrics(y, score, pred, regions):
    """
    PR-AUC / precision / recall broken out per region (where both classes present).

    Args:
        y (np.ndarray): true labels.
        score (np.ndarray): risk scores (higher = more likely positive).
        pred (np.ndarray): 0/1 predictions for precision/recall.
        regions (np.ndarray): region label per row.

    Returns:
        dict[str, float]: {'pr_auc_<region>', 'precision_<region>', 'recall_<region>', ...}.
    """
    out = {}
    for region in REGIONS:
        m = regions == region
        if m.any() and len(set(y[m])) > 1:
            out[f"pr_auc_{region}"] = average_precision_score(y[m], score[m])
            out[f"precision_{region}"] = precision_score(y[m], pred[m], zero_division=0)
            out[f"recall_{region}"] = recall_score(y[m], pred[m], zero_division=0)
    return out


def run_dummy(parts):
    """
    Random no-skill baseline (always predicts the class prior).

    Args:
        parts (dict): the train/validate/test split.

    Returns:
        dict: validation metrics (PR-AUC ~ base rate, F1 = 0, plus per-region PR-AUC).
    """
    # Dummy ignores the features, but still needs an X of the right shape — reuse
    # the 'base' variant's columns just to supply one.
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, "base")
    feats = numeric + categorical

    model = DummyClassifier(strategy="prior")
    with mlflow.start_run(run_name="dummy-no-skill"):
        model.fit(parts["train"][feats], parts["train"][TARGET])
        val = parts["validate"]
        y = val[TARGET].to_numpy()
        regions = val["region"].to_numpy()
        proba = model.predict_proba(val[feats])[:, 1]   # constant = base rate
        pred = model.predict(val[feats])                # always the majority class (0)

        metrics = {"pr_auc": average_precision_score(y, proba), "f1": f1_score(y, pred),
                   "precision": precision_score(y, pred, zero_division=0),
                   "recall": recall_score(y, pred, zero_division=0)}
        metrics.update(per_region_metrics(y, proba, pred, regions))

        mlflow.log_params({"model": "dummy", "strategy": "prior", "variant": "reference"})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
    return metrics


def run_heuristic(parts):
    """
    Naive persistence heuristic: risk ~ large_30d; predict positive if large_30d >= 1.

    Args:
        parts (dict): the train/validate/test split.

    Returns:
        dict: validation metrics (PR-AUC ranks by large_30d, F1 uses the >= 1 rule).
    """
    with mlflow.start_run(run_name="heuristic-recent-activity"):
        val = parts["validate"]
        y = val[TARGET].to_numpy()
        regions = val["region"].to_numpy()
        # Rank risk by how much significant activity there's been recently...
        score = val[HEURISTIC_SIGNAL].to_numpy().astype(float)
        # ...and flag a positive if there was ANY significant quake in the last 30 days.
        pred = (val[HEURISTIC_SIGNAL].to_numpy() >= 1).astype(int)

        metrics = {"pr_auc": average_precision_score(y, score), "f1": f1_score(y, pred),
                   "precision": precision_score(y, pred, zero_division=0),
                   "recall": recall_score(y, pred, zero_division=0)}
        metrics.update(per_region_metrics(y, score, pred, regions))

        mlflow.log_params({"model": "heuristic", "rule": f"{HEURISTIC_SIGNAL} >= 1",
                           "variant": "reference"})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
    return metrics


def main():
    """Run both reference baselines and print them for comparison."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    parts = split_data(load_features())
    dummy = run_dummy(parts)
    heur = run_heuristic(parts)

    print("Reference baselines (validation):")
    print(f"  dummy-no-skill            PR-AUC {dummy['pr_auc']:.3f}  F1 {dummy['f1']:.3f}  "
          f"precision {dummy['precision']:.3f}  recall {dummy['recall']:.3f}")
    print(f"  heuristic ({HEURISTIC_SIGNAL} >= 1)  PR-AUC {heur['pr_auc']:.3f}  F1 {heur['f1']:.3f}  "
          f"precision {heur['precision']:.3f}  recall {heur['recall']:.3f}")
    print("  Per-region PR-AUC (dummy | heuristic):")
    for region in REGIONS:
        d = dummy.get(f"pr_auc_{region}", float("nan"))
        h = heur.get(f"pr_auc_{region}", float("nan"))
        print(f"    {region:<11} {d:.3f} | {h:.3f}")


# Only run when launched directly (e.g. "python src/train_reference_baselines.py").
if __name__ == "__main__":
    main()