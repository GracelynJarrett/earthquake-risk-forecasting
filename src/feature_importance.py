"""
feature_importance.py — Which features does the chosen model actually rely on?

Complements the ablation: the ablation asked "does ADDING a feature change the score?";
this asks "how much does the FITTED model weight each feature internally?"

Three views of the chosen pooled `depth` feature set:
  - XGBoost (the Week 4 model): "gain" importance — how much each feature improved the
    trees' decisions on average when used to split. This is the one that tells us what
    the model we're carrying forward actually relies on.
  - Logistic regression (Week 3 baseline): because the features are standardized, the
    coefficients are directly comparable — larger |coef| = more influence; the sign gives
    direction (positive = pushes toward "a big quake is coming").
  - Random Forest: Gini feature_importances_ as a cross-check (RF overfit, so read with care).

All are re-fit on TRAIN only, reusing the exact pipeline from train_baseline.py.
"""

# pandas: assemble and sort the importance tables.
import pandas as pd
# sys: let the caller inspect a different feature set from the command line.
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
# permutation_importance: shuffle each feature on VALIDATION and measure the score
# drop — the honest cross-check on XGBoost's internal gain numbers.
from sklearn.inspection import permutation_importance

# Reuse the loaders, feature config, preprocessing, and logistic model.
from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, build_model, TARGET,
)
# Reuse the exact Random Forest settings from the RF experiment.
from train_ransomforest import RF_PARAMS
# Reuse the XGBoost pipeline builder so importances reflect the real Week 4 model.
from train_xgboost import build_xgb_model


# The chosen feature set (the model we carry to Week 4).
VARIANT = "depth"


def logreg_importance(parts, cfg):
    """
    Fit the chosen logistic regression and return its standardized coefficients.

    Returns:
        pd.DataFrame: columns [feature, coefficient], sorted by |coefficient| (desc).
    """
    numeric, categorical = feature_columns(cfg, VARIANT)
    feature_cols = numeric + categorical
    model = build_model(build_preprocessor(numeric, categorical))
    model.fit(parts["train"][feature_cols], parts["train"][TARGET])

    # Feature names AFTER preprocessing (region one-hot expands into 3 columns).
    names = model.named_steps["pre"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    # Sort by absolute influence (biggest driver first), keeping the sign.
    return df.iloc[df["coefficient"].abs().sort_values(ascending=False).index]


def rf_importance(parts, cfg):
    """
    Fit the Random Forest on the same feature set and return its Gini importances.

    Returns:
        pd.DataFrame: columns [feature, importance], sorted by importance (desc).
    """
    numeric, categorical = feature_columns(cfg, VARIANT)
    feature_cols = numeric + categorical
    model = Pipeline([
        ("pre", build_preprocessor(numeric, categorical)),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])
    model.fit(parts["train"][feature_cols], parts["train"][TARGET])

    names = model.named_steps["pre"].get_feature_names_out()
    importances = model.named_steps["clf"].feature_importances_
    return (pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False))


def xgb_importance(parts, cfg):
    """
    Fit the XGBoost model on the chosen feature set and return its gain importances.

    "Gain" = how much each feature improved the trees' decisions on average when it
    was used to split — the most meaningful XGBoost importance (vs. raw split counts).
    A feature the trees never found useful gets 0.

    Returns:
        pd.DataFrame: columns [feature, importance], sorted by importance (desc).
    """
    numeric, categorical = feature_columns(cfg, VARIANT)
    feature_cols = numeric + categorical

    # Same balanced positive-class weight the trainer uses, so these importances
    # reflect the model we actually build (mirrors train_xgboost.train_and_log).
    y_train = parts["train"][TARGET]
    n_pos = int((y_train == 1).sum())
    spw = ((y_train == 0).sum() / n_pos) if n_pos else 1.0

    model = build_xgb_model(build_preprocessor(numeric, categorical), scale_pos_weight=spw)
    model.fit(parts["train"][feature_cols], parts["train"][TARGET])

    # Read importance as "gain" (avg decision-improvement per split), aligned to the
    # preprocessed feature names (region one-hot expands into 3 columns).
    clf = model.named_steps["clf"]
    clf.importance_type = "gain"
    names = model.named_steps["pre"].get_feature_names_out()
    importances = clf.feature_importances_
    return (pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False))


def permutation_importance_xgb(parts, cfg, n_repeats=10):
    """
    Permutation importance for the XGBoost model, measured on VALIDATION.

    The honest cross-check on gain: fit on train, then for each feature, shuffle that
    one column in the VALIDATION data and see how much the validation PR-AUC drops.
    A big drop = the model genuinely relies on that feature to generalize; a near-zero
    (or negative) drop = the feature isn't earning its place. Measured on validation,
    never test, because the question is about generalization.

    Returns:
        pd.DataFrame: columns [feature, importance_mean, importance_std], sorted desc.
                      importance = average drop in validation PR-AUC when the feature
                      is scrambled (across n_repeats shuffles).
    """
    numeric, categorical = feature_columns(cfg, VARIANT)
    feature_cols = numeric + categorical

    # Same balanced weight as the trainer, fit on TRAIN only.
    y_train = parts["train"][TARGET]
    n_pos = int((y_train == 1).sum())
    spw = ((y_train == 0).sum() / n_pos) if n_pos else 1.0
    model = build_xgb_model(build_preprocessor(numeric, categorical), scale_pos_weight=spw)
    model.fit(parts["train"][feature_cols], y_train)

    # Shuffle each raw feature in VALIDATION and measure the PR-AUC drop. Passing the
    # raw columns means 'region' is permuted as ONE feature (not 3 one-hot columns),
    # so this reads more cleanly than gain. average_precision = PR-AUC, our headline.
    result = permutation_importance(
        model, parts["validate"][feature_cols], parts["validate"][TARGET],
        scoring="average_precision", n_repeats=n_repeats, random_state=42, n_jobs=-1,
    )
    return (pd.DataFrame({
                "feature": feature_cols,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            })
            .sort_values("importance_mean", ascending=False))


def main():
    """
    Print XGBoost gain + permutation importance, logistic coefficients, and RF importances.

    Usage:
        python src/feature_importance.py                # inspect the default set ('depth')
        python src/feature_importance.py longhz_135a    # inspect any variant (e.g. with new features)
    """
    # Optionally inspect a different feature set (so we can see the new long-horizon
    # features' importance). Reassigns the module-level VARIANT the functions read.
    global VARIANT
    if len(sys.argv) > 1:
        VARIANT = sys.argv[1]

    parts = split_data(load_features())
    cfg = load_variants()

    print(f"Feature importance for the chosen feature set ('{VARIANT}')\n")

    print("=== XGBoost — gain importance (the Week 4 model) ===")
    print("(higher = the feature improved the trees' decisions more)")
    for _, row in xgb_importance(parts, cfg).iterrows():
        print(f"  {row['feature']:<28} {row['importance']:.3f}")

    print("\n=== XGBoost — permutation importance on VALIDATION (the honest check) ===")
    print("(drop in validation PR-AUC when the feature is scrambled; mean +/- std)")
    for _, row in permutation_importance_xgb(parts, cfg).iterrows():
        print(f"  {row['feature']:<20} {row['importance_mean']:+.4f}  +/- {row['importance_std']:.4f}")

    print("\n=== Logistic regression — standardized coefficients ===")
    print("(sign = direction; |value| = influence)")
    for _, row in logreg_importance(parts, cfg).iterrows():
        print(f"  {row['feature']:<28} {row['coefficient']:+.3f}")

    print("\n=== Random Forest — Gini importance (cross-check; RF overfits) ===")
    for _, row in rf_importance(parts, cfg).iterrows():
        print(f"  {row['feature']:<28} {row['importance']:.3f}")


# Only run when launched directly (e.g. "python src/feature_importance.py").
if __name__ == "__main__":
    main()