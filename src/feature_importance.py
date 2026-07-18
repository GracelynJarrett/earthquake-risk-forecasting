"""
feature_importance.py — Which features does the chosen model actually rely on?

Complements the ablation: the ablation asked "does ADDING a feature change the score?";
this asks "how much does the FITTED model weight each feature internally?"

Two views of the chosen pooled `depth` feature set:
  - Logistic regression (the selected model): because the features are standardized, the
    coefficients are directly comparable — larger |coef| = more influence; the sign gives
    direction (positive = pushes toward "a big quake is coming").
  - Random Forest: Gini feature_importances_ as a cross-check (RF overfit, so read with care).

Both are re-fit on TRAIN only, reusing the exact pipeline from train_baseline.py.
"""

# pandas: assemble and sort the importance tables.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Reuse the loaders, feature config, preprocessing, and logistic model.
from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, build_model, TARGET,
)
# Reuse the exact Random Forest settings from the RF experiment.
from train_ransomforest import RF_PARAMS


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


def main():
    """Print logistic-regression coefficients and Random Forest importances."""
    parts = split_data(load_features())
    cfg = load_variants()

    print(f"Feature importance for the chosen feature set ('{VARIANT}')\n")

    print("=== Logistic regression — standardized coefficients ===")
    print("(sign = direction; |value| = influence)")
    for _, row in logreg_importance(parts, cfg).iterrows():
        print(f"  {row['feature']:<28} {row['coefficient']:+.3f}")

    print("\n=== Random Forest — Gini importance (cross-check; RF overfits) ===")
    for _, row in rf_importance(parts, cfg).iterrows():
        print(f"  {row['feature']:<28} {row['importance']:.3f}")


# Only run when launched directly (e.g. "python src/feature_importance.py").
if __name__ == "__main__":
    main()