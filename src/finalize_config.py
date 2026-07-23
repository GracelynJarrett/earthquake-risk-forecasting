"""
finalize_config.py — Freeze the final model's decision config to config/final_model.json
(Week 4, Day 5, step 1 of finalize).

Writes a human-readable, git-committable record of EXACTLY what the final model is: the feature
set, tuned hyperparameters, per-region F2 thresholds, per-region base-rate tier anchors, and the
calibration/split notes. The numbers are RECOMPUTED from the tuned model on train/validation (never
test), so the config can't drift from the actual model or be hand-mistyped.

This is step 1 of the lock-before-you-look finalize: freeze the config, register the model, THEN
unseal test once.

Structure:
  - build_final_config : assemble the config dict (reuses our existing tuning/tier functions)
  - main               : write it to config/final_model.json and print a summary
"""

import json
from pathlib import Path

from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, TARGET, REGIONS,
)
from train_xgboost import build_xgb_model
from tune_thresholds import (
    fit_and_predict, sweep_region, pick_threshold, BEST_VARIANT, SCALE_POS_WEIGHT, BETA,
)
from risk_tiers import region_base_rates, HIGH_MULT


OUTPUT_PATH = Path(__file__).parent.parent / "config" / "final_model.json"


def build_final_config():
    """
    Assemble the frozen decision config for the final model (numbers only, reproducible).

    Recomputes the per-region F2 thresholds and base-rate anchors from the tuned model on
    train/validation (never test), and reads the tuned hyperparameters straight off
    build_xgb_model so the config can't drift from the actual model.

    Returns:
        dict: the full final-model configuration, ready to serialize to JSON.
    """
    parts = split_data(load_features())
    cfg = load_variants()
    numeric, categorical = feature_columns(cfg, BEST_VARIANT)
    feature_cols = numeric + categorical

    # Tuned model + validation probabilities (same path as tune_thresholds / risk_tiers).
    val, proba = fit_and_predict(BEST_VARIANT)

    # Per-region F2 thresholds (favor recall) + per-region base-rate tier anchors.
    thresholds = {}
    for r in REGIONS:
        m = (val["region"] == r).to_numpy()
        _, curve = sweep_region(val.loc[m, TARGET].to_numpy(), proba[m])
        thresholds[r] = round(pick_threshold(curve, "fbeta", BETA), 3)
    base_rates = {r: round(v, 4) for r, v in region_base_rates(parts["train"]).items()}

    # Read tuned hyperparameters off the model itself (build_xgb_model = source of truth).
    hp = build_xgb_model(build_preprocessor(numeric, categorical),
                         SCALE_POS_WEIGHT).named_steps["clf"].get_params()
    hyperparams = {k: hp[k] for k in ["n_estimators", "max_depth", "learning_rate",
                                      "min_child_weight", "subsample", "colsample_bytree"]}
    hyperparams.update({"gamma": hp.get("gamma") or 0, "reg_lambda": hp.get("reg_lambda") or 1})

    return {
        "model": "xgboost_tuned",
        "variant": BEST_VARIANT,
        "features": feature_cols,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "hyperparameters": hyperparams,
        "threshold_objective": f"F{BETA}",
        "thresholds": thresholds,
        "tier_method": "per_region_base_rate",
        "base_rates": base_rates,
        "high_mult": HIGH_MULT,
        "calibration": "isotonic_per_region (fit on validation)",
        "split": "temporal (train<=2018, validate 2019-21, test 2022-26, 7d embargo)",
    }


def main():
    """Build the final-model config and write it to config/final_model.json."""
    config = build_final_config()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote final-model config to {OUTPUT_PATH}\n")
    print(json.dumps(config, indent=2))


# Only run when launched directly (e.g. "python src/finalize_config.py").
if __name__ == "__main__":
    main()