"""
risk_tiers.py — Low / Medium / High risk reporting for the region-blind model
(Week 4, Day 3→4, experiment #3).

Instead of a blunt yes/no, first responders get a risk LEVEL. This has two parts:

  Part A — Calibration check: do the model's predicted probabilities match reality?
           (stakeholder promise: "when we say 20% risk, it happens ~20% of the time".)
  Part B — Tiers: split predictions into Low / Medium / High, with boundaries chosen
           FROM the calibration evidence (added after we read Part A).

Reuses the exact region-blind model from tune_thresholds.py (same fit on TRAIN, same
validation probabilities); TEST stays sealed.

Structure:
  - calibration_table : bin predictions, compare predicted vs observed rate
  - main              : print the calibration tables (overall + per region)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
# IsotonicRegression: learns a monotonic map from raw probability -> observed rate,
# fixing over/under-confidence (and the Low/Medium tier inversion).
from sklearn.isotonic import IsotonicRegression

# Reuse the fitted region-blind model + validation probabilities from #4.
from tune_thresholds import fit_and_predict
from train_baseline import load_features, split_data, TARGET, REGIONS


def calibration_table(y, proba, n_bins=10):
    """
    Bin predictions and compare predicted probability vs observed frequency.

    Splits the 0..1 probability range into n_bins equal-width bins; for each bin reports
    how many days fell in it, the average PREDICTED probability, and the actual OBSERVED
    positive rate. A well-calibrated model has observed ≈ predicted in every bin.

    Args:
        y (array-like): true 0/1 labels.
        proba (array-like): predicted P(quake) for the same rows.
        n_bins (int): number of equal-width probability bins.

    Returns:
        pd.DataFrame: columns [bin, n, mean_pred, observed] for non-empty bins.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)   # which bin each row lands in

    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "mean_pred": round(float(proba[m].mean()), 3),   # what the model claimed
            "observed": round(float(y[m].mean()), 3),        # what actually happened
        })
    return pd.DataFrame(rows)


# Tier boundaries as multiples of each region's normal (base-rate) weekly risk.
# Low: below normal; Medium: normal up to HIGH_MULT x normal; High: >= HIGH_MULT x normal.
HIGH_MULT = 1.5


def region_base_rates(train):
    """Each region's 'normal' weekly risk = its positive rate on TRAIN (leakage-safe)."""
    return {r: float(train.loc[train["region"] == r, TARGET].mean()) for r in REGIONS}


def assign_tiers(val, proba, base_rates, high_mult=HIGH_MULT):
    """
    Label each validation day Low / Medium / High relative to its region's base rate.

    Low    : predicted risk below the region's normal weekly risk
    Medium : normal up to high_mult x normal
    High   : high_mult x normal and above

    Returns:
        pd.DataFrame: a copy of the region + label columns with added 'proba' and 'tier'.
    """
    out = val[["region", TARGET]].copy()
    out["proba"] = proba
    tier = np.full(len(out), "Low", dtype=object)
    for r in REGIONS:
        b = base_rates[r]
        m = (out["region"] == r).to_numpy()
        p = out["proba"].to_numpy()
        tier[m & (p >= b)] = "Medium"
        tier[m & (p >= high_mult * b)] = "High"
    out["tier"] = tier
    return out


def tier_summary(tiers):
    """Per (region, tier): number of days, their share, and the OBSERVED quake rate."""
    rows = []
    for r in REGIONS:
        sub = tiers[tiers["region"] == r]
        for t in ["Low", "Medium", "High"]:
            g = sub[sub["tier"] == t]
            rows.append({
                "region": r, "tier": t, "days": len(g),
                "share": round(len(g) / len(sub), 3) if len(sub) else 0.0,
                "observed_rate": round(float(g[TARGET].mean()), 3) if len(g) else float("nan"),
            })
    return pd.DataFrame(rows)


def calibrate_per_region(val, proba):
    """
    Fit a per-region isotonic calibrator on validation and return calibrated probabilities.

    Isotonic regression learns a monotonic map from raw probability -> observed rate,
    separately per region (each region's probabilities mean something different). Fit on
    VALIDATION (our tuning set, like the thresholds); confirmed on TEST at Day 5.

    Returns:
        tuple[np.ndarray, dict]: (calibrated proba aligned to val, {region: fitted IsotonicRegression}).
    """
    cal = np.empty(len(val))
    models = {}
    reg = val["region"].to_numpy()
    y = val[TARGET].to_numpy()
    for r in REGIONS:
        m = reg == r
        iso = IsotonicRegression(out_of_bounds="clip")
        cal[m] = iso.fit_transform(proba[m], y[m])   # calibrated prob for this region's days
        models[r] = iso
    return cal, models


def main():
    """Calibration check + Low/Med/High tiers for the region-blind model (validation)."""
    val, proba = fit_and_predict()
    y = val[TARGET].to_numpy()

    print("Calibration check — region-blind model (validation)\n")
    print(f"Overall Brier score: {brier_score_loss(y, proba):.4f}  (lower = better; 0 is perfect)\n")

    print("=== Overall ===")
    print(calibration_table(y, proba).to_string(index=False))

    for region in REGIONS:
        m = (val["region"] == region).to_numpy()
        print(f"\n=== {region.title()} ===")
        print(calibration_table(val.loc[m, TARGET].to_numpy(), proba[m]).to_string(index=False))

    # --- Part B: Low/Medium/High tiers, per-region base-rate bands ---
    train = split_data(load_features())["train"]
    base_rates = region_base_rates(train)
    tiers = assign_tiers(val, proba, base_rates)

    print("\n\nRisk tiers — per-region base-rate bands")
    print("Cutoffs (region: normal / high):")
    for r in REGIONS:
        b = base_rates[r]
        print(f"  {r:<11} normal>={b:.3f}   high>={HIGH_MULT * b:.3f}")

    print("\nObserved quake rate by tier (should climb Low -> Medium -> High):")
    print(tier_summary(tiers).to_string(index=False))

    # --- After isotonic calibration (fit per region on validation) ---
    cal_proba, _ = calibrate_per_region(val, proba)
    print(f"\n\n=== After isotonic calibration ===")
    print(f"Overall Brier score: {brier_score_loss(y, cal_proba):.4f}  (was uncalibrated above)")
    cal_tiers = assign_tiers(val, cal_proba, base_rates)
    print("\nObserved quake rate by tier — calibrated (should now climb cleanly):")
    print(tier_summary(cal_tiers).to_string(index=False))


# Only run when launched directly (e.g. "python src/risk_tiers.py").
if __name__ == "__main__":
    main()