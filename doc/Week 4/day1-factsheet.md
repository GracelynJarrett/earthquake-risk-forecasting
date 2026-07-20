# Week 4 — Day 1 Fact-Sheet (XGBoost + Feature Importance)

*Verified numbers only — raw material for the Week-4 report. Date run: 2026-07-20.*
*All results are on the **validation** split (2019–21). Test (2022–26) remains sealed.*

---

## 1. Setup (what was run)

- **Model:** XGBoost (`XGBClassifier`), script `src/train_xgboost.py`, MLflow experiment `earthquake-xgboost`.
- **Feature set:** `depth` variant = 8 base recent-activity features + `avg_depth_30d` (the Week-3 winner).
- **Hyperparameters (Day-1 defaults, untuned):** n_estimators=300, max_depth=4, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, eval_metric=aucpr, random_state=42.
- **Imbalance handling:** `scale_pos_weight = 3.044` (train negatives ÷ positives), computed on train only.
- **Split sizes / class balance:**
  - Train: 20,799 rows, 5,143 positive (24.7% positive).
  - Validation: 3,267 rows, 919 positive (28.1% positive).
  - Train positive rate by region: **Japan 40.6%**, Greece 19.7%, **California 13.9%** (≈3× spread).

---

## 2. Headline result — XGBoost does NOT beat the baseline (yet)

Validation metrics, all on the `depth` set (PR-AUC is the fair, threshold-independent yardstick):

| Model | PR-AUC | F1 | Precision | Recall | Log-loss |
|---|---|---|---|---|---|
| Logistic regression (Wk3 baseline) | **0.360** | 0.385 | 0.332 | 0.459 | 0.701 |
| Random forest | 0.333 | 0.388 | 0.329 | 0.473 | 0.683 |
| **XGBoost (Day 1)** | **0.339** | 0.387 | 0.324 | 0.480 | 0.707 |

- **Number to beat: 0.360** (logistic). XGBoost lands at 0.339 — slightly below baseline, roughly tied with RF.
- **Overfitting signal:** XGBoost train PR-AUC 0.629 vs. validation 0.339 — a large gap; untuned trees are
  memorizing training-era patterns. Reining this in is the Day-3 tuning target.

---

## 3. Per-region behavior — Japan "cries wolf"

XGBoost per-region validation (threshold 0.5):

| Region | PR-AUC | Precision | Recall |
|---|---|---|---|
| California | 0.333 | 0.485 | 0.198 |
| Japan | 0.282 | 0.300 | **0.958** |
| Greece | 0.359 | 0.365 | 0.216 |

- **Japan:** recall 0.96 but precision 0.30 — a flat 0.5 threshold + global balancing makes the model
  predict "quake coming" for Japan almost every day. Driven by Japan's high base rate (40.6%), not a modeling
  flaw. → the target of per-region weighting (#5) + per-region threshold (#4) on Day 3.

---

## 4. Feature importance — two views disagree

**XGBoost gain (internal training bookkeeping):**

| Feature | Gain |
|---|---|
| region = Japan | 0.683 |
| region = California | 0.154 |
| avg_depth_30d / max_mag_30d | 0.021 |
| avg_mag_30d / quakes_7d | 0.019 |
| quakes_30d | 0.018 |
| large_30d | 0.016 |
| avg_mag_7d | 0.015 |
| trend_7d | 0.014 |
| days_since_last | 0.012 |
| region = Greece | 0.008 |

→ Gain says region is ~83% of the model.

**Permutation importance on VALIDATION (drop in PR-AUC when a feature is scrambled; the honest check):**

| Feature | Δ PR-AUC | ± std |
|---|---|---|
| quakes_7d | **+0.0323** | 0.0030 |
| avg_mag_30d | +0.0215 | 0.0047 |
| max_mag_30d | +0.0144 | 0.0051 |
| large_30d | +0.0143 | 0.0024 |
| avg_mag_7d | +0.0091 | 0.0028 |
| avg_depth_30d | +0.0084 | 0.0046 |
| region | +0.0069 | 0.0051 |
| quakes_30d | +0.0046 | 0.0037 |
| trend_7d | +0.0023 | 0.0021 |
| days_since_last | **−0.0007** | 0.0005 |

→ Permutation says the **recent-activity features drive generalization** (led by `quakes_7d`); **region is 7th**.

**Reconciliation (key finding):** gain over-credits `region` because it's a big early split that separates base
rates. But PR-AUC measures *ranking* of risky days, and the activity features do that work — so scrambling region
barely moves the held-out score. **When gain and permutation conflict, trust permutation** (tied to real
performance). Takeaway: the model uses genuine seismic signal, not just a region shortcut; Japan's over-prediction
is a threshold/base-rate issue, not the model ignoring activity.

---

## 5. Day-2 shortlist (evidence-based)

- **Prune candidates (#7):** `days_since_last` (permutation −0.0007, pure noise) and `trend_7d` (+0.0023,
  negligible). Both near-zero across gain, permutation, and logistic coefficients.
- **Add candidates (#6):** longer-horizon large-magnitude counts (1 / 3 / 5+ years). Motivated by short-window
  activity (`quakes_7d`, `large_30d`) carrying the signal — test whether longer windows add more.

---

## Status
Day 1 complete: XGBoost trained + logged in MLflow ✅ · gain importance recorded ✅ · permutation cross-check ✅.
