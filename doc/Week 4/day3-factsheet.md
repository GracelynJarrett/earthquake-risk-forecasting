# Week 4 — Day 3 Fact-Sheet (Thresholds #4 / Weighting #5 / Tiers + Calibration #3)

*Verified numbers only — raw material for the Week-4 report. Date run: 2026-07-21.*
*All results on the **validation** split (2019–21), region-blind lean model (`pruned_nodepth_noreg`,
6 recent-activity features). Test (2022–26) stays sealed until Day 5.*

---

## 1. Per-region decision thresholds (#4) — objective: F2 (recall-first)

F2 weights recall 2× precision (a missed quake is the costly error for first responders).
Chosen thresholds and their validation metrics vs. the default 0.5 (unweighted model):

| Region | Cutoff | Precision | Recall | F2 | (vs 0.5) |
|---|---|---|---|---|---|
| California | 0.060 | 0.234 | 0.988 | 0.601 | 0.5 → R 0.143, F2 0.170 |
| Japan | 0.240 | 0.309 | 0.997 | 0.689 | 0.5 → R 0.075, F2 0.088 |
| Greece | 0.140 | 0.321 | 0.988 | 0.698 | 0.5 → R 0.048, F2 0.059 |

Strategy comparison (F2 vs F1.5 vs precision-floor 0.35):
- **F1.5 ≈ F2** — near-identical cutoffs/scores; easing the recall weight changed almost nothing.
- **Precision floor 0.35 is infeasible for Japan** (precision ceiling ~0.31; forcing it collapses
  recall — on the weighted model it fell to 0.009). California/Greece reach it only at low recall.
- **Key finding:** thresholds only slide along a fixed precision-recall curve — they cannot lift it.
  The ~0.30 precision ceiling is a *model* limit, so cutting false alarms needs a better model, not a
  better cutoff.

---

## 2. Per-region class weighting (#5) — REJECTED

Validation PR-AUC by weighting scheme (overall + per region):

| Scheme | Overall | California | Japan | Greece |
|---|---|---|---|---|
| none (weight 1) | 0.3663 | 0.365 | 0.290 | 0.449 |
| global (3.04) | 0.3677 | 0.357 | 0.303 | 0.440 |
| region (per-region) | 0.3516 | 0.369 | 0.310 | **0.361** |

- `none` ≈ `global` (within noise) → weighting mostly moves the operating point (thresholds already
  do that), it does **not** lift the curve.
- Per-region weighting is **worse overall** — nudged CA/JP up a hair but crashed **Greece −0.088**.
- **Verdict:** keep no per-region weighting.

---

## 3. Calibration (#3) — unweighted model wins

Does the predicted probability match reality? (Stakeholder promise: "20% means 20%.")

| Model | val PR-AUC | Brier | pred mean | Calibration |
|---|---|---|---|---|
| weighted (spw 3.04) | 0.3677 | 0.257 | 0.50 | badly over-confident (says 65% → 33%, 99% → 63%) |
| **unweighted (spw 1)** | 0.3663 | **0.204** | **0.28** | tight (says 35% → 35%, 71% → 75%) |
| + isotonic (per region) | — | **0.190** | — | monotonic, best |

- Reference: a constant "predict-the-28%-base-rate" model scores Brier 0.202 — the *weighted* model
  (0.257) is worse than that; the unweighted (0.204) matches it and adds resolution; isotonic beats it.
- **Decision: adopt the UNWEIGHTED model** (`scale_pos_weight=1`) — same ranking, honest probabilities.
  Class weighting inflates probabilities without improving ranking, so it was dropped.

---

## 4. Low / Medium / High tiers (#3) — per-region base-rate bands

Tiers relative to each region's *normal* weekly risk (train base rate): Low < normal,
Medium normal→1.5×, High ≥ 1.5×. Cutoffs: CA 0.139/0.208, Japan 0.406/0.609, Greece 0.197/0.295.

Observed quake rate by tier — RAW vs. isotonic-CALIBRATED:

| Region | Low | Medium | High |
|---|---|---|---|
| California (raw) | 0.224 | 0.162 ⚠️ | 0.405 |
| **California (calibrated)** | **0.107** | **0.192** | **0.362** ✅ |
| Greece (raw) | 0.220 | 0.262 | 0.413 |
| **Greece (calibrated)** | **0.062** | **0.258** | **0.425** ✅ |
| Japan (raw) | 0.323 | 0.282 ⚠️ | empty |
| **Japan (calibrated)** | **0.306** (100% of days) | empty | empty ⚠️ |

- **California + Greece work** after calibration — observed risk climbs cleanly across tiers.
- **Japan is single-tier (all Low).** The model can't rank Japan's risk (PR-AUC ~0.29), and Japan was
  genuinely quieter in 2019–21 (~30%) than its training-era norm (41%) — so "at/below normal every week"
  is partly an honest read, partly a resolution failure. Documented limitation.

---

## 5. Model state after Day 3
- **Model:** XGBoost, 6 recent-activity features, region-blind, **no class weighting**, Day-1 default
  hyperparameters (not yet tuned).
- **Decision layer:** per-region F2 thresholds (CA 0.060 / JP 0.240 / GR 0.140).
- **Risk output:** per-region isotonic calibration → Low/Med/High base-rate tiers.
- **Validation PR-AUC ≈ 0.366** (beats logistic baseline 0.360); still overfitting (train ≈ 0.61).

## 6. Caveats / carry-forward
- Thresholds + calibration were fit on **validation**; **test (Day 5) is the honest confirmation.**
- Two open problems for tuning: **overfitting** (train 0.61 vs val 0.37) and **Japan's weak ranking**.
- Decisions currently span `tune_thresholds.py` + `risk_tiers.py`; `train_xgboost.py` still uses the old
  balanced weight → **consolidate into one registered model on Day 5.**

## Status
Day 3 complete: #4 thresholds (F2) ✅ · #5 weighting rejected ✅ · #3 calibration + tiers ✅
(unweighted + isotonic adopted; CA/Greece tiered, Japan single-tier documented).