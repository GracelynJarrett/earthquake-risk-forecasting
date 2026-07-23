# Week 4 — Final Test Evaluation Fact-Sheet (Day 5, step 3)

*Verified numbers only — raw material for the Week-4 report. Date run: 2026-07-22.*
*The one-shot honest evaluation on the SEALED test split (2022–26), run once after the model was
frozen and registered. No decisions were made after seeing these numbers.*

---

## 1. What was evaluated
- **Final model:** tuned XGBoost, region-blind, **unweighted** (`scale_pos_weight=1`). Frozen in
  `config/final_model.json`, registered as **`earthquake-risk-model` v1** in MLflow.

  - **Chosen features (6, region-blind):** `quakes_7d`, `quakes_30d`, `large_30d`, `avg_mag_7d`,
    `avg_mag_30d`, `max_mag_30d` — recent-activity only; **no region, no depth, no long-horizon
    counts** (all three hurt validation performance, per Day 2).

  - **Tuned hyperparameters:** `max_depth=2`, `learning_rate=0.03`, `min_child_weight=50` (searched)
    plus `n_estimators=300`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0`, `reg_lambda=1`
    (fixed / XGBoost defaults). Selected from a **108-combo grid search on validation** — collapsed the
    overfit gap from 0.24 to 0.07 and lifted validation PR-AUC 0.366 → 0.390.
    
- **Rebuilt deterministically:** model fit on TRAIN, per-region isotonic calibrators fit on
  VALIDATION, then applied ONCE to TEST. (Fit on train only to match the frozen config and stay
  comparable to validation; deployment may refit on train+validation.)
- Results logged to MLflow run **`final-test-eval`**.

---

## 2. Ranking — validation → test

| Metric | Validation | Test | Change |
|---|---|---|---|
| **Overall PR-AUC** | 0.3901 | **0.4003** | +0.010 |
| **Overall ROC-AUC** | 0.5995 | **0.6589** | +0.059 |
| California PR-AUC | 0.3775 | 0.2241 | **−0.153** |
| Japan PR-AUC | 0.3591 | 0.4593 | **+0.100** |
| Greece PR-AUC | 0.4217 | 0.2518 | **−0.170** |

- **Overall generalized** — test PR-AUC (0.40) slightly *beats* validation; ROC-AUC rose. No overall
  overfitting to validation.
- **Per-region is unstable across periods:** California and Greece dropped sharply on 2022–26 while
  Japan rose to become the best region. Regional performance is period-dependent — the overall number
  held only because Japan's gain offset the others' losses.

---

## 3. Alerts at the frozen F2 thresholds (test)

| Region | Threshold | Precision | Recall | F2 |
|---|---|---|---|---|
| California | 0.100 | 0.163 | **1.000** | 0.494 |
| Japan | 0.305 | 0.419 | **0.982** | **0.774** |
| Greece | 0.175 | 0.196 | **0.914** | 0.527 |

- **The safety goal generalized:** recall stayed **0.91–1.00** on unseen data — "catch almost every
  quake-week" held up, which is the core first-responder need. Precision is low (many false alarms), as
  expected. **Japan is genuinely strong** (F2 0.77, precision 0.42).

---

## 4. Risk tiers (test) — did NOT generalize cleanly

Calibrated Low/Med/High, observed quake rate (days):

| Region | Low | Medium | High |
|---|---|---|---|
| California | 0.000 (5) | 0.163 (1575) | 0.197 (61) |
| Japan | 0.412 (1380) | 0.309 (55) | 0.471 (206) |
| Greece | 0.173 (375) | 0.217 (461) | 0.201 (805) |

- Tiers climbed cleanly on validation but are **flat or non-monotonic on test** (Japan Medium < Low;
  Greece High < Medium; California barely separates). The calibration + tier boundaries fit on 2019–21
  **do not transfer reliably to 2022–26.**
- Calibrated **Brier held (0.190** vs validation ~0.187) — overall probability quality is stable, but
  the fine-grained gradation is not.

---

## 5. Honest bottom line
- **Generalizes on unseen data:** overall ranking (PR-AUC 0.40, ROC-AUC 0.66) and the **yes/no alert**
  with high recall (0.91–1.00) — the core safety function.
- **Does NOT generalize reliably:** the **three-tier gradation** out-of-period, and **per-region
  performance swings** between eras.
- **This is exactly why test was sealed.** Validation tiers were optimistic; test told the truth. The
  honest product story: lean on the high-recall alert + communicate uncertainty; present tiers *with*
  the period-instability caveat rather than as a promise.

## 6. Carry-forward
- **Deployment (Day 6):** may refit the model on train+validation (more data) before serving.
- **Known limitations to document for stakeholders:** per-region performance is period-dependent, and
  the Low/Med/High tiers are less reliable than the binary alert.
- **Future work:** address regional instability / tier reliability (more data, region-specific modeling).

## Status
Finalize complete: config frozen ✅ · `earthquake-risk-model` v1 registered ✅ · test evaluated once +
logged to MLflow ✅. Honest final headline: **test PR-AUC 0.40, ROC-AUC 0.66**, high-recall alerts hold,
tiers carry a documented caveat.