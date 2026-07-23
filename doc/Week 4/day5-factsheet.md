# Week 4 — Day 5 Fact-Sheet (Leakage Demo #2 / Hyperparameter Tuning)

*Verified numbers only — raw material for the Week-4 report. Date run: 2026-07-22.*
*Model: region-blind lean set (`pruned_nodepth_noreg`, 6 recent-activity features), unweighted.*
*All results on the **validation** split unless noted. Test (2022–26) stays sealed until final.*

---

## 1. Leakage demonstration (#2) — random split inflates EVERY metric

Cautionary demo (`src/leakage_demo.py`): same model, features, and split sizes — only the split
method changes. ALL data (2000–2026) is shuffled, then re-split randomly into the same train/
validate sizes as the temporal split. (Throwaway model — our real model never trains on test.)

Overall validation metrics, honest temporal vs. random (leaky):

| Metric | Better | Temporal (honest) | Random (leaky) | Effect |
|---|---|---|---|---|
| PR-AUC | higher | 0.366 | 0.500 | inflated +0.134 |
| ROC-AUC | higher | 0.589 | 0.730 | inflated +0.141 |
| F1 | higher | 0.143 | 0.223 | inflated |
| Precision | higher | 0.481 | 0.723 | inflated |
| Recall | higher | 0.084 | 0.132 | inflated |
| Log-loss | lower | 0.608 | 0.494 | "better" (lower) |
| Brier | lower | 0.204 | 0.161 | "better" (lower) |

Per-region PR-AUC (temporal → random): California 0.365 → 0.344, **Japan 0.290 → 0.585 (+0.295)**,
Greece 0.449 → 0.397.

- **Every metric — ranking, threshold, and probabilistic — moves the "looks better" way.** You cannot
  catch leakage by switching metrics; you catch it with an honest split.
- **The inflation is almost entirely Japan** (+0.295), the region with the densest temporal clustering
  (aftershock sequences); California/Greece are flat. Leakage flatters the region we're actually worst at.
- Honest **ROC-AUC is 0.589** — barely above a coin flip. Cheating dresses it up as 0.730.
- *Why it leaks even with no date feature:* neighboring days share ~7–30-day windows, so a random split
  puts near-duplicate rows on both sides — the model "predicts" copies it already memorized.

**Re-run on the TUNED final model (regularized) — leaks much less:** temporal 0.390 → random 0.427
(**+0.037** overall; Japan 0.359 → 0.502, +0.143). The inflation shrinks from +0.134 (untuned) to +0.037
because a heavily-regularized model (depth 2, min_child_weight 50) *can't memorize* the near-twin
neighbors as aggressively — so random splitting has less to exploit. **Teaching point: an overfit model
leaks far more than a regularized one.** Both split models (untuned demo values above; tuned re-run here)
are logged to MLflow experiment `earthquake-leakage-demo` (runs `temporal_honest`, `random_leaky`).

---

## 2. Hyperparameter tuning — regularization search

Config `config/gxboost_tunning.yaml`; 108 combos (learning_rate × max_depth × min_child_weight ×
gamma × reg_lambda); fit on train, score PR-AUC on validation (no CV shuffling → no leakage).

**Winner:** `learning_rate=0.03, max_depth=2, min_child_weight=50, gamma=0, reg_lambda=1`.

| Metric | Day-1 defaults | Tuned | Change |
|---|---|---|---|
| Overall val PR-AUC | 0.366 | **0.390** | +0.024 |
| Overfit gap (train − val) | ~0.24 | **0.069** | −0.17 |
| Japan val PR-AUC | 0.290 | **0.359** | +0.069 |
| California | 0.365 | 0.378 | +0.013 |
| Greece | 0.449 | 0.422 | −0.027 |

- **The overfitting gap collapsed** (0.24 → 0.069) — the main win. The winning recipe is "keep it simple":
  shallowest trees, heaviest leaf regularization, slowest learning rate. Confirms overfitting was the cap.
- **Japan improved most** (+0.069) — from near-random toward usable ranking.
- **MLflow:** all 108 combos logged to experiment `earthquake-xgboost-tuning`; the winner (with the fitted
  model artifact) logged to `earthquake-xgboost` as run `tuned-xgboost` (val PR-AUC 0.3901).
- *Caveat:* 0.390 is the best of 108 on validation → mildly optimistic; **test confirms at finalize.**
  The gap collapse and Japan gain are robust, not luck.

---

## 3. Tuned-model risk tiers (#3 re-checked) — Japan comes alive

Isotonic-calibrated per-region base-rate tiers, now on the TUNED model (Brier 0.187):

| Region | Low (obs rate) | Medium | High |
|---|---|---|---|
| California | 0.100 (10 d) | 0.191 (949 d) | **0.538** (130 d) |
| **Japan** | 0.290 (1037 d) | **0.526** (19 d) | **0.667** (33 d) |
| Greece | 0.141 (170 d) | 0.210 (219 d) | 0.377 (700 d) |

- **All three regions now support Low/Med/High**, observed risk climbing cleanly across tiers.
- **Japan went from single-tier ("all Low") to three tiers** — the tuned model flags 33 high-risk weeks
  where **67% actually had a significant quake**. A real capability gain, not just a metric bump.
- *Caveat:* Japan's Medium/High tiers are small (19 / 33 days) — encouraging but statistically noisy;
  validation-fit, confirmed on test at finalize.

---

## 4. Model state after tuning
- **Model:** XGBoost, 6 region-blind recent-activity features, **unweighted**, **tuned**
  (`max_depth=2, learning_rate=0.03, min_child_weight=50`), baked into `build_xgb_model`.
- **Decision layer:** per-region F2 thresholds; **risk output:** isotonic-calibrated Low/Med/High tiers.
- **Validation PR-AUC ≈ 0.390** (up from 0.366 untuned, baseline 0.360); overfit gap 0.069.

## 5. Carry-forward (finalize)
- **Register** the `tuned-xgboost` MLflow run as the versioned final model.
- **Re-tune the F2 thresholds on the TUNED model** — the earlier thresholds (CA 0.060 / JP 0.240 /
  GR 0.140) were computed on the untuned model; re-run `tune_thresholds.py` now that `build_xgb_model`
  is tuned.
- **Run the sealed TEST set once** for the honest final numbers (thresholds + calibration + tiers).

## Status
Leakage demo (#2) ✅ · hyperparameter tuning ✅ (winner logged to MLflow) · tuned-model tiers re-checked
✅ (Japan now 3 tiers). Remaining: register final model + confirm on test.