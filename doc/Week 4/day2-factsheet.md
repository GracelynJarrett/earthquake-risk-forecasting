# Week 4 — Day 2 Fact-Sheet (Add Features #6 / Prune Features #7)

*Verified numbers only — raw material for the Week-4 report. Date run: 2026-07-21.*
*All results are XGBoost on the **validation** split (2019–21). Test (2022–26) remains sealed.*
*Reference points: `depth` XGBoost = 0.339 (Day 1); logistic baseline = **0.360** (number to beat).*

---

## 1. What was built (#6 features)

Added four backward-looking columns to `src/build_features.py` (same leakage-safe rolling
pattern as `large_30d`): counts of region-significant quakes over the trailing windows.

- `large_365d` (1 yr), `large_1095d` (3 yr), `large_1825d` (5 yr), `large_alltime` (running total).
- **Verification:** no NaN, no negatives; window ordering holds everywhere (30d ≤ 1y ≤ 3y ≤ 5y ≤ all);
  `large_alltime` non-decreasing per region. End-of-record counts (all-time / 5-yr):
  Japan 1069 / 173, Greece 397 / 71, California 344 / 58 — mirror the known base-rate spread.

---

## 2. Long-horizon ablation (#6) — adding history HURTS

Incremental variants built on `depth`:

| Variant | val PR-AUC | train PR-AUC | #feats |
|---|---|---|---|
| longhz_1 (+1yr) | 0.3402 | 0.653 | 11 |
| depth (reference) | 0.3387 | 0.629 | 10 |
| longhz_135 (+1/3/5yr) | 0.3366 | 0.691 | 13 |
| longhz_13 (+1/3yr) | 0.3154 | 0.684 | 12 |
| longhz_135a (+all-time) | 0.2976 | 0.702 | 14 |

- As windows are added, **train PR-AUC rises (0.63 → 0.70) while validation falls (0.34 → 0.30)** —
  textbook overfitting. Only the 1-yr count is neutral (+0.0015 vs depth, within noise).
- **Verdict: #6 rejected.** Long-horizon counts do not help; more of them actively hurt.

---

## 3. Feature-value A/B tests (#7) — depth and region both HURT

| Test (differ by one thing) | with | without | effect of the feature |
|---|---|---|---|
| **depth** (base vs depth) | 0.3387 | 0.3493 | **−0.0106** (hurts) |
| **depth** (pruned vs pruned_nodepth) | 0.3376 | 0.3475 | **−0.0099** (hurts) |
| **region** (pruned_nodepth ± region) | 0.3475 | **0.3677** | **−0.0202** (hurts) |
| region (muddied: hz1_pruned ± region) | 0.3300 | 0.3362 | −0.0062 |
| days_since_last + trend_7d (depth vs pruned) | 0.3387 | 0.3376 | −0.0011 (noise) |

- **`avg_depth_30d` hurts** — two independent pairs agree at ~−0.01 (2–3× the noise band).
  Notable: depth was Week 3's *winning* add for logistic regression → feature value is model-specific.
- **`region` hurts** — clean test on the best lean set: removing region *gains* +0.0202
  (Greece PR-AUC 0.383 → 0.440, Japan 0.291 → 0.303, California ~flat). With region, XGBoost splits
  on it early and overfits each region's training-era patterns.
- **`days_since_last` + `trend_7d`** — free to drop (noise), confirming Day 1.

---

## 4. Feature importance incl. new features (variant `longhz_135a`)

Permutation importance on VALIDATION (Δ val PR-AUC when scrambled; negative = feature HURTS):

| Helps ↑ | Δ | | Hurts ↓ | Δ |
|---|---|---|---|---|
| quakes_7d | +0.0061 | | large_365d | −0.0176 |
| avg_mag_30d | +0.0057 | | large_1095d | −0.0121 |
| max_mag_30d | +0.0034 | | avg_depth_30d | −0.0086 |
| large_30d | +0.0023 | | quakes_30d | −0.0042 |
| trend_7d | +0.0007 (noise) | | large_1825d | −0.0040 |
| avg_mag_7d | −0.0003 (noise) | | region | −0.0034 |
| days_since_last | −0.0007 (noise) | | large_alltime | −0.0010 |

- The long-horizon features are the **most harmful of all**; `large_365d` worst at −0.0176.
- Gain (same run) still over-credits region (`region_japan` 0.606) — the Day-1 gain-vs-permutation
  gap persists; permutation is the trustworthy view.

---

## 5. Day-2 result — the best set is the leanest

**Best XGBoost feature set: the 6 recent-activity features only**
`quakes_7d, quakes_30d, large_30d, avg_mag_7d, avg_mag_30d, max_mag_30d` — **no region, no depth,
no long-horizon** (variant `pruned_nodepth_noreg`).

| | val PR-AUC | California | Japan | Greece | #feats |
|---|---|---|---|---|---|
| Best set (`pruned_nodepth_noreg`) | **0.3677** | 0.357 | 0.303 | 0.440 | 6 |
| depth (Day-1 XGBoost) | 0.3387 | 0.333 | 0.282 | 0.359 | 10 |
| logistic baseline | 0.360 | — | — | — | — |

- **First XGBoost model to beat the 0.360 baseline** — achieved by *removing* features, not adding.
- Theme of the day: depth, region, and long-horizon counts were all **crutches for overfitting**;
  stripping them forces the model to rank days by real seismic activity, which generalizes better.

---

## 6. Caveats / carry-forward
- Many comparisons on one validation set → trust the **repeated** effects (depth, long-horizon, region);
  do **not** chase ±0.003–0.004 wiggles (e.g. `quakes_30d`). Final confirmation is on **test** (Day 5).
- Still overfitting (best set: train ~0.61 vs val ~0.37, gap ~0.24) → **Day 3 tuning is the real lever.**
- **Day 3 interaction:** a region-free pooled model gives identical probabilities for identical
  activity across regions → **per-region thresholds (#4)** carry the region-specific decision;
  **per-region class weighting (#5)** needs a rethink (no region for a pooled model to weight on).

## Status
Day 2 complete: #6 features built + ablated (rejected) ✅ · #7 pruning done (depth/region/long-horizon
all dropped) ✅ · best set beats baseline (0.368 vs 0.360) ✅ · importance re-checked with new features ✅.