# ML Model Experimentation Report — Earthquake Risk Forecasting (Week 3)

> **FACT-SHEET (verified numbers).** Bullets + tables to expand into prose. Every metric
> pulled from MLflow (`earthquake-baseline`, `earthquake-per-region`). Validation metrics
> unless noted; TEST is sealed until Week 4.

---

## 1. Feature Engineering Summary

**Grain:** one row per (region, day). Features look **backward** from day D (data available by
then); the label looks **forward** (a region-significant quake in days D+1–D+7). Built by
`src/build_features.py`; stored in `earthquakes.db` → table `features` (28,989 rows, 16 cols).

**Final features & transformations** (all preprocessing fit on TRAIN only):

| Feature | Transformation | Why |
|---|---|---|
| `quakes_7d`, `quakes_30d` | rolling count (7d/30d); median-impute; standardize | recent activity level |
| `large_30d` | rolling count of region-significant quakes (30d); impute; scale | recent significant activity |
| `avg_mag_7d`, `avg_mag_30d` | rolling mean magnitude; median-impute quiet-day blanks; scale | recent size |
| `max_mag_30d` | rolling max magnitude (30d); scale | recent worst event |
| `trend_7d` | this week's count − last week's | activity accelerating/decaying |
| `days_since_last` | days since last quake; scale | quiescence signal |
| `region` | one-hot (California/Japan/Greece) | per-region base-rate shift |
| *[ablation]* `avg_depth_30d` | rolling mean depth (30d); scale | geologic (tested on/off) |
| *[ablation]* `avg_dist_30d` | rolling mean distance-to-fault (30d); scale | geologic (tested on/off) |
| *[ablation]* `centroid_lat_30d`, `centroid_lon_30d` | rolling mean lat/lon (30d); scale | where activity clusters (tested on/off) |

- **Imputation:** quiet-window blanks (e.g. `avg_mag_7d` on weeks with 0 quakes) filled with the
  **train median** (fit on train, applied to val/test). Counts are 0 (not blank) on quiet days.
- **Scaling:** StandardScaler (logistic regression needs comparable scales), fit on train.
- **Encoding:** `region` → one-hot.

**Dropped / excluded since Week 2:**
- `avg_mag_24h` — dropped (noisy/redundant at daily grain).
- `dmin`, `rms`, `nst`, `gap` — excluded (leakage; see Week-2 EDA §5).
- `month` / `season` — excluded; re-checked above threshold this week: apparent monthly peaks
  were the 2011 Tōhoku (200/255 March events) and 2019 Ridgecrest (34/56 July events) aftershock
  sequences, **not** a calendar pattern.
- raw `year` — excluded (would encode the 2009 catalog artifact).
- `lat/lon` centroid — proposed as core, demoted to **ablation**, tested, **not adopted** (no gain).

---

## 2. Experiment Design

**Model families tried (this week = simple, interpretable baselines before Week-4 XGBoost):**
- Logistic regression — **pooled** (one model, `region` one-hot) and **per-region** (one model each).
- Random Forest (tree ensemble) — pooled, 5 variants; a preview of the tree family, testing whether
  nonlinearity / feature interactions help where the linear model can't.
- **Location-only variants** (prof feedback) — region + geology/location features with the
  recent-activity features removed, to test whether lat/lon or fault-line carry signal alone.
- **Region×feature interactions** (prof feedback) — pooled logreg with per-region fault-distance
  (and activity) slopes, to get region-specific behavior without training separate models.
- Two **reference baselines**: a random no-skill `DummyClassifier`, and a naive persistence
  heuristic (`large_30d ≥ 1`).

**Splits — strictly temporal, by date, no shuffling** (a shuffled split would let the model see
the future → leakage). Pooled calendar across regions:

| Split | Dates | Rows | Positive rate |
|---|---|---|---|
| Train | 2000 – 2018 | 20,799 | 24.7% |
| Validation | 2019 – 2021 | 3,267 | 28.1% |
| Test (sealed) | 2022 – 2026 | 4,923 | 26.0% |

- **7-day embargo** trims the end of train and validation so a block's 7-day-ahead label can't
  reach the next block (42 region-day rows dropped).
- Test is untouched until every modeling decision is final.

**Metrics & why:**
- **PR-AUC** (primary) — imbalance-aware, ranking-based; right for a ~25%-positive target where we
  care about ranking risky days. Random-guess PR-AUC ≈ base rate.
- **F1** — threshold (0.5) quality, imbalance-aware.
- **Precision & recall** (prof feedback) — mission-critical for first responders: precision = of the
  alarms we raise, how many are real ("don't cry wolf"); recall = of the real events, how many we catch.
- **log-loss** — probability calibration (the quantity logistic regression minimizes).
- Reported **overall and per region** (base rates differ: CA 15.3% / Greece 21.0% / Japan 39.7%).

**Leakage guardrails (all verified by `src/check_leakage.py` — 6/6 PASS):** temporal split w/
embargo; `FEATURE_COLUMNS` allow-list (detection-quality cols never used); preprocessing fit on
train only (imputer median = train 3.9652 ≠ full-data 4.1333); backward features / forward label.

---

## 3. Experiment Results

**32 runs across 2 MLflow experiments** (`earthquake-baseline` = 17: 5 logistic-regression variants
+ 2 reference baselines + 5 Random Forest + 3 location-only + 2 interaction; `earthquake-per-region`: 15).
*[Insert MLflow screenshot(s) of both experiments here.]*

### 3a. Pooled ablation + reference baselines (validation)

| Model / variant | PR-AUC | F1 | log-loss | CA | Japan | Greece |
|---|---|---|---|---|---|---|
| dummy (no-skill) | 0.281 | 0.000 | — | 0.231 | 0.306 | 0.307 |
| heuristic (`large_30d≥1`) | 0.342 | 0.418 | — | 0.361 | 0.340 | 0.328 |
| **base** | **0.360** | 0.378 | 0.702 | 0.421 | 0.331 | 0.296 |
| **depth** | **0.360** | 0.385 | 0.701 | 0.421 | 0.328 | 0.302 |
| latlon | 0.357 | 0.382 | 0.701 | 0.424 | 0.320 | 0.288 |
| both | 0.347 | 0.372 | 0.696 | 0.380 | 0.328 | 0.344 |
| faultline | 0.344 | 0.373 | 0.699 | 0.385 | 0.319 | 0.332 |

Chosen model **train vs validation** (overfit check): `depth` train PR-AUC 0.424 / F1 0.471 /
loss 0.638 → val 0.360 / 0.385 / 0.701. Modest gap → low overfitting.

### 3b. Per-region models (best variant per region, validation)

| Region | Per-region best | Pooled best (same region) | Winner |
|---|---|---|---|
| California | 0.419 (depth) | ~0.42 (base/depth) | tie |
| Japan | 0.296 (base) | **0.331** (base) | pooled +0.035 |
| Greece | 0.313 (depth) | **0.344** (both) | pooled +0.031 |

### 3c. Random Forest — pooled ablation (tree family), train vs. validation

| RF variant | train PR-AUC | val PR-AUC | val F1 | overfit gap | CA | Japan | Greece |
|---|---|---|---|---|---|---|---|
| base | 0.756 | 0.336 | 0.394 | 0.419 | 0.366 | 0.305 | **0.399** |
| depth | 0.785 | 0.333 | 0.388 | 0.452 | 0.368 | 0.291 | 0.379 |
| faultline | 0.793 | 0.342 | 0.397 | 0.451 | 0.383 | 0.318 | 0.391 |
| both | 0.815 | 0.335 | 0.383 | 0.481 | 0.369 | 0.310 | 0.361 |
| latlon | 0.813 | 0.329 | 0.388 | 0.484 | 0.378 | 0.292 | 0.351 |

Untuned settings (`n_estimators=300`, `class_weight="balanced"`, `min_samples_leaf=20`); tuning is Week 4.

### 3d. Precision & recall — the "don't cry wolf" view (validation)

| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| dummy (no-skill) | 0.281 | 0.000 | 0.000 | 0.000 |
| heuristic (`large_30d≥1`) | 0.342 | 0.293 | 0.733 | 0.418 |
| logreg base | 0.360 | 0.327 | 0.448 | 0.378 |
| **logreg depth (chosen)** | 0.360 | 0.332 | 0.459 | 0.385 |
| random forest depth | 0.333 | 0.329 | 0.473 | 0.388 |

Precision ≈ **0.30 for every model** → ~2 of every 3 alarms are false at threshold 0.5. The heuristic
is high-recall / low-precision (0.733 / 0.293 — it cries wolf a lot).

**Per-region at threshold 0.5, chosen `depth` model (a key finding):**

| Region | Precision | Recall | Behavior |
|---|---|---|---|
| California | 0.637 | 0.230 | cautious — few alarms, 64% real |
| Japan | 0.306 | 1.000 | flags **everything** (catches all, 69% false) |
| Greece | 0.333 | 0.093 | flags **almost nothing** (misses 91%) |

→ a single 0.5 threshold behaves completely differently per region; **per-region decision thresholds**
are a Week-4 / deployment fix. This is why PR-AUC (threshold-free) is the fairer comparison metric.

### 3e. Feature importance (chosen `depth` model)

**Logistic regression — standardized coefficients** (sign = direction, |value| = influence):
`region_japan` +0.53 · `region_california` −0.39 · `avg_mag_30d` +0.28 · `large_30d` +0.25 ·
`quakes_7d` +0.23 · `region_greece` −0.21 · then `avg_depth_30d` **+0.06** and all others ≈ 0.
→ the model leans on **region (base rate) + recent significant-activity magnitude/count**; the added
`avg_depth_30d` is **barely used** (confirms the ablation), and explains why the `large_30d` heuristic
is competitive.

**Random Forest — Gini importance** ranks `avg_depth_30d` / magnitudes on top and `region` near the
bottom — but that reflects a known Gini bias toward continuous features (and RF overfit), so the
**ablation (actual score change) is the more trustworthy "does it help" test.**

### 3f. "Remove the temporal features" — location-only test (prof feedback, validation)

| Model | PR-AUC | CA | Japan | Greece |
|---|---|---|---|---|
| dummy floor | 0.281 | 0.231 | 0.306 | 0.307 |
| FULL base (with temporal) | 0.360 | 0.421 | 0.331 | 0.296 |
| location_only (region + geology) | 0.292 | 0.179 | 0.279 | 0.324 |
| latlon_only | 0.304 | 0.276 | 0.306 | 0.289 |
| faultline_only | 0.294 | 0.192 | 0.286 | **0.350** |

- Remove temporal → overall collapses to ~the floor (0.29–0.30 vs 0.360): **recent-activity features
  carry almost all the signal.**
- **lat/lon: no independent signal** (`latlon_only` ≈ floor) — a third confirmation.
- **fault-line: a real Greece-specific signal** — `faultline_only` Greece **0.350** beats the floor
  (0.307) and even the full model's Greece (0.296); but it hurts California.

### 3g. Region×feature interactions (prof feedback, validation) — bar: base 0.360 / Greece 0.296

| Model | PR-AUC | Precision | Recall | CA | Japan | Greece |
|---|---|---|---|---|---|---|
| base + region×fault-dist | 0.345 | 0.321 | 0.432 | 0.366 | 0.331 | 0.309 |
| base + region×(fault-dist + activity) | 0.355 | 0.313 | 0.398 | 0.362 | 0.349 | 0.299 |

Interactions did **not** beat base: Greece barely moved (0.296 → 0.309), California was hurt
(0.421 → ~0.37), and overall dropped. A hand-built *linear* interaction is too blunt (the effect stays
diluted amid the dominant temporal features) — but it confirms the region-specific structure is real
and motivates **XGBoost** (which learns interactions natively) in Week 4.

### Interpretations (one line each)
- **dummy:** confirms the floor — PR-AUC = base rate, F1 = 0.
- **heuristic:** surprisingly strong — **beats logreg on overall F1 (0.418) and on Japan (0.340) &
  Greece (0.328)**; logreg only clearly wins California + overall PR-AUC by 0.018.
- **base vs depth:** tied on PR-AUC; depth edges F1 and Greece, adds a physical feature, harms none.
- **faultline / both:** distance-to-fault **helps Greece** (0.296→0.344) but **hurts California**
  (0.421→0.380) — a regional trade-off that cancels in the pooled average.
- **latlon:** no improvement — location centroid doesn't help this yes/no target.
- **per-region:** loses to pooled on Japan/Greece — each model trains on only ~1/3 the data
  (~2,300 rows) and is starved; pooling borrows strength across regions.
- **Random Forest:** overfits untuned (train ~0.76–0.82 vs val ~0.33, gap ~0.45) and does **not**
  beat logistic regression overall (best val 0.342 vs 0.360). But it is **clearly better on Greece**
  (0.399 vs logreg ≤0.344 and heuristic 0.328) — trees capture a pattern there the linear model
  misses — while worse on California. Motivates a tuned, regularized tree (XGBoost) in Week 4.

---

## 4. Model Selection & Justification

**Candidate for Week-4 tuning: pooled logistic regression, `depth` feature set.**

Justification (trade-offs, not just the top number):
- **Performance:** top PR-AUC (tied `base`), best F1 among pooled variants; beats the per-region
  models (Japan/Greece) and the no-skill floor comfortably.
- **Complexity / interpretability / speed:** linear, coefficients inspectable, trains in seconds;
  `depth` adds one physically meaningful feature at negligible cost.
- **Overfitting risk:** low (train 0.424 vs val 0.360 PR-AUC).
- **Honest caveat (the key trade-off):** it only **narrowly** beats the naive `large_30d`
  heuristic overall (0.360 vs 0.342) and **trails it on F1 and on Japan/Greece**. So the model is
  mostly capturing "recent activity persists," adding real value only where features are dense
  (California). → **Week-4 goal: XGBoost must beat the heuristic on Japan/Greece**, or the added
  complexity isn't justified.
- **Why not Random Forest (yet):** the untuned forest overfit badly (train↔val gap ~0.45) and didn't
  beat logistic regression overall, so it's not the Week-3 pick — but its clear Greece gain is
  evidence the tree family is worth pursuing **with tuning**, which is exactly Week 4's XGBoost step.
- **Precision/recall reality (prof feedback):** precision ≈ 0.30 at threshold 0.5 (~2 of 3 alarms
  false), and the pooled model's per-region behavior is extreme (Japan flags everything, Greece almost
  nothing). A **per-region decision threshold** is a required deployment step, separate from model choice.
- **Region-specific structure is real but hard to exploit linearly:** separate per-region models (data
  starvation) and hand-built region×feature interactions (dilution + California cost) both failed to beat
  the pooled model. The fault-line→Greece signal appeared **three independent ways** (ablation, Random
  Forest, location-only) → a strong case that a tuned **XGBoost** (native interactions) is the right
  Week-4 tool to finally exploit it.

---

## 5. Companion deliverables (separate files, root `doc/`)
- **`implementation-plan.md`** — add Week-3 update: on track; pooled logreg (`depth`) selected;
  per-region rejected (data starvation); heuristic is the Week-4 bar.
- **`claude.md`** — update only if AI-usage instructions changed (likely minor/no change).
- **`ai-usage-log.md`** — Week-3 entry: tasks assisted, prompts that worked, corrections.

*MLflow deliverable:* export/screenshot both experiments' run comparison for the report folder.