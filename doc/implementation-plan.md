# Implementation Plan — Earthquake Risk Forecasting

*Living 5-week plan of execution (the "how"). Companion to `schedule.md` (the "when").*
*Living document: when the plan changes, add a dated note to the Revision Log — do not rewrite history.*

---

## End Goal
An honest, **leakage-free** ML system that predicts, each day for each region (California, Japan, Greece), whether a **region-significant** earthquake will occur in the next 7 days. An automated pipeline (pull → clean → store → model → serve) delivers a weekly risk forecast to first responders through a dashboard.

---

## System Architecture / Data Flow
```
USGS FDSN API
   → fetch_usgs.py       (raw yearly pulls per region → data/raw/)
   → clean_data.py       (filter, normalize, + distance-to-fault → data/processed/)
   → load_to_sqlite.py   (→ data/earthquakes.db, table "earthquakes", indexed)
   → [Wk3] feature engineering + model (tracked in MLflow)
   → [Wk4] Airflow orchestration (Docker)
   → [Wk5] FastAPI endpoint → Next.js dashboard
```

---

## Modeling Details
- **Target:** per-region, per-day, 7-day-ahead binary — 1 if ≥1 quake at/above the region threshold occurs in the next 7 days. **One model** with `region` as a feature.
- **Region thresholds:** California 4.5 · Greece 5.0 · Japan 5.5 (base rates 15.3% / 21.0% / 39.7%).
- **Features (as-of the prediction day, per region):** region (one-hot); quake counts (7d, 30d); large-quake count (30d, ≥ region threshold); days since last quake; max magnitude (30d); average magnitude (7d, 30d); activity trend (this week vs. last); *[ablation]* avg depth (30d), avg distance-to-fault (30d), recent-activity centroid (avg lat/lon, 30d).
- **Excluded:** `dmin`/`rms`/`nst`/`gap` (leakage), month/season (no seasonality — the apparent above-threshold monthly peaks were the 2011 Tōhoku and 2019 Ridgecrest aftershock sequences, not a calendar pattern), raw year (2009 artifact).
- **Metrics:** PR-AUC and F1 (imbalance-aware).
- **Leakage guardrails:** strict temporal split; a `FEATURE_COLUMNS` allow-list; imputation/scaling fit on training data only; correlations checked *within* each region.

---

## Execution Plan by Phase

### Phase 1 — Data Foundation (Week 2) ✅
- Built `fetch_usgs.py`, `clean_data.py`, `load_to_sqlite.py`; data pulled, cleaned, and stored in SQLite (176,376 rows).
- Completed EDA; finalized target and feature list; added distance-to-fault.

### Phase 2 — Model Experimentation (Week 3) ✅
1. Set up MLflow.
2. Define the temporal train/validate/test split (by date, no shuffling).
3. Feature engineering: build the as-of-day feature table (leakage-safe); impute quiet-day gaps on the training set only.
4. Train the baseline logistic regression (variant 1: base features).
5. Run the **5-run ablation** (base / +depth / +faultline / +both / +lat-lon); compare by PR-AUC/F1; select the best feature set.
6. Run leakage checks and document results.
- *Outcome:* pooled logistic regression (`depth`) selected; per-region models rejected (data starvation); a naive `large_30d` heuristic set as the Week-4 bar; leakage checks 6/6 pass (see Revision Log).

### Phase 3 — Tuning, Pipeline & Deployment (Week 4)
1. Train XGBoost with the winning feature set (**pooled, `depth`**); hyperparameter-tune.
2. Compare XGBoost to the baseline **and to the `large_30d` heuristic** (the bar to beat, especially Japan/Greece); choose the final model.
3. Build the Airflow DAG (fetch → clean → load → predict) via Docker; test end-to-end.

### Phase 4 — Serving & Business Layer (Week 5)
1. Build the FastAPI endpoint serving weekly per-region risk.
2. Build the Next.js dashboard for first responders.
3. Polish and deliver the final presentation.

---

## Maintenance & Retraining (production design)
Two cadences, kept separate:
- **Inference / refresh — weekly** (runs in the Airflow DAG): pull fresh USGS data, recompute the recent-activity features, and generate a new forecast using the *current* model. Because the features are rolling windows computed at prediction time, the forecast reflects recent activity **without** retraining.
- **Retraining — periodic (monthly/quarterly) + drift-triggered:** relearn the model on an **expanding window** (all data up to the retrain date), always validating on a *later* hold-out so the temporal split is preserved (no future leakage). Retrain early if MLflow metrics drift or a catalog change appears (e.g., a 2009-style reporting shift).
- **For this project:** train once for the Week 3–4 deliverables; automate only the weekly inference refresh in Airflow; the retraining cadence above is documented as the intended production design.

---

## Future / Stretch (not current scope)
- **Region expansion** for a balanced Ring-of-Fire study (e.g., add New Zealand [Ring] + a non-Ring region like Turkey).
- **Recent-activity location context** — display where recent quakes clustered (descriptive only; not city-level prediction).
- **Refine distance-to-fault** to a proper geodesic distance.

---

## Revision Log *(add new entries; do not rewrite past ones)*

### Week 2 — July 2026
- **Target revised** from a flat 4.5+ to **region-specific thresholds** — validation showed 4.5+/7-day was ~99% positive for Japan (trivial). Thresholds: CA 4.5 · Greece 5.0 · Japan 5.5.
- **Excluded detection-quality features** (`dmin`/`rms`/`nst`/`gap`) after finding their magnitude correlations were a regional confound (Simpson's paradox) or target leakage (`nst`).
- **Added** the `distance_to_fault_km` feature (instructor-approved).
- **Region expansion** (Ring-of-Fire study) recorded as a **future** extension, not current scope.
- **Airflow** kept in Week 4; the ingestion DAG may be pulled forward to Week 3 if there's slack.
- **Maintenance cadence defined:** weekly inference refresh (Airflow) vs. periodic/drift-triggered retraining on an expanding, future-validated window.

### Week 3 — July 2026
- **MLflow + temporal split set up:** pooled by-date split — train ≤2018 / validate 2019–2021 / test 2022–present — with a **7-day embargo** trimming the end of the train and validate blocks so a block's 7-day-ahead label cannot peek across the boundary.
- **Seasonality re-checked above threshold** (Week-2 presentation feedback): a chi-square flagged Japan (March) and California (July) as non-uniform, but the year-breakdown showed the peaks were the **2011 Tōhoku** (200 of 255 March events) and **2019 Ridgecrest** (34 of 56 July events) aftershock sequences — not a calendar pattern. `month`/`season` stays excluded.
- **Feature list refined:** dropped `average magnitude (24h)` (noisy at daily grain, redundant with the 7d average); added a leakage-safe **recent-activity centroid** (avg lat/lon, 30d) as an ablation feature.
- **Ablation expanded to 5 runs:** base / +depth / +faultline / +both / +lat-lon.
- **Baseline + 5-run ablation completed (Day 3–4):** pooled logistic regression; PR-AUC / F1 / log-loss logged to MLflow (`earthquake-baseline`). `base` and `depth` tied best (val PR-AUC 0.360); `latlon` added nothing (location centroid **not adopted**); `faultline`/`both` help Greece but hurt California — a regional trade-off that cancels in the pooled average.
- **Per-region models tested and rejected:** 15 models (3 regions × 5 variants, `earthquake-per-region`). The pooled model won on Japan (+0.035) and Greece (+0.031) because each per-region model trains on only ~⅓ the data (~2,300 rows) and is starved. The "three separate models" hypothesis is rejected with evidence.
- **Reference baselines added:** a no-skill `DummyClassifier` (PR-AUC = base rate) and a naive persistence heuristic (`large_30d ≥ 1`; val PR-AUC 0.342 / F1 0.418). The heuristic **beats** the logistic regression on F1 and on Japan/Greece — the learned model only narrowly earns its keep. **Week-4 bar: XGBoost must beat this heuristic on the hard regions.**
- **Random Forest previewed (5 pooled variants):** an untuned tree ensemble overfit badly (train PR-AUC ~0.76–0.82 vs val ~0.33) and did not beat logistic regression overall (best val 0.342 vs 0.360), but was **clearly better on Greece** (0.399). It confirms the linear model's limits and motivates the tuned XGBoost planned for Week 4 — it is *not* the Week-3 pick.
- **Leakage checks pass (6/6, `src/check_leakage.py`):** temporal embargo, split disjointness, feature allow-list, train-only preprocessing, forward-only label. The check also caught and fixed a storage-dtype bug (six feature columns saved as TEXT); model values were unaffected.
- **Model selected for Week 4:** pooled logistic regression, `depth` feature set.
- **Status: on track** — Week-3 core requirements (CR4–CR5) met.
