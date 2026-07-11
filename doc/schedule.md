# Schedule & Checklist — Earthquake Risk Forecasting

*Living checklist. `[x]` = done, `[ ]` = to do. Each task notes how we know it's complete.*

**Presentation cadence:** a progress presentation every **Day 5** — *except Weeks 4 & 5*, where there's no school on Day 5, so those presentations move to **Day 4**.

---

## Week 1 — Project Proposal ✅
**Goal:** Create the project proposal and pitch.
**Done when:** proposal completed/approved and pitch presented.

- [x] **Day 1 — Topic + setup** — created GitHub repo and folder structure

- [x] **Day 2 — Docs** — filled in proposal / md files

- [x] **Day 3 — Docs + pitch** — finished proposal and 
pitch outline

- [x] **Day 4 — Present pitch** — pitched to class; project approved

- [x] **Day 5 — Holiday**

---

## Week 2 — Data Foundation ✅ *(core complete; docs finishing)*
**Goal:** Pull USGS data, clean it, store it in SQLite, and deliver the data-understanding report.
**Done when:** data pulled → cleaned → in SQLite, and report + notebook submitted.

- [x] **Day 1 — Pull data**
  - [x] Test pull (last 24 h, worldwide)
  - [x] Historical pull, 3 regions, 2000–present → `data/raw/`
  - *Done when: raw CSVs exist for all three regions*

- [x] **Day 2 — Analyze (EDA notebook)**
  - [x] Data profile, quality checks, 4 core graphs, deep-dives, correlation/leakage analysis
  - [x] Written interpretations for each graph
  - *Done when: EDA complete with interpretations*

- [x] **Day 3 — Clean + store**
  - [x] `clean_data.py` — filter to earthquakes, normalize `status`
  - [x] Added `distance_to_fault_km` feature (leakage-safe)
  - [x] `load_to_sqlite.py` → `earthquakes.db` *(Day 4 goal done early)*
  - *Done when: cleaned data is queryable in SQLite (176,376 rows)*

- [x] **Day 4 — Documentation**
  - [x] Finalize target (region-specific thresholds) + feature list
  - [x] `data-understanding-report.md` (Sections 1–5)
  - [x] Notebook Section 6 (features + target)
  - [x] `stakeholder-notes.md`
  - [x] `implementation-plan.md`
  - [x] `claude.md` refresh
  - [x] `schedule.md` update *(this file)*
  - *Done when: all Week-2 documents are complete*

- [x] **Day 5 (Fri) — Present + submit**
  - [x] Present the data-understanding walkthrough (notebook)
  - [x] week 2 audience notes
  - [x] `ai-usage-log.md` (Week 2 entry)
  - [x] Commit + push; merge branch to main
  - *Done when: presented and all deliverables submitted (due Sunday)*

---

## Week 3 — Model Experimentation *(CR4–CR5)*
**Goal:** Set up MLflow, build the baseline logistic regression, and run the 4-variant feature ablation — leakage-safe.
**Done when:** baseline trained, the 4 variants compared in MLflow, and the best feature set chosen.
**Behind-schedule signal:** baseline isn't running by end of week.

- [ ] **Day 1 — MLflow + temporal split**
  - Set up MLflow; define train/validate/test **by date** (no shuffling)
  - *Done when: MLflow runs; split created with no date overlap*

- [ ] **Day 2 — Feature engineering (leakage-safe)**
  - Build recent-activity windows, `region` encoding, all "as-of-day" features
  - Impute quiet-day gaps **fit on training data only**
  - *Done when: feature table built for every (region, day) row*

- [ ] **Day 3 — Baseline model**
  - Train logistic regression (variant 1: base features); log PR-AUC / F1 to MLflow
  - *Done when: baseline metrics are logged*

- [ ] **Day 4 — 4-variant ablation**
  - base / +depth / +faultline / +both — same split, settings, and metric
  - *Done when: 4 runs logged; best feature set chosen*

- [ ] **Day 5 — Present + leakage checks / buffer**
  - **Present** Week 3 progress (baseline + 4-variant ablation results)
  - Verify no leakage; document results
  - *Done when: presented; leakage checks pass and results documented*

---

## Week 4 — Tuning, Pipeline & Deployment *(CR6–CR7)*
**Goal:** Train and tune XGBoost, pick the best model, and automate the pipeline with Airflow.
**Done when:** final model chosen and Airflow runs ingest → clean → load → predict end-to-end.
**Behind-schedule signal:** final model not chosen or Airflow not running.

- [ ] **Day 1 — XGBoost** (winning feature set) — *Done when: trained and logged in MLflow*

- [ ] **Day 2 — Hyperparameter tuning** — *Done when: tuned model + best params logged*

- [ ] **Day 3 — Pick final model + build Airflow DAG** (via Docker) — *Done when: final model chosen/documented and the DAG (fetch → clean → load → predict) is defined*

- [ ] **Day 4 — Test full pipeline + Present** — test the DAG end-to-end, then **present** Week 4 progress *(presentation is Day 4 — no school Day 5)* — *Done when: the DAG runs end-to-end and Week 4 is presented*

- [ ] **Day 5 — No school**

---

## Week 5 — Serving & Business Layer *(CR8–CR9)*
**Goal:** Serve the model via FastAPI, build a Next.js dashboard, and prepare the final presentation.
**Done when:** the dashboard shows weekly risk forecasts for California, Japan, and Greece, backed by the pipeline.
**Behind-schedule signal:** dashboard not showing forecasts or FastAPI not serving predictions.

- [ ] **Day 1 — FastAPI endpoint** serves the forecast — *Done when: endpoint returns per-region weekly risk*

- [ ] **Day 2 — Next.js project setup** — *Done when: the scaffold runs locally*

- [ ] **Day 3 — Build the dashboard** — *Done when: dashboard fetches and displays forecasts*

- [ ] **Day 4 — Polish + Present final project** — finish the dashboard and deliver the **final presentation** *(presentation is Day 4 — no school Day 5)* — *Done when: dashboard complete and the final project is presented*

- [ ] **Day 5 — No school**
