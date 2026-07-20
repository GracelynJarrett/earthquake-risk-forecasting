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
**Goal:** Set up MLflow, build the baseline logistic regression, and run the 5-run feature ablation — leakage-safe.
**Done when:** baseline trained, the 5 variants compared in MLflow, and the best feature set chosen.
**Behind-schedule signal:** baseline isn't running by end of week.

- [x] **Day 1 — MLflow + temporal split**
  - Set up MLflow; define train/validate/test **by date** (no shuffling)
  - *Done when: MLflow runs; split created with no date overlap*

- [x] **Day 2 — Feature engineering (leakage-safe)**
  - Build recent-activity windows, all "as-of-day" features → `features` table in `earthquakes.db`
  - `region` one-hot + impute quiet-day gaps **deferred to Day 3** (fit on training data only)
  - *Done when: feature table built for every (region, day) row*

- [x] **Day 3 — Baseline model**
  - Train logistic regression (variant 1: base features); log PR-AUC / F1 to MLflow
  - *Done when: baseline metrics are logged*

- [x] **Day 4 — 5-run ablation**
  - base / +depth / +faultline / +both / +lat-lon — same split, settings, and metric
  - Bonus: 15 per-region models tested — pooled model won (best set: base/depth)
  - *Done when: 5 runs logged; best feature set chosen*

- [x] **Day 5 — Present + leakage checks / buffer**
  - **Present** Week 3 progress (baseline + 5-run ablation results)
  - Verify no leakage; document results
  - *Done when: presented; leakage checks pass and results documented*

---

## Week 4 — Model Refinement, Deployment & Pipeline *(CR6–CR7)*
**Goal:** Train/tune XGBoost, run seven model-refinement experiments, lock and register the final model in MLflow, then stand up a minimal inference endpoint + Airflow DAG.
**Done when:** all seven experiments run and captured in fact-sheets, final model chosen and registered/versioned in MLflow, the inference endpoint returns per-region risk, the DAG is defined, and the Model Deployment report is submitted.
**Behind-schedule signal:** final model not chosen, or experiment results not documented by the Day 4 present.

*Experiment fact-sheets:* each experiment day (1–4) produces a short fact-sheet of verified numbers as we go, so the Day 6–7 report is prose assembly, not fresh analysis.

- [x] **Day 1 — XGBoost + feature importance** 
  - [x] train/tune XGBoost on the winning feature set, then measure **feature importance/significance (#1)**
  - [x] *Done when: model trained and logged in MLflow; importance ranking recorded*

- [ ] **Day 2 — Feature experiments** 
  - [ ] **add temporal features (#6)** (large-magnitude counts over the last 1 / 3 / 5+ years, backward-looking only) and 
  - [ ]**prune the features that hurt the model most (#7)**, then re-evaluate 
  - [ ] *Done when: expanded + pruned feature sets are trained and compared in MLflow*

- [ ] **Day 3 — Per-region imbalance tuning** 
  - [ ] **per-region class weighting (#5)** 
  - [ ] **per-region prediction threshold (#4)** 
  - [ ] *Done when: per-region weights + thresholds are tuned and logged*

- [ ] **Day 4 — Confidence reporting + Present** 
  - [] bucket predictions into **Low / Medium / High confidence (#3)**
  - []**calibration check** (a "70%" means roughly 70%), then 
  - [ ]**present** Week 4 progress *(presentation is Day 4 — no school Day 5)* 
  - [ ] *Done when: confidence buckets produced, calibration checked, and Week 4 presented*

- [ ] **Day 5 — No school (working day) — Leakage demo + final model** 
  - [ ] run the **random-split leakage demonstration (#2)** to show how much shuffling inflates the metrics vs. our honest temporal split, then **pick the final model** and 
  - [ ]**register/version it in MLflow** 
  - [ ]*Done when: leakage demo documented; final model chosen and registered in MLflow*

- [ ] **Day 6 — Inference endpoint + Airflow DAG**       
  - [] build a minimal FastAPI endpoint that loads the registered model and returns per-region risk + confidence 
  - [] **define the Airflow DAG** (fetch → clean → load → predict) *(full end-to-end Docker test deferred to Week 5)* 
  - [] *Done when: endpoint returns a real per-region forecast and the DAG is defined*

- [ ] **Day 7 — Deployment report + submit** 
  - [] write the **Model Deployment report (Section 4)** 
  - [] *Done when: report complete and all Week-4 deliverables submitted (due Sunday)*

---

## Week 5 — Serving & Business Layer *(CR8–CR9)*
**Goal:** Serve the model via FastAPI, build a Next.js dashboard, and prepare the final presentation.
**Done when:** the dashboard shows weekly risk forecasts for California, Japan, and Greece, backed by the pipeline.
**Behind-schedule signal:** dashboard not showing forecasts or FastAPI not serving predictions.

- [ ] **Day 1 — Harden endpoint + finish Airflow pipeline** — expand the Week-4 FastAPI endpoint and run the DAG end-to-end in Docker *(deferred from Week 4)* — *Done when: endpoint serves per-region weekly risk + confidence, and the DAG runs ingest → clean → load → predict end-to-end*

- [ ] **Day 2 — Next.js project setup** — *Done when: the scaffold runs locally*

- [ ] **Day 3 — Build the dashboard** — *Done when: dashboard fetches and displays forecasts*

- [ ] **Day 4 — Polish + Present final project** — finish the dashboard and deliver the **final presentation** *(presentation is Day 4 — no school Day 5)* — *Done when: dashboard complete and the final project is presented*

- [ ] **Day 5 — No school**
