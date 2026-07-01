# Schedule — Earthquake Risk Forecasting

## Week 1 — Project Proposal
**Goal:** Create a project proposal and pitch for Earthquake Risk Forecasting
**Done when:** The proposal is completed and turned in. The pitch is ready for Thursday's presentation
**Behind-schedule signal:** Pitch is not done, md files are not done

- **Day 1:** Find a topic, create GitHub and set up folders
- **Day 2:** Work on filling in the md files
- **Day 3:** Work on filling in the md files, create pitch
- **Day 4:** Present pitch
- **Day 5:** Holiday, no school

---

## Week 2 — Data Foundation
**Goal:** Pull the data from USGS, clean the data, and add the data to SQLite
**Done when:** Data can be successfully pulled from USGS, cleaned, and stored in SQLite
**Behind-schedule signal:** The data not in SQLite

- **Day 1:** Pull data from USGS
- **Day 2:** Analyze data
- **Day 3:** Clean data
- **Day 4:** Put data in SQLite
- **Day 5:** Buffer

---

## Week 3 — Model Experimentation
**Goal:** Build a baseline logistic regression model with a strict temporal train/validate/test split and set up MLflow
**Done when:** The baseline model is trained and metrics are in MLflow
**Behind-schedule signal:** baseline model isn't running

- **Day 1:** Set up MLflow, and define train/validate/test split
- **Day 2:** Pull data from SQLite, train baseline logistic regression model
- **Day 3:** Log the baseline model metrics in MLflow, check for data leakage
- **Day 4:** buffer / Start XGBoost model
- **Day 5:** buffer / start XGBoost model


---

## Week 4 — Tuning, Pipeline & Deployment
**Goal:** Train and tune an XGBoost model, compare it to the baseline, pick the best model and automate pipeline with Airflow
**Done when:** The best model is chosen and Airflow is running the pipeline end-to-end
**Behind-schedule signal:** Final model not chosen, Airflow isn't running the pipeline

- **Day 1:** Create XGBoost model
- **Day 2:** Hyperparameter tuning
- **Day 3:** Pick best model, Build the Airflow pipeline
- **Day 4:** Build Airflow pipeline
- **Day 5:** Test the full automated pipeline/buffer

---

## Week 5 — Serving & Business Layer
**Goal:** Serve the model through a FastAPI endpoint, build a Next.js dashboard and prepare for final presentation
**Done when:** The Dashboard shows the weekly risk forecast for California, Japan, and Greece, backed by the pipeline, and the presentation is ready
**Behind-schedule signal:** Dashboard is not showing forecasts or FastAPI isn't serving predictions

- **Day 1:** Build FastAPI endpoint and serves the forecast
- **Day 2:** Set up the Next.js Project
- **Day 3:** Work on Next.js
- **Day 4:** Polish Dashboard, prepare for presentation
- **Day 5:** Present project
