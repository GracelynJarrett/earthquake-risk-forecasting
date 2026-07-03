# Earthquake Risk Forecasting

A machine-learning project that forecasts the likelihood of a **large earthquake
(magnitude 4.5 or higher)** occurring within the next **7 days** for three of the
world's most seismically active regions: **California, Japan, and Greece**.

The goal is to turn live, public earthquake data into a single, easy-to-read weekly
risk signal for people who plan around this hazard — emergency managers and disaster
preparedness teams. A key focus of the project is building the model *honestly*, with
a strict temporal train/validate/test split to avoid the data leakage that inflates
many earthquake-prediction models.

**Approach:** a gradient-boosted classifier (XGBoost), benchmarked against a logistic
regression baseline.
**Output:** an interactive dashboard showing each region's 7-day risk forecast.

## Tech Stack
- **Data:** USGS Earthquake Catalog API → cleaned with pandas → stored in SQLite
- **Modeling:** scikit-learn (baseline), XGBoost, tracked with MLflow
- **Pipeline:** Airflow (automated data pull → clean → store → predict)
- **Serving:** FastAPI (model API) + Next.js (dashboard)
- **Language:** Python

## Project Status
Course project for Applied AI — Project 2 (5-week build). Currently: **Week 1 —
project proposal complete.**

## Documentation
- [Proposal](doc/proposal.md) — full project proposal
- [Schedule](doc/schedule.md) — 5-week plan
- [AI Usage Log](doc/ai_usage.md) — weekly record of how AI tools were used
- [Daily Log](doc/daily_log.md) — day-by-day progress notes
- [Claude.md](Claude.md) — AI usage guidelines
