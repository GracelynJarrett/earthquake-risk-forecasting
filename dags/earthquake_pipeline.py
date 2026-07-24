"""
earthquake_pipeline.py — Airflow DAG for the inference pipeline (Week 4, Day 6).

Every 2 days: fetch the last 30 days of quakes -> clean -> predict per-region risk.
Each task just runs one of our existing scripts, so the pipeline reuses code as-is.
(Defined for Week 4; the full end-to-end Docker run is a Week-5 task.)
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# Where the project lives INSIDE the Airflow container (set by docker-compose mount).
PROJECT_DIR = os.environ.get("AIRFLOW_PROJECT_DIR", "/opt/airflow/project")

# If a task fails (e.g. USGS is briefly down), retry once after 5 minutes.
default_args = {"retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="earthquake_risk_pipeline",
    description="Every 2 days: fetch last 30d -> clean -> predict per-region risk",
    schedule=timedelta(days=2),          # run every 2 days
    start_date=datetime(2026, 1, 1),     # fixed past date; catchup off so no backfill
    catchup=False,
    default_args=default_args,
    tags=["earthquake", "inference"],
) as dag:

    # 1) Pull the last 30 days of quakes for all three regions.
    fetch = BashOperator(
        task_id="fetch_recent",
        bash_command=f"cd {PROJECT_DIR} && python src/fetch_usgs.py recent",
    )

    # 2) Clean the raw pull into the modeling-ready catalog.
    clean = BashOperator(
        task_id="clean",
        bash_command=f"cd {PROJECT_DIR} && python src/clean_data.py",
    )

    # 3) Build today's features, run the model, and save the per-region forecast.
    predict = BashOperator(
        task_id="predict",
        bash_command=f"cd {PROJECT_DIR} && python src/predict.py",
    )

    # Order: fetch -> clean -> predict (each waits for the previous to succeed).
    fetch >> clean >> predict