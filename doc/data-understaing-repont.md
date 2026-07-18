# Data Understanding Report — Earthquake Risk Forecasting

*Predicting significant earthquake activity from historical USGS seismic data (California, Japan, Greece).*

---

## Section 1: Data Source & Ingestion

### Data Source & Access
- **Source:** United States Geological Survey (USGS) Earthquake Catalog
- **Access method:** Public **FDSN Event web service API**
- **Endpoint:** `https://earthquake.usgs.gov/fdsnws/event/1/query`
- **Query parameters used:** `format=geojson`, start date, end date, minimum magnitude, geographic bounding box (min/max latitude and longitude)
- **Update frequency:** Near real-time — new events typically appear within ~1 minute of detection

### Data Collected
- **Regions:** California, Japan, Greece
- **Criteria:** magnitude ≥ 2.0; January 2000 – July 2026; regional bounding boxes
- Because the API caps a single request at 20,000 events, data was pulled **one year at a time per region** and merged into the final historical catalog.

### Ingestion Workflow (three scripts)
1. **`fetch_usgs.py`** — query the USGS API, download records, store raw files in `data/raw/`
2. **`clean_data.py`** — remove unwanted event types, normalize fields, add distance-to-fault, store to `data/processed/`
3. **`load_to_sqlite.py`** — load cleaned data into SQLite and build indexes for querying

### Planned Airflow Automation (Week 4)
The three scripts map directly to a linear Apache Airflow DAG:

```
fetch_usgs.py  →  clean_data.py  →  load_to_sqlite.py
```

- Linear DAG structure, scheduled refresh interval (weekly)
- Docker already prepared for deployment
- Current execution is manual; orchestration is planned for Week 4

### Major Findings & Changes from the Original Proposal
1. **Regional catalog completeness differs.** California and Greece are largely complete near magnitude 2.0, but Japan's effective floor is **≈ 2.7** — smaller events are missing, which inflates Japan's apparent share of moderate/large quakes.
2. **2009 catalog artifact.** Greece's recorded count dropped sharply and permanently; Japan dipped temporarily then recovered. This reflects reporting practices, not real seismicity.

   | Year | Greece earthquakes |
   |---|---|
   | 2008 | ~3,636 |
   | 2009 | ~127 |

3. **Regional volume differences.** California has ~4× more recorded quakes than Japan or Greece (dense U.S. monitoring network).
4. **Non-earthquake events removed.** 1,621 events (quarry blasts, explosions, nuclear tests, other non-seismic types) were removed.
5. **Feature leakage discovered.** The detection-quality variables (`nst`, `gap`, `dmin`, `rms`) leak magnitude information and were excluded (see Leakage Analysis in Section 3).
6. **Target definition revised.** The proposed flat "≥4.5 within 7 days" was ~99% positive for Japan (near-trivial), so the target now uses **region-specific thresholds**.
7. **Domain feature added.** Following instructor feedback, **distance to nearest plate boundary** was added — a physically meaningful geologic predictor.

---

## Section 2: Data Profile

### Dataset Shape
- **Raw dataset:** 177,997 records · 16 columns · January 2000 – July 2026

| Region | Records (raw) |
|---|---|
| California | 117,178 |
| Japan | 33,664 |
| Greece | 27,155 |
| **Total** | **177,997** |

### Variables
- **Text:** `id`, `region`, `place`, `mag_type`, `type`, `status`
- **Date/time:** `time` (UTC, ISO-8601)
- **Numeric:** `magnitude`, `longitude`, `latitude`, `depth_km`, `nst`, `gap`, `dmin`, `rms`, `distance_to_fault_km`

### Missing Data
Missing values occur almost entirely in the detection-quality fields:

| Column | Missing |
|---|---|
| dmin | 25.7% |
| rms | 12.8% |
| nst | 7.2% |
| gap | 4.4% |
| mag_type | ~0% |

**Columns with no missing values:** `time`, `magnitude`, `latitude`, `longitude`, `depth_km`. Because the affected columns are excluded from modeling (leakage), no immediate imputation is required.

### Duplicate Records
0 duplicate IDs and 0 duplicate rows — the year-by-year extraction prevented overlap between requests.

### Data Anomalies (all real, none corrupt)
- **Negative depths:** 4,560 events (min −3.49 km) — valid events above the sea-level datum in high-elevation California/Nevada
- **Deep earthquakes:** to 686 km — the genuine Bonin Islands deep-focus sequence near Japan
- **Extreme magnitude:** max 9.1 — the 2011 Tōhoku earthquake
- **Low-confidence locations:** ~524 events with azimuthal `gap` > 320° (plus a few high `rms`/`dmin`) — retained and documented as real, not corrupt

### Data Quality
- No corrupted files; API-delivered structured data was highly reliable
- **~99.1% usable out of the box** (only 0.9% removed as non-earthquakes)

### Cleaning Summary
- **Removed:** 1,621 non-earthquake events
- **Standardized:** `status` casing (`REVIEWED`→`reviewed`, `AUTOMATIC`→`automatic`)
- **Retained:** automatic detections, deep earthquakes, negative-depth events

### Final Counts
Stored in the SQLite table `earthquakes`, indexed on `region` and `time`:

| Region | Records |
|---|---|
| California | 115,564 |
| Japan | 33,657 |
| Greece | 27,155 |
| **Total** | **176,376** |

### Key Statistics — Magnitude by Region

| Region | Median | Min | Max | ≥4.5 share |
|---|---|---|---|---|
| California | 2.31 | 2.0 | 7.2 | 0.3% |
| Greece | 3.40 | 2.0 | 7.0 | 6.8% |
| Japan | 4.40 | 2.7 | 9.1 | 47.8% |

**Depth:** range −3.49 km to 686 km; most events shallow, with a deep subduction-zone tail in Japan. **Distribution shape:** many small quakes with a steep drop-off toward large ones; Japan's curve is shifted right by its higher completeness floor.

### Reviewed vs. Automatic Events
- **All 5,125 automatic events are in California** (4.4% of its records); Japan and Greece are 100% reviewed
- Automatic events are small and recent: median magnitude 2.2, 96.7% below 3.0, concentrated 2024–2026 (review backlog); only one is ≥4.5
- **Cause:** USGS's U.S. networks feed automatic detections, while foreign catalogs submit only human-reviewed events
- **Decision:** retained (negligible effect on the target)

---

## Section 3: Preprocessing, EDA & Feature Candidates

### Preprocessing Strategy
- **Completed cleaning:** removed non-earthquake events; standardized categorical values; preserved legitimate extremes
- **Categorical encoding:** `region` → one-hot (for logistic regression); `type`/`status`/`place`/`mag_type` are metadata, not features
- **Scaling:** numeric features scaled for logistic regression (fit on training data only); XGBoost is tree-based and needs no scaling

### Imputation Plan
Any imputation will occur **after** the train/test split and be **fit on training data only**, to prevent temporal data leakage. (The recent-average-magnitude features are blank on quiet days and will be filled at that stage.)

### Outlier Treatment
Extreme values are **kept**: major earthquakes are the primary phenomenon of interest, and removing them would discard the most meaningful information.

### Exploratory Data Analysis — Findings
- **Magnitude distribution by region:** Japan's distribution is shifted toward larger magnitudes because smaller events are under-reported. *Implication:* regional bias must be addressed during modeling.
- **Activity through time:** the 2011 Tōhoku spike and the 2009 Greece discontinuity stand out. *Implication:* temporal effects shape feature engineering and validation design.
- **Magnitude vs. depth:** no strong relationship — Japan's magnitude 4–5 quakes have a median depth of ~35 km, and the 9.1 was shallow (~29 km). *Implication:* depth may add little; test it via ablation.
- **Geographic mapping:** quakes cluster tightly around tectonic plate boundaries. *Implication:* distance-to-fault is geologically justified.
- **Event clustering:** median inter-event time ~0.9 h in California, with the mean far above the median (heavy right-skew). *Implication:* activity is clustered, not random — recent activity should be predictive.
- **Foreshock behavior:** before Tōhoku, daily activity rose from ~1 to ~34 events/day (a real M7.3 foreshock). *Implication:* activity surges can be informative, though foreshocks are not universal.
- **Seasonality:** none detected — even among region-significant events, the apparent monthly peaks (Japan March, California July) trace to the 2011 Tōhoku and 2019 Ridgecrest aftershock sequences, not a calendar cycle. *Implication:* month/season features are unnecessary.

### Leakage Analysis (headline finding)
The pooled correlations of magnitude with `dmin` (**0.63**) and `rms` (**0.61**) appeared strong — but they **vanished within each region** (≈ 0). This is a **regional confound (Simpson's paradox)**: regions with larger average magnitudes also have larger `dmin`, so pooling three regions created a spurious trend. Using these would let the model secretly encode *region*.

Separately, magnitude correlated strongly with `nst` **within** Japan (0.73) and Greece (0.66) — a genuine relationship, but **target leakage**, because station count is measured *from the very quake being predicted* and is unknowable beforehand.

**Result:** all four detection-quality variables (`dmin`, `rms`, `nst`, `gap`) were excluded from modeling — directly addressing the data leakage this project set out to avoid.

### Prediction Target
Binary classification, **per region, per day, 7-day-ahead**: label = 1 if at least one earthquake at/above the region threshold occurs in the next 7 days, else 0. Features use only data available up to the prediction day (leakage-safe). One model is trained with `region` as a feature.

| Region | Threshold | Positive rate |
|---|---|---|
| California | 4.5 | 15.3% |
| Greece | 5.0 | 21.0% |
| Japan | 5.5 | 39.7% |

Region-specific thresholds were adopted because a flat 4.5+ was near-trivial for Japan (~99% positive).

### Candidate Features
All features are measured as of the prediction day, per region (aggregated to the region-day grain):

**Core features**
- `region` (one-hot)
- Earthquake count — last 7 days; last 30 days
- Large-earthquake count — last 30 days (at/above the region threshold)
- Days since last earthquake
- Maximum magnitude — last 30 days
- Average magnitude — last 7 days; last 30 days
- Activity trend (current week vs. previous week)

**Ablation features (tested on/off)**
- Average depth — last 30 days
- Average distance-to-fault — last 30 days
- Recent-activity centroid — average latitude & longitude, last 30 days

**Excluded features:** `dmin`, `rms`, `nst`, `gap` (leakage); month/season (no seasonality — apparent above-threshold monthly peaks traced to the 2011 Tōhoku and 2019 Ridgecrest aftershock sequences, not a calendar pattern); raw year (would encode the 2009 catalog artifact)

### Planned Modeling Comparison
A five-run ablation study (on the baseline logistic regression) isolates the value of the geologic and location features:

1. Base · 2. Base + Depth · 3. Base + Distance-to-Fault · 4. Base + Both · 5. Base + Recent-activity location (lat/lon)

All variants share the same temporal split, settings, and evaluation metric, and are compared with **imbalance-aware metrics (PR-AUC, F1)**. The best-performing feature set carries forward to the Week 4 XGBoost implementation.

---

## Section 4: Deep Learning / Unstructured Data
**Not applicable** — this is a classical/tabular machine-learning project (see Section 3).

---

## Section 5: Revised Core Requirements & Schedule

### Revised Core Requirements (granular · specific · measurable)
- **CR1 — Ingestion:** Automatically pull USGS FDSN data (magnitude 2.0+, California/Japan/Greece, 2000–present) and store it in a SQLite database. *Done when: the pipeline runs end-to-end and the database holds the cleaned catalog.*
- **CR2 — Cleaning:** Remove non-earthquake events, normalize fields, and exclude leakage-prone columns. *Done when: 0 non-earthquake rows remain and column roles are documented.*
- **CR3 — Target:** Predict, per region per day, whether a region-significant quake (CA 4.5+ · Greece 5.0+ · Japan 5.5+) occurs in the next 7 days. *Done when: the binary label and per-region base rates are defined (15.3% / 21.0% / 39.7%).*
- **CR4 — Baseline:** Train a logistic-regression baseline on a strict **temporal** train/validate/test split, logged in MLflow, with leakage checks. *Done when: metrics are logged and leakage checks pass.*
- **CR5 — Feature evaluation:** Run the 5-run ablation (base / +depth / +faultline / +both / +lat-lon) and select the best feature set by PR-AUC/F1. *Done when: 5 runs are logged and a winner is chosen.*
- **CR6 — Improved model:** Train and tune XGBoost, compare to the baseline, and select the best. *Done when: the tuned model and comparison are documented.*
- **CR7 — Automation:** Orchestrate ingest → clean → load (→ predict) with Airflow on a schedule. *Done when: the DAG runs end-to-end.*
- **CR8 — Serving:** Serve weekly per-region risk through a FastAPI endpoint. *Done when: the endpoint returns forecasts.*
- **CR9 — Dashboard:** A Next.js dashboard shows the weekly forecast for all three regions. *Done when: the dashboard displays live forecasts.*

### Key Revisions from the Proposal
1. Region-specific target thresholds (was a flat 4.5+)
2. Detection-quality features excluded for leakage
3. Distance-to-fault feature added
4. Region expansion (e.g., New Zealand + a non-Ring-of-Fire region) documented as a **future** extension, not part of the current scope

### Finalized Schedule

| Week | Focus | Milestone | Requirements |
|---|---|---|---|
| 1 | Proposal & pitch | Approved proposal | — |
| 2 | Data foundation | Data pulled, cleaned, and stored in SQLite; report delivered | CR1–CR3 |
| 3 | Model experimentation | MLflow set up; baseline + 5-run ablation | CR4–CR5 |
| 4 | Tuning, pipeline & deployment | XGBoost tuned, best model chosen, Airflow pipeline | CR6–CR7 |
| 5 | Serving & business layer | FastAPI + Next.js dashboard; final presentation | CR8–CR9 |
