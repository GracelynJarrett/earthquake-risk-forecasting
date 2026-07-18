# ML Model Experimentation Report — Earthquake Risk Forecasting (Week 3)

> All metrics are pulled from MLflow (`earthquake-baseline`, `earthquake-per-region`) and reported
> on the **validation** split unless noted; the **TEST** split is sealed until Week 4.

---

## 1. Feature Engineering Summary

My model has many different features: 9 set features and 4 added features ([ablation]) to see whether accuracy, loss, and precision increased. All these features are used to answer the question "Will there be an earthquake above the region threshold in the next 7 days". This means that a single row is from a single region on a given date in the last 26 years.

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

If one of the temporal features had no data, I would fill it in from the training data. For example, if 'avg_mag_7d' had no data, then we would take the median from the training data, which is 3.9652. This ensures we have no null values. We also had to scale some of the features, because the model works better when all features are on the same scale. Scaling is calculated only from the train data, since we don't want the test data to influence how the scaling works. However, the same scaling is still applied to the train, validation, and test data — the test data just isn't used to determine the scale, only to apply it afterward.

**Dropped features.** Although I have many features, I had to drop some because they added noise and caused leakage.

- 'avg_mag_24h' was dropped due to causing noise and causing redundancy for the daily grain.
- `dmin`, `rms`, `nst`, and `gap` were all dropped to reduce leakage; see the notebook analysis for the heat map and p-values.
- 'month' and 'season' were all excluded because they showed no correlation with earthquake predictions. There are some months that have more earthquakes than others, but that's because there were many aftershocks after a very large earthquake.
- 'year' was excluded because it would have encoded the 2009 catalog artifact.
- `lat/lon` are [ablation] features that have been tested and have been proven to have no gain for the model. Originally, lat/long was going to be part of the set features, but I decided to add it to the [ablation] features to see if the model would change. The models' metrics declined, indicating that lat and lon introduced noise. So, the final base model does not have lat or lon.

---

## 2. Experiment Design

**Model families.** I tried many different model families to try and find the best combination of features for my base model.

- A **logistic regression** model was used for most of my model experimentation and was part of my original plan for my base model. Logistic regression was used to assess the impact of adding the [ablation] features. It was also used to assess whether we should train one model per region.
- An untuned **random forest** model was used to confirm whether a non-linear model would help improve the basic model. These models did not outperform logistic regression overall; there was significant overfitting. However, it significantly improved Greece. This confirms that, with tuning, a nonlinear model could improve my model's accuracy, precision, and loss.
- A model with **location-only variants** (region and geology features, with recent-activity features removed) was created to test whether location alone carries a predictive signal. Without the temporal features, performance dropped close to the no-skill floor, showing that recent earthquake activity is responsible for almost all the model's predictive power, not location alone.
- **Region-and-feature interaction** was used to see if it was possible to train one model but have different features per region. The overall model's PR-AUC decreased.
- Two **reference baseline** models, a dummy (no-skill classifier) and a naïve heuristic, were used to give the base model a floor. This means the base model needs to beat the floor models' metrics, or it was not worth moving on. In the end, the chosen base model beat most of the floor models' metrics.

**Train/validate/test split structure.** I split the data strictly by date: train 2000–2018 (72%), validate 2019–2021 (11%), test 2022–present (17%). Splitting my data means the model doesn't see the future, which could cause data leakage. To guarantee there is no data leakage, I added an embargo at the end of each data split. This means the last 7 days of the split were trimmed, ensuring there is no accidental leakage. To confirm this, I ran a leakage check for the temporal split, the feature allow-list, and train-only preprocessing; the check showed no data leakage.

| Split | Dates | Rows | Positive rate |
|---|---|---|---|
| Train | 2000 – 2018 | 20,799 | 24.7% |
| Validation | 2019 – 2021 | 3,267 | 28.1% |
| Test (sealed) | 2022 – 2026 | 4,923 | 26.0% |

**Metrics & why.**

- My data is imbalanced, with only about 25% of the region's days expected to have an earthquake in the next 7 days. So, I use **PR-AUC** (Precision-Recall Area Under the Curve). PR-AUC summarizes the entire trade-off value into a single number. PR-AUC is a great way to ensure the model is ranking risky days correctly.
- **F1** complements PR-AUC: while PR-AUC considers all probability thresholds, F1 evaluates one specific threshold. F1 combines precision and recall into a single number at that threshold, rewarding models that do reasonably well on both — though a lopsided model can still score high by maximizing just one (for example, the naïve heuristic earned the highest F1 by pushing recall very high despite low precision). This means F1 is judging the practical, real-world quality of the specific alarm threshold.
- **Precision and recall** together make F1, but they mean different things in the real world, so looking at them separately helps. Precision looks at how many of the predicted earthquakes were real — a high precision means fewer false alarms. Recall measures how many real earthquakes the model caught — a high recall means fewer missed events. This means that, depending on what matters more in each situation — avoiding false alarms or avoiding missed earthquakes — I can choose a model or threshold that favors one over the other.
- My model doesn't just answer yes or no; it gives a probability. **Log-loss** checks whether the probabilities are trustworthy. Log-loss penalizes the model for confidently wrong answers and rewards it for correct ones. Since logistic regression is a probability-based model, checking log-loss tells you whether the model's confidence levels are meaningful, which matters a lot if downstream, someone wants to set a custom alert threshold based on risk tolerance rather than just trusting a fixed 0.5 cutoff.
- Finally, all of these metrics are reported both overall and broken down by region, since the three regions have very different base rates (California ~15%, Greece ~21%, Japan ~40%) — a single overall score can hide the fact that the model performs very differently from one region to the next.

---

## 3. Experiment Results

**32 runs across 2 MLflow experiments** (`earthquake-baseline` = 17: 5 logistic-regression variants
+ 2 reference baselines + 5 Random Forest + 3 location-only + 2 interaction; `earthquake-per-region`: 15).
**All runs, both experiments** (MLflow parallel-coordinates compare view):

![earthquake-baseline — all 17 runs compared](images/Screenshot%202026-07-17%20215429.png)
*Figure 1 — `earthquake-baseline`: all 17 runs (logistic-regression variants, references, Random Forest, location-only, interactions), colored by validation PR-AUC.*

![earthquake-per-region — all 15 runs compared](images/Screenshot%202026-07-17%20215629.png)
*Figure 2 — `earthquake-per-region`: all 15 per-region runs (3 regions × 5 variants).*

### 3.1 Chosen model — pooled logistic regression (`depth`)

Pooled logistic regression model with the depth feature was chosen as the best model. The three regions were one-hot regions, and the avg_depth_30d was added to the model. PR-AUC, F1, and log-loss metrics all show that the model does not overfit; PR-AUC train 0.424 vs. PR-AUC val 0.360, train F1 0.471 vs. val F1 0.385, and train log-loss 0.638 vs. val log-loss 0.701. The precision metric was added a little later, but the train 0.400 vs. val 0.332 shows that, of all the alarms the model raises, about 1 in 3 is real. The depth logistic regression model was the best among the base, latlon, faultline, and depth + faultline models. It had the best F1 score and was tied with the base model for the best PR-AUC metric. However, it's a different story for the individual regions. California is cautious and mostly correct, with a precision of 0.637 and a recall of 0.230; Japan floods with alarms, precision 0.306, recall 1.000; and Greece misses most events, precision 0.333, recall 0.093.

**Pooled ablation + reference baselines (validation):**

| Model / variant | PR-AUC | F1 | log-loss | CA | Japan | Greece |
|---|---|---|---|---|---|---|
| dummy (no-skill) | 0.281 | 0.000 | — | 0.231 | 0.306 | 0.307 |
| heuristic (`large_30d≥1`) | 0.342 | 0.418 | — | 0.361 | 0.340 | 0.328 |
| **base** | **0.360** | 0.378 | 0.702 | 0.421 | 0.331 | 0.296 |
| **depth** | **0.360** | 0.385 | 0.701 | 0.421 | 0.328 | 0.302 |
| latlon | 0.357 | 0.382 | 0.701 | 0.424 | 0.320 | 0.288 |
| both | 0.347 | 0.372 | 0.696 | 0.380 | 0.328 | 0.344 |
| faultline | 0.344 | 0.373 | 0.699 | 0.385 | 0.319 | 0.332 |

**Precision & recall — the "don't cry wolf" view (validation):**

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

![Validation PR-AUC by model family](images/Screenshot%202026-07-16%20191613.png)
*Figure 3 — Validation PR-AUC by model family: dummy (floor) < heuristic < logistic regression; Random Forest lower.*

![Validation precision by model family](images/Screenshot%202026-07-16%20192206.png)
*Figure 4 — Validation precision by model family: the dummy is 0; every learned model sits near ~0.30.*

![The 12 core baseline runs compared](images/Screenshot%202026-07-17%20215509.png)
*Figure 5 — The 12 core `earthquake-baseline` runs (5 logistic-regression + 5 Random Forest + 2 reference baselines), with run names visible.*

### 3.2 Per-region models

Logistic regression was also trained separately per region. The pooled model **beat** the per-region
models on Japan and Greece — each per-region model trains on only ~1/3 the data (~2,300 rows) and is
starved, while pooling borrows strength across regions.

| Region | Per-region best | Pooled best (same region) | Winner |
|---|---|---|---|
| California | 0.419 (depth) | ~0.42 (base/depth) | tie |
| Japan | 0.296 (base) | **0.331** (base) | pooled +0.035 |
| Greece | 0.313 (depth) | **0.344** (both) | pooled +0.031 |

### 3.3 Random Forest

A Random Forest model was chosen to see what a non-linear regression model would look like. All random forest models were untuned with n_estimators=300, class_weight="balanced", and min_samples_leaf=20 settings. The random forest base model didn't have any added features, just region and the temporal features. The base model had a PR-AUC train of 0.756 vs. PR-AUC val of 0.336, precision train of 0.540 vs. precision val of 0.337, and F1 score train of 0.648 vs. val of 0.394. This model is overfitted, as shown by the large gap between the train and val PR-AUC metrics, which also means it does not beat the depth logistic model. The base Random Forest model clearly performed better for Greece (0.399 PR-AUC) than any logistic regression variant (best was 0.344). I am motivated to apply an XGBoost-tuned model to this dataset to see whether I can improve the per-region metrics while reducing overfitting.

| RF variant | train PR-AUC | val PR-AUC | val F1 | overfit gap | CA | Japan | Greece |
|---|---|---|---|---|---|---|---|
| base | 0.756 | 0.336 | 0.394 | 0.419 | 0.366 | 0.305 | **0.399** |
| depth | 0.785 | 0.333 | 0.388 | 0.452 | 0.368 | 0.291 | 0.379 |
| faultline | 0.793 | 0.342 | 0.397 | 0.451 | 0.383 | 0.318 | 0.391 |
| both | 0.815 | 0.335 | 0.383 | 0.481 | 0.369 | 0.310 | 0.361 |
| latlon | 0.813 | 0.329 | 0.388 | 0.484 | 0.378 | 0.292 | 0.351 |

### 3.4 Feature importance (chosen `depth` model)

**Logistic regression — standardized coefficients** (sign = direction, |value| = influence):
`region_japan` +0.53 · `region_california` −0.39 · `avg_mag_30d` +0.28 · `large_30d` +0.25 ·
`quakes_7d` +0.23 · `region_greece` −0.21 · then `avg_depth_30d` **+0.06** and all others ≈ 0.
→ the model leans on **region (base rate) + recent significant-activity magnitude/count**; the added
`avg_depth_30d` is **barely used** (confirms the ablation), and explains why the `large_30d` heuristic
is competitive.

**Random Forest — Gini importance** ranks `avg_depth_30d` / magnitudes on top and `region` near the
bottom — but that reflects a known Gini bias toward continuous features (and RF overfit), so the
**ablation (actual score change) is the more trustworthy "does it help" test.**

### 3.5 "Remove the temporal features" — location-only test

A logistic regression model using only the region and geology features was created to assess whether location alone has predictive power. Three variants were tested: a general location-only model, a lat/lon-only model, and a fault-distance-only model. All three models got almost the same metrics, only being off by 0.001–0.015 of a metric. The geology-only model's PR-AUC of 0.292 was higher than the floor PR-AUC (0.281) but lower than my logistic regression model with temporal features (0.360). This shows that when removing all temporal features from the model, performance collapses to nearly the floor model, showing that recent earthquake activity accounts for nearly all the model's predictive power. Surprisingly, the fault-distance-only model scored 0.350 on Greece, beating the full model's 0.296. This suggests that distance-to-fault carries a real, region-specific signal — particularly useful in Greece — even though location alone isn't broadly predictive. Combined with the Random Forest results and the earlier feature ablation findings, this is a third independent signal that region-specific structure is real, which is why XGBoost, a model that can learn these interactions natively, is the planned Week 4 step. So, before turning to the XGBoost model, I will try combining different geological and temporal features to find the best model to tune.

| Model | PR-AUC | CA | Japan | Greece |
|---|---|---|---|---|
| dummy floor | 0.281 | 0.231 | 0.306 | 0.307 |
| FULL base (with temporal) | 0.360 | 0.421 | 0.331 | 0.296 |
| location_only (region + geology) | 0.292 | 0.179 | 0.279 | 0.324 |
| latlon_only | 0.304 | 0.276 | 0.306 | 0.289 |
| faultline_only | 0.294 | 0.192 | 0.286 | **0.350** |

![The three location-only runs](images/Screenshot%202026-07-17%20215322.png)
*Figure 6 — The three location-only runs (location / lat-lon / fault-distance).*

### 3.6 Region×feature interactions

Per-region fault-distance (and activity) slopes were added to the pooled model to get region-specific
behavior without training separate models. It did **not** beat `base`: Greece barely moved
(0.296 → 0.309), California was hurt (0.421 → ~0.37), and overall dropped — a hand-built *linear*
interaction is too blunt (the effect stays diluted amid the dominant temporal features). This motivates
**XGBoost** (which learns interactions natively) in Week 4.

| Model | PR-AUC | Precision | Recall | CA | Japan | Greece |
|---|---|---|---|---|---|---|
| base + region×fault-dist | 0.345 | 0.321 | 0.432 | 0.366 | 0.331 | 0.309 |
| base + region×(fault-dist + activity) | 0.355 | 0.313 | 0.398 | 0.362 | 0.349 | 0.299 |

![Location-only and interaction runs compared](images/Screenshot%202026-07-17%20215556.png)
*Figure 7 — The location-only and region×interaction runs compared.*

---

## 4. Model Selection & Justification

The selected base model is the pooled logistic regression with the depth feature. This model has the highest PR-AUC among the logistic regression and random forest models evaluated, and the best F1 among the logistic regression variants (a couple of the untuned random forest variants edged it on F1, but with much worse PR-AUC and heavy overfitting). Logistic regression was chosen for my base model due to its interpretable results, whereas Random Forest is harder to interpret. Logistic regression models also train and predict quickly, allowing me to experiment more in less time. The Random Forest model also showed overfitting, while the depth logistic regression model had little to no overfitting. However, this model barely beats the naive heuristic model and loses to it in Japan and Greece. The untuned Random Forest showed much higher training metrics but severe overfitting, and did not outperform logistic regression on overall validation PR-AUC; however, it did outperform logistic regression specifically on Greece. This supports the idea that, with tuning and clear feature selection in the XGBoost model, we would be able to improve individual regions and overall metrics, beating the heuristic and the depth logistic regression model. Specifically, XGBoost needs to beat the naive heuristic in Japan and Greece — the two regions where the current logistic regression model falls short.

---

## 5. Companion deliverables (separate files, root `doc/`)
- **`implementation-plan.md`** — Week-3 update: on track; pooled logreg (`depth`) selected;
  per-region rejected (data starvation); heuristic is the Week-4 bar.
- **`claude.md`** — update only if AI-usage instructions changed (likely minor/no change).
- **`ai-usage-log.md`** — Week-3 entry added.

*MLflow deliverable:* run-comparison screenshots for both experiments are embedded above (Figures 1–7); source images are in `doc/Week 3/images/`.

