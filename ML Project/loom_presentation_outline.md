# Loom Video Presentation Outline (2-3 Minutes)
**Candidate Presentation Guide: Freight Rate Prediction & December 2025 Forecasting**

---

## Time Allocation Summary (Target: 2 min 30 sec)
- **0:00 - 0:25**: Introduction & Objective Overview
- **0:25 - 0:55**: Data Exploration & Data Hygiene Handling
- **0:55 - 1:35**: Validation Strategy & Rate-Per-Mile Formulation
- **1:35 - 2:10**: Model Selection, Performance & December Chart Analysis
- **2:10 - 2:30**: Codebase Walkthrough & Closing

---

## Detailed Script & Talking Points

### 1. Introduction (0:00 - 0:25)
> "Hi everyone! In this presentation, I'm walking through my Machine Learning solution for Spotter's Freight Rate Prediction assessment. The primary objective is to build an out-of-time predictive pipeline for US freight rates (`posted_rate`), predict rates for 12,000 validation loads in November–December 2025, and forecast December 2025 daily rates for a fixed Lexington to Fort Wayne dry van lane."

---

### 2. Data Exploration & Quality Issues Addressed (0:25 - 0:55)
> "During exploratory data analysis of the 48,000 labeled development loads, I identified three critical data quality insights:
> 1. **Missing Values**: ~300 rows lacked `weight` and ~374 lacked `market_index`. I addressed this by imputing weights using equipment-level medians (e.g., 32k lbs for Dry Van, 35k lbs for Reefer) and applying temporal forward-fill interpolation for `market_index` to prevent data leakage.
> 2. **Target Variance & Scale**: Total rates ranged wildly from $57 to over $25,000. However, **Rate Per Mile (RPM)** had a stable average of $2.21/mi across equipment types.
> 3. **Spatial Validation**: pickup and delivery lat/lon coordinates correlated 0.9995 with reported distances, revealing an average circuity factor of 1.19x relative to straight-line Haversine distance."

---

### 3. Training & Validation Strategy (0:55 - 1:35)
> "Because freight markets experience macro-economic shifts and seasonality over time, standard random K-Fold cross-validation would cause severe temporal data leakage.
> 
> To accurately simulate the real-world evaluation on November–December 2025 loads, I implemented an **Out-of-Time Temporal Split**:
> - **Train Set (Jan 1 – Aug 31, 2025)**: 38,477 loads (80.2% of labeled data).
> - **Validation Holdout (Sep 1 – Oct 31, 2025)**: 9,523 out-of-time loads (19.8%).
> 
> Additionally, instead of predicting raw total rates directly, I modeled **Rate Per Mile (RPM = posted_rate / distance)** as the target variable. This prevents extreme long-distance loads from skewing gradient updates and reduced our Mean Absolute Error (MAE) significantly."

---

### 4. Reasoning Behind Chosen Model & December Chart (1:35 - 2:10)
> "I benchmarked Ridge Regression, Random Forests, LightGBM, XGBoost, and CatBoost on our out-of-time holdout. 
> 
> Gradient Boosted Decision Trees outperformed traditional baselines dramatically:
> - LightGBM achieved an **MAE of $151.42** and a **MAPE of 6.85%**.
> - CatBoost achieved the highest single-model R² of **0.8153**.
> 
> To maximize stability, I constructed a **Weighted Ensemble** (40% LightGBM + 30% XGBoost + 30% CatBoost), yielding an out-of-time **MAE of $151.59, MAPE of 6.98%, and R² of 0.8133**.
> 
> Running `score.py` on our December predictions generated a clean forecast for the Lexington to Fort Wayne lane. The predicted daily rates hover stably between **$808 and $820** ($2.25/mi to $2.28/mi), capturing midweek dispatch peaks and holiday weekend troughs while avoiding any outlier spikes."

---

### 5. Code Walkthrough & Repository Overview (2:10 - 2:30)
> "Briefly highlighting the code structure:
> - `src/features.py`: Contains our modular `FreightFeaturePipeline` which handles spatial features (haversine, bearing, midpoint), cyclical date transformations (sine/cosine day of week/year), and Bayesian smoothed target encoding for shipping lanes.
> - `src/train.py`: Handles model training, temporal validation metric logging, and artifact persistence.
> - `src/predict.py`: Generates `validation_predictions.csv` and `december_predictions.csv` cleanly.
> 
> Thank you for reviewing my assessment!"
