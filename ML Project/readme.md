# Spotter Machine Learning Engineer Assessment: Freight Rate Prediction

## Executive Overview
This repository contains a production-ready Machine Learning solution for predicting US freight rates (`posted_rate`) across shipping lanes. Built for Spotter's Machine Learning Engineer assessment, the project processes historical load data (`train-test.csv`), handles out-of-time validation loads (`validation.csv`), models daily rate behavior for fixed lane December 2025 inputs (`december-chart-inputs.csv`), and produces candidate reports and verification artifacts.

### Key Performance Highlights (Out-of-Time Temporal Holdout)
- **Mean Absolute Error (MAE):** **$151.59**
- **Mean Absolute Percentage Error (MAPE):** **6.98%**
- **Root Mean Squared Error (RMSE):** **$659.33**
- **Coefficient of Determination ($R^2$):** **0.8133**

---

## Repository Structure

```
.
├── freight-rate-ml-assessment.pdf # Assessment instructions PDF
├── score.py                        # Official Spotter submission validation script
├── train-test.csv                  # 48,000 labeled development loads (Jan - Oct 2025)
├── validation.csv                  # 12,000 evaluation loads (Nov - Dec 2025)
├── december-chart-inputs.csv       # 31 fixed lane December 2025 input rows
├── validation-predictions-template.csv # Prediction template format
├── validation_predictions.csv     # Final 12,000 predictions for validation.csv
├── december_predictions.csv       # Final 31 predictions for December 2025
├── requirements.txt                # Python environment dependencies
├── generate_report.py              # PDF/DOCX report generator script
├── Spotter_MLE_Assessment_Report.pdf # Final assessment candidate report (PDF)
├── Spotter_MLE_Assessment_Report.docx# Final assessment candidate report (DOCX)
├── loom_presentation_outline.md    # 2-3 minute Loom video presentation script
├── scorer_results/
│   └── candidate_december.png      # Generated December 2025 rate chart
├── eda_plots/                      # Generated EDA visualizations
├── models/
│   └── freight_model_pipeline.pkl  # Saved production model artifact
└── src/
    ├── eda.py                      # Exploratory Data Analysis script
    ├── features.py                 # Feature engineering & preprocessing pipeline
    ├── train.py                    # Model training, CV & ensemble script
    └── predict.py                  # Inference script for validation & December inputs
```

---

## Environment Setup & Requirements

### Installation
Ensure Python 3.9+ is installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run the Complete Pipeline

Run the end-to-end workflow using the following sequential commands:

### 1. Exploratory Data Analysis & Visualizations
Generates dataset statistics and saves distribution plots into `eda_plots/`:
```bash
python src/eda.py
```

### 2. Model Training & Out-of-Time Temporal Validation
Trains Ridge, Random Forest, LightGBM, XGBoost, and CatBoost models on temporal splits (Jan–Aug train, Sep–Oct validation), logs metric benchmarks, and saves the retrained ensemble artifact to `models/freight_model_pipeline.pkl`:
```bash
python src/train.py
```

### 3. Inference & Prediction Generation
Generates `validation_predictions.csv` (12,000 rows) and `december_predictions.csv` (31 rows):
```bash
python src/predict.py
```

### 4. Official Spotter Validation & Chart Generation
Runs Spotter's official scorer script to validate predictions format and render `scorer_results/candidate_december.png`:
```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv --output-dir scorer_results
```

### 5. PDF & DOCX Report Generation
Generates `Spotter_MLE_Assessment_Report.pdf` and `Spotter_MLE_Assessment_Report.docx`:
```bash
python generate_report.py
```

---

## Methodology & Architectural Highlights

### 1. Temporal Out-of-Time Validation Scheme
Freight spot markets fluctuate over time due to seasonal demand and fuel/market index shifts. Standard random K-Fold cross-validation leaks temporal information. We evaluated models using a strict **out-of-time temporal split**:
- **Train Set (Jan 1 – Aug 31, 2025):** 38,477 loads (80.2%)
- **Validation Holdout (Sep 1 – Oct 31, 2025):** 9,523 loads (19.8%)
- **Final Retraining:** 100% of 48,000 development loads for final production inference.

### 2. Rate Per Mile (RPM) Target Formulation
Rate Per Mile (RPM) Target FormulationDirectly modeling the total USD rate introduces extreme target variance across short-haul and long-haul loads (spanning from $57 to $25,533). Predicting absolute rates causes long-haul trips to disproportionately dominate model loss gradients during training.To stabilize variance and ensure balanced learning across all trip distances, the target variable is normalized to Rate Per Mile (RPM).1. Training TransformationDuring model training, the target feature is transformed into per-mile rates:$$\text{RPM} = \frac{\text{posted\_rate}}{\text{distance}}$$2. Inference & Price ReconstructionAt inference time, the model predicts $\widehat{\text{RPM}}$. The total price is reconstructed using the trip distance, with a minimum price floor constraint of $1.00 applied:$$\text{Predicted Rate} = \max(\widehat{\text{RPM}} \times \text{distance},\ 1.0)$$Why


### 3. Feature Engineering Architecture (32 Features)
- **Geographic / Spatial:** Haversine distance, distance ratio (`distance / haversine`), bearing angle (degrees), latitude/longitude deltas, midpoint coordinates.
- **Temporal & Cyclical:** Sine/cosine cyclical encodings for `dayofweek` and `dayofyear`, month, quarter, weekend binary flags.
- **Load Density & Weight:** Weight per mile, weight ratio relative to equipment medians, equipment one-hot encodings.
- **Bayesian Smoothed Encoded Statistics:** Smoothed target encoding for pickup-delivery lanes, pickup cities, and delivery cities.

---

## Model Benchmark Results

Evaluation metrics on the out-of-time Sep–Oct 2025 holdout dataset:

| Model Architecture | MAE ($) | RMSE ($) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression Baseline** | $226.89 | $682.70 | 9.67% | 0.7999 |
| **Random Forest Regressor** | $185.73 | $716.30 | 8.24% | 0.7797 |
| **XGBoost Regressor** | $168.04 | $676.08 | 8.01% | 0.8037 |
| **CatBoost Regressor** | $155.60 | $655.86 | 7.08% | 0.8153 |
| **LightGBM Regressor** | **$151.42** | $661.79 | **6.85%** | 0.8119 |
| **Weighted Ensemble (LGB+XGB+CAT)** | **$151.59** | **$659.33** | **6.98%** | **0.8133** |

---

## Fixed December 2025 Chart Analysis

The generated chart `scorer_results/candidate_december.png` models rates for Lexington to Fort Wayne (360 miles, Dry Van, 32,000 lbs):
- **Rate Stability:** Predicted daily rates stay bounded between **$808.31 and $820.35** ($2.25/mi to $2.28/mi), reflecting real-world Midwest Dry Van pricing.
- **Weekly Dynamics:** Periodic midweek peaks correspond to shipping dispatch volumes, with subtle dips on weekend windows and end-of-year holidays.

---

## Deliverables Summary
1. Complete modular Python codebase in `src/`.
2. `validation_predictions.csv` (12,000 predictions).
3. `december_predictions.csv` (31 predictions).
4. `scorer_results/candidate_december.png` chart from `score.py`.
5. Candidate reports: `Spotter_MLE_Assessment_Report.pdf` & `Spotter_MLE_Assessment_Report.docx`.
6. Loom video presentation script: `loom_presentation_outline.md`.
