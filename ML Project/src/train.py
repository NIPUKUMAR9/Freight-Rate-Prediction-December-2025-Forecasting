"""
Model Training and Evaluation Pipeline for Spotter Freight Rate Prediction.
Evaluates Ridge, Random Forest, LightGBM, XGBoost, CatBoost, and Ensemble models using out-of-time temporal CV.
Saves the top-performing model artifact for inference.
"""

import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from features import FreightFeaturePipeline, get_feature_columns


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE) in %."""
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Model") -> dict:
    """Calculates regression performance metrics for USD Posted Rate predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {"Model": label, "MAE ($)": mae, "RMSE ($)": rmse, "MAPE (%)": mape, "R2": r2}
    print(f"[{label:20s}] MAE: ${mae:7.2f} | RMSE: ${rmse:7.2f} | MAPE: {mape:5.2f}% | R2: {r2:.4f}")
    return metrics


def run_training(data_path: str = "train-test.csv", model_dir: str = "models") -> dict:
    os.makedirs(model_dir, exist_ok=True)
    
    print("=== LOADING & PREPROCESSING DATA ===")
    df_train_raw = pd.read_csv(data_path)
    df_train_raw['date'] = pd.to_datetime(df_train_raw['date'])
    df_train_raw = df_train_raw.sort_values('date').reset_index(drop=True)
    
    # 1. Out-of-time Temporal Validation Split (Train: Jan-Aug 2025, Val: Sep-Oct 2025)
    split_date = pd.to_datetime("2025-09-01")
    df_tr = df_train_raw[df_train_raw['date'] < split_date].copy()
    df_val = df_train_raw[df_train_raw['date'] >= split_date].copy()
    
    print(f"Full Train-Test size: {len(df_train_raw):,}")
    print(f"Temporal Split -> Train (Jan-Aug): {len(df_tr):,}, Holdout (Sep-Oct): {len(df_val):,}")
    
    # 2. Fit Pipeline on Training Portion
    pipeline = FreightFeaturePipeline()
    pipeline.fit(df_tr)
    
    X_tr = pipeline.transform(df_tr, is_training=True)
    X_val = pipeline.transform(df_val, is_training=False)
    
    feature_cols = get_feature_columns()
    print(f"Number of engineered features: {len(feature_cols)}")
    
    # Target variables: Rate Per Mile (RPM)
    y_tr_rpm = df_tr['posted_rate'].values / df_tr['distance'].values
    y_val_rpm = df_val['posted_rate'].values / df_val['distance'].values
    y_val_actual = df_val['posted_rate'].values
    dist_val = df_val['distance'].values

    # 3. Model Training & Comparisons
    print("\n=== MODEL COMPARISON (OUT-OF-TIME TEMPORAL HOLDOUT) ===")
    models = {}
    predictions = {}
    results = []
    
    # Model 1: Ridge Baseline
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr[feature_cols], y_tr_rpm)
    pred_rpm_ridge = ridge.predict(X_val[feature_cols])
    pred_rate_ridge = np.maximum(pred_rpm_ridge * dist_val, 1.0)
    results.append(evaluate_predictions(y_val_actual, pred_rate_ridge, "Ridge Baseline"))
    models['ridge'] = ridge
    predictions['ridge'] = pred_rate_ridge

    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr[feature_cols], y_tr_rpm)
    pred_rpm_rf = rf.predict(X_val[feature_cols])
    pred_rate_rf = np.maximum(pred_rpm_rf * dist_val, 1.0)
    results.append(evaluate_predictions(y_val_actual, pred_rate_rf, "Random Forest"))
    models['rf'] = rf
    predictions['rf'] = pred_rate_rf

    # Model 3: LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_tr[feature_cols], y_tr_rpm)
    pred_rpm_lgb = lgb_model.predict(X_val[feature_cols])
    pred_rate_lgb = np.maximum(pred_rpm_lgb * dist_val, 1.0)
    results.append(evaluate_predictions(y_val_actual, pred_rate_lgb, "LightGBM"))
    models['lgb'] = lgb_model
    predictions['lgb'] = pred_rate_lgb

    # Model 4: XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_tr[feature_cols], y_tr_rpm)
    pred_rpm_xgb = xgb_model.predict(X_val[feature_cols])
    pred_rate_xgb = np.maximum(pred_rpm_xgb * dist_val, 1.0)
    results.append(evaluate_predictions(y_val_actual, pred_rate_xgb, "XGBoost"))
    models['xgb'] = xgb_model
    predictions['xgb'] = pred_rate_xgb

    # Model 5: CatBoost Regressor
    cat_model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(X_tr[feature_cols], y_tr_rpm)
    pred_rpm_cat = cat_model.predict(X_val[feature_cols])
    pred_rate_cat = np.maximum(pred_rpm_cat * dist_val, 1.0)
    results.append(evaluate_predictions(y_val_actual, pred_rate_cat, "CatBoost"))
    models['cat'] = cat_model
    predictions['cat'] = pred_rate_cat

    # Model 6: Weighted Ensemble (LightGBM + XGBoost + CatBoost)
    pred_rate_ens = 0.40 * pred_rate_lgb + 0.30 * pred_rate_xgb + 0.30 * pred_rate_cat
    results.append(evaluate_predictions(y_val_actual, pred_rate_ens, "Ensemble (LGB+XGB+CAT)"))
    predictions['ensemble'] = pred_rate_ens

    # 4. Final Retraining on 100% of Labeled Data (Jan - Oct 2025)
    print("\n=== RETRAINING FINAL ENSEMBLE ON FULL 48,000 TRAIN DATASET ===")
    full_pipeline = FreightFeaturePipeline()
    full_pipeline.fit(df_train_raw)
    X_full = full_pipeline.transform(df_train_raw, is_training=True)
    y_full_rpm = df_train_raw['posted_rate'].values / df_train_raw['distance'].values

    final_lgb = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    final_lgb.fit(X_full[feature_cols], y_full_rpm)

    final_xgb = xgb.XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    final_xgb.fit(X_full[feature_cols], y_full_rpm)

    final_cat = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
    final_cat.fit(X_full[feature_cols], y_full_rpm)

    artifact = {
        "pipeline": full_pipeline,
        "feature_cols": feature_cols,
        "models": {
            "lgb": final_lgb,
            "xgb": final_xgb,
            "cat": final_cat
        },
        "weights": {"lgb": 0.40, "xgb": 0.30, "cat": 0.30}
    }
    
    artifact_path = os.path.join(model_dir, "freight_model_pipeline.pkl")
    joblib.dump(artifact, artifact_path)
    print(f"Saved complete trained model pipeline artifact to: {artifact_path}")

    return {"results": pd.DataFrame(results), "artifact_path": artifact_path}


if __name__ == "__main__":
    run_training()
