"""
Inference Pipeline for Spotter Freight Rate Model.
Generates predictions for validation.csv and december-chart-inputs.csv.
Outputs validation_predictions.csv and december_predictions.csv.
"""

import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd


def generate_predictions(
    data_dir: str = ".",
    model_path: str = "models/freight_model_pipeline.pkl",
    val_out: str = "validation_predictions.csv",
    dec_out: str = "december_predictions.csv"
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please run src/train.py first.")

    print("=== LOADING TRAINED MODEL ARTIFACT ===")
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]
    models = artifact["models"]
    weights = artifact["weights"]

    # 1. Predict on validation.csv
    print("\n--- Generating Predictions for validation.csv ---")
    val_file = Path(data_dir) / "validation.csv"
    df_val = pd.read_csv(val_file)
    print(f"Validation dataset loaded: {len(df_val):,} rows")

    X_val = pipeline.transform(df_val, is_training=False)

    # Ensemble RPM prediction
    pred_rpm_lgb = models["lgb"].predict(X_val[feature_cols])
    pred_rpm_xgb = models["xgb"].predict(X_val[feature_cols])
    pred_rpm_cat = models["cat"].predict(X_val[feature_cols])

    pred_rpm = (
        weights["lgb"] * pred_rpm_lgb +
        weights["xgb"] * pred_rpm_xgb +
        weights["cat"] * pred_rpm_cat
    )

    pred_rate = np.maximum(pred_rpm * df_val["distance"].values, 1.0)
    
    # Save validation_predictions.csv (exact columns: load_id, predicted_rate)
    df_val_preds = pd.DataFrame({
        "load_id": df_val["load_id"],
        "predicted_rate": np.round(pred_rate, 2)
    })
    df_val_preds.to_csv(val_out, index=False)
    print(f"Saved {len(df_val_preds):,} validation predictions to: {val_out}")

    # 2. Predict on december-chart-inputs.csv
    print("\n--- Generating Predictions for december-chart-inputs.csv ---")
    dec_file = Path(data_dir) / "december-chart-inputs.csv"
    df_dec = pd.read_csv(dec_file)
    print(f"December inputs loaded: {len(df_dec):,} rows")

    X_dec = pipeline.transform(df_dec, is_training=False)

    pred_rpm_dec_lgb = models["lgb"].predict(X_dec[feature_cols])
    pred_rpm_dec_xgb = models["xgb"].predict(X_dec[feature_cols])
    pred_rpm_dec_cat = models["cat"].predict(X_dec[feature_cols])

    pred_rpm_dec = (
        weights["lgb"] * pred_rpm_dec_lgb +
        weights["xgb"] * pred_rpm_dec_xgb +
        weights["cat"] * pred_rpm_dec_cat
    )

    pred_rate_dec = np.maximum(pred_rpm_dec * df_dec["distance"].values, 1.0)

    # Keep original 7 columns and format
    df_dec_preds = df_dec.copy()
    df_dec_preds["predicted_rate"] = np.round(pred_rate_dec, 2)
    df_dec_preds.to_csv(dec_out, index=False)
    print(f"Saved {len(df_dec_preds):,} December predictions to: {dec_out}")
    print("Inference step complete!")


if __name__ == "__main__":
    generate_predictions()
