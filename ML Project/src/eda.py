"""
Exploratory Data Analysis (EDA) Script for Spotter Freight Rate Prediction.
Analyzes feature distributions, missing values, rate per mile dynamics, and generates plots.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

def run_eda(data_dir: str = ".", output_dir: str = "eda_plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = Path(data_dir) / "train-test.csv"
    val_path = Path(data_dir) / "validation.csv"
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_val['date'] = pd.to_datetime(df_val['date'])
    
    print("=== DATASET OVERVIEW ===")
    print(f"Train samples: {len(df_train):,}, Validation samples: {len(df_val):,}")
    print(f"Train Date Range: {df_train['date'].min().strftime('%Y-%m-%d')} to {df_train['date'].max().strftime('%Y-%m-%d')}")
    print(f"Val Date Range:   {df_val['date'].min().strftime('%Y-%m-%d')} to {df_val['date'].max().strftime('%Y-%m-%d')}")
    
    # Rate per mile
    df_train['rpm'] = df_train['posted_rate'] / df_train['distance']
    
    # 1. Target Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df_train['posted_rate'], kde=True, ax=axes[0], color="#064A56", bins=50)
    axes[0].set_title("Distribution of Posted Rate ($)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Posted Rate ($)")
    axes[0].set_ylabel("Count")
    
    sns.histplot(df_train['rpm'], kde=True, ax=axes[1], color="#2E8B57", bins=50)
    axes[1].set_title("Distribution of Rate Per Mile ($/mi)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Rate Per Mile ($/mi)")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "rate_distributions.png"), dpi=200)
    plt.close()
    
    # 2. Posted Rate vs Distance
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_train.sample(5000, random_state=42), x="distance", y="posted_rate", hue="equipment", alpha=0.6, palette="deep", ax=ax)
    ax.set_title("Posted Rate vs Distance by Equipment Type", fontsize=13, fontweight='bold')
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Posted Rate ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "rate_vs_distance.png"), dpi=200)
    plt.close()
    
    # 3. Monthly Trend of Average Rate per Mile
    df_train['month'] = df_train['date'].dt.month
    monthly_rpm = df_train.groupby(['month', 'equipment'])['rpm'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=monthly_rpm, x="month", y="rpm", hue="equipment", marker="o", linewidth=2.5, ax=ax)
    ax.set_title("Monthly Trend: Rate Per Mile by Equipment (Jan - Oct 2025)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month (2025)")
    ax.set_ylabel("Average Rate Per Mile ($/mi)")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "monthly_rpm_trend.png"), dpi=200)
    plt.close()

    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ['posted_rate', 'rpm', 'distance', 'weight', 'market_index', 'quote_signal']
    corr = df_train[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()

    print(f"EDA Completed successfully! Plots saved in '{output_dir}/'.")

if __name__ == "__main__":
    run_eda()
