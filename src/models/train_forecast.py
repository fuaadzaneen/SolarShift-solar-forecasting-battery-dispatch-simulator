"""
Stage 2 Model Training & Walk-Forward Cross-Validation Pipeline.
Trains HistGradientBoostingRegressor for 24-hour horizon day-ahead PV generation.
Compares ML predictions against 24-hr persistence and Clear-Sky baselines.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from src.ingest.prepare_data import PROCESSED_DATA_PATH
from src.models.forecaster import build_forecasting_dataset
from src.metrics.forecast_metrics import evaluate_forecast_models

FORECAST_RESULTS_PATH = os.path.join("data", "processed", "kochi_forecast_results.csv")
FORECAST_METRICS_PATH = os.path.join("data", "processed", "forecast_metrics_summary.csv")


def run_stage2_forecasting():
    """
    Executes Stage 2 Forecasting:
    1. Prepares walk-forward feature matrix.
    2. Performs strict time-series train/test split:
       - Train: 2021-01-03 to 2022-12-31 (~2 full years)
       - Test: 2023-01-01 to 2023-12-31 (1 full unseen year)
    3. Trains HistGradientBoostingRegressor.
    4. Evaluates ML model vs Persistence vs Clear-Sky baselines overall and across seasons.
    5. Saves outputs and comparison charts.
    """
    print("=" * 65)
    print("SOLARSHIFT: EXECUTING STAGE 2 DAY-AHEAD FORECASTING ENGINE")
    print("=" * 65)

    df_raw = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])
    df_feat = build_forecasting_dataset(df_raw)

    # Walk-forward temporal split (Train on 2021-2022, Test on 2023)
    split_date = pd.Timestamp("2023-01-01 00:00:00")
    train_df = df_feat[df_feat["timestamp"] < split_date].copy()
    test_df = df_feat[df_feat["timestamp"] >= split_date].copy()

    print(f"[SPLIT] Training Set: {len(train_df)} samples ({train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()})")
    print(f"[SPLIT] Test Set:     {len(test_df)} samples ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})")

    features = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "hour",
        "pv_clearsky_baseline",
        "pv_lag_24", "pv_lag_48",
        "ghi_lag_24", "ghi_lag_48",
        "past_24h_mean_pv", "past_24h_max_pv", "past_24h_cloud_cover",
        "forecast_cloud_cover", "forecast_temp_air", "forecast_relative_humidity"
    ]

    target = "pv_kw"

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    print(f"[TRAIN] Training HistGradientBoostingRegressor on {len(features)} features...")
    # Clean, explainable hyperparameter choices
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on test set
    test_preds = model.predict(X_test)
    # Clip physically: PV cannot be negative and cannot exceed 5 kW inverter rating
    test_preds = np.clip(test_preds, 0.0, 5.0)

    # Force night hours (where clearsky is 0) to exactly zero to avoid phantom night predictions
    night_mask = test_df["pv_clearsky_baseline"].values <= 0.001
    test_preds[night_mask] = 0.0

    test_df["pv_ml_forecast"] = np.round(test_preds, 3)

    # Evaluation across models and Kerala seasons
    print("[EVAL] Evaluating ML model vs Persistence vs Clear-Sky Baselines...")
    metrics_summary = evaluate_forecast_models(
        test_df,
        actual_col="pv_kw",
        pred_cols=["pv_persistence_baseline", "pv_clearsky_baseline", "pv_ml_forecast"]
    )

    print("\n" + "=" * 80)
    print("FORECASTING PERFORMANCE COMPARISON (TEST YEAR: 2023)")
    print("=" * 80)
    print(metrics_summary.to_string(index=False))

    # Save metrics and predictions
    metrics_summary.to_csv(FORECAST_METRICS_PATH, index=False)
    test_df.to_csv(FORECAST_RESULTS_PATH, index=False)
    print(f"\n[OUTPUT] Predictions saved to: {FORECAST_RESULTS_PATH}")
    print(f"[OUTPUT] Metrics saved to:      {FORECAST_METRICS_PATH}")

    # Generate a visual comparison plot for representative days
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "forecast_seasonal_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    sample_dates = {
        ("Dry Winter (Jan 15)", "2023-01-15"): axes[0, 0],
        ("Pre-Monsoon Summer (Apr 15)", "2023-04-15"): axes[0, 1],
        ("Southwest Monsoon (Jul 15)", "2023-07-15"): axes[1, 0],
        ("Post-Monsoon (Oct 15)", "2023-10-15"): axes[1, 1],
    }

    for (title, date_str), ax in sample_dates.items():
        day_sub = test_df[test_df["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
        if len(day_sub) > 0:
            hrs = day_sub["hour"].values
            ax.plot(hrs, day_sub["pv_kw"], "k-o", label="Actual PV (Truth)", linewidth=2.0)
            ax.plot(hrs, day_sub["pv_ml_forecast"], "g--s", label="ML Forecast (HistGBR)", linewidth=1.8)
            ax.plot(hrs, day_sub["pv_persistence_baseline"], "b:", label="24h Persistence", linewidth=1.4)
            ax.plot(hrs, day_sub["pv_clearsky_baseline"], "r-.", alpha=0.5, label="Clear-Sky Ceiling", linewidth=1.2)
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel("Hour of Day (IST)", fontsize=9)
            ax.set_ylabel("Power Output (kW)", fontsize=9)
            ax.set_ylim(-0.2, 4.0)
            ax.grid(True, linestyle="--", alpha=0.5)
            if ax == axes[0, 0]:
                ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("SolarShift: 24-Hour Day-Ahead Solar Forecasting Across Kerala Seasons (Kochi, 2023)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT] Generated visual comparison chart at: {plot_path}")

    return test_df, metrics_summary


if __name__ == "__main__":
    run_stage2_forecasting()
