"""
Forecasting Evaluation Metrics Module.
Calculates MAE, RMSE, and Normalized RMSE (nRMSE) across the test window
and stratified by Kerala's 4 meteorological seasons.
"""

import numpy as np
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity_kw: float = 5.0) -> dict:
    """
    Calculate MAE, RMSE, and capacity-normalized nRMSE (%).
    """
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    nrmse_cap_pct = float((rmse / capacity_kw) * 100.0)
    
    # Daylight-only metrics (where solar generation is possible: y_true > 0 or y_pred > 0)
    daylight_mask = (y_true > 0.05) | (y_pred > 0.05)
    if np.sum(daylight_mask) > 0:
        daylight_mae = float(np.mean(np.abs(error[daylight_mask])))
        daylight_rmse = float(np.sqrt(np.mean(error[daylight_mask] ** 2)))
    else:
        daylight_mae = 0.0
        daylight_rmse = 0.0

    return {
        "MAE_kW": round(mae, 4),
        "RMSE_kW": round(rmse, 4),
        "nRMSE_capacity_pct": round(nrmse_cap_pct, 2),
        "Daylight_MAE_kW": round(daylight_mae, 4),
        "Daylight_RMSE_kW": round(daylight_rmse, 4)
    }


def evaluate_forecast_models(
    df_eval: pd.DataFrame,
    actual_col: str = "pv_kw",
    pred_cols: list = None
) -> pd.DataFrame:
    """
    Evaluates multiple predictions overall and by season.
    
    Parameters:
    -----------
    df_eval : pd.DataFrame with ground truth, season, and predictions
    actual_col : str
    pred_cols : list of str
    
    Returns:
    --------
    pd.DataFrame containing formatted performance comparison.
    """
    if pred_cols is None:
        pred_cols = ["pv_persistence_baseline", "pv_clearsky_baseline", "pv_ml_forecast"]

    results = []

    # Overall evaluation
    y_true = df_eval[actual_col].values
    for col in pred_cols:
        if col in df_eval.columns:
            m = calculate_metrics(y_true, df_eval[col].values)
            m["Model"] = col
            m["Season"] = "All Seasons (Overall)"
            m["Samples"] = len(df_eval)
            results.append(m)

    # Seasonal breakdown
    seasons = ["Dry Winter", "Pre-Monsoon", "Southwest Monsoon", "Post-Monsoon"]
    for season in seasons:
        df_sub = df_eval[df_eval["season"] == season]
        if len(df_sub) == 0:
            continue
        y_sub_true = df_sub[actual_col].values
        for col in pred_cols:
            if col in df_sub.columns:
                m = calculate_metrics(y_sub_true, df_sub[col].values)
                m["Model"] = col
                m["Season"] = season
                m["Samples"] = len(df_sub)
                results.append(m)

    df_res = pd.DataFrame(results)
    col_order = ["Model", "Season", "Samples", "MAE_kW", "RMSE_kW", "nRMSE_capacity_pct", "Daylight_MAE_kW"]
    return df_res[col_order]


if __name__ == "__main__":
    y_t = np.array([0.0, 1.0, 2.5, 3.0, 0.0])
    y_p = np.array([0.0, 1.2, 2.2, 2.8, 0.0])
    print(calculate_metrics(y_t, y_p))
