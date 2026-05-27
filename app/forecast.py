"""Future forecasting, confidence intervals, and surge detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils import DATE_COLUMN, DEFAULT_TARGET, LAG_DAYS, ROLLING_WINDOWS, ForecastArtifact


def future_dates(history_df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """Create daily future dates after the last observed date."""

    last_date = pd.to_datetime(history_df[DATE_COLUMN].max())
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def confidence_interval(predictions: np.ndarray, residual_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Create simple 95% confidence intervals using residual spread."""

    horizon = len(predictions)
    safe_std = float(residual_std or np.std(predictions) or 1.0)
    widening_factor = np.sqrt(np.arange(1, horizon + 1))
    margin = 1.96 * safe_std * widening_factor
    lower = np.maximum(0, predictions - margin)
    upper = predictions + margin
    return lower, upper


def build_future_feature_row(
    history_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    artifact: ForecastArtifact,
) -> pd.DataFrame:
    """Build one recursive future feature row for ML models."""

    history = history_df.copy()
    feature_row: dict[str, float] = {}

    # Fill all unknown operational predictors with recent historical averages.
    for column in artifact.feature_columns:
        if column in history.columns and pd.api.types.is_numeric_dtype(history[column]):
            feature_row[column] = float(history[column].tail(7).mean())
        else:
            feature_row[column] = 0.0

    target_values = history[artifact.target_column].astype(float).to_numpy()
    for lag in LAG_DAYS:
        feature_row[f"{artifact.target_column}_lag_{lag}"] = (
            float(target_values[-lag]) if len(target_values) >= lag else float(target_values[-1])
        )

    for window in ROLLING_WINDOWS:
        recent = target_values[-window:] if len(target_values) >= window else target_values
        feature_row[f"{artifact.target_column}_rolling_mean_{window}"] = float(np.mean(recent))
        feature_row[f"{artifact.target_column}_rolling_std_{window}"] = float(np.std(recent, ddof=0))

    feature_row["day"] = float(forecast_date.day)
    feature_row["week"] = float(forecast_date.isocalendar().week)
    feature_row["month"] = float(forecast_date.month)
    feature_row["day_of_week"] = float(forecast_date.dayofweek)
    feature_row["is_weekend"] = float(forecast_date.dayofweek in [5, 6])

    return pd.DataFrame([{column: feature_row.get(column, 0.0) for column in artifact.feature_columns}])


def forecast_with_ml(artifact: ForecastArtifact, history_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Generate recursive forecasts for ML models."""

    working_history = history_df.sort_values(DATE_COLUMN).copy()
    predictions: list[float] = []

    for date in future_dates(working_history, horizon):
        feature_row = build_future_feature_row(working_history, pd.Timestamp(date), artifact)
        x_future = artifact.scaler.transform(feature_row) if artifact.scaler is not None else feature_row
        prediction = float(artifact.model.predict(x_future)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)

        next_row = working_history.iloc[-1].copy()
        next_row[DATE_COLUMN] = date
        next_row[artifact.target_column] = prediction
        working_history = pd.concat([working_history, pd.DataFrame([next_row])], ignore_index=True)

    return np.asarray(predictions)


def forecast_future(artifact: ForecastArtifact, history_df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Forecast future values with confidence intervals."""

    if horizon not in [7, 14, 30]:
        raise ValueError("Forecast horizon must be 7, 14, or 30 days.")

    dates = future_dates(history_df, horizon)

    if artifact.model_type == "naive":
        last_value = float(history_df[artifact.target_column].iloc[-1])
        predictions = np.repeat(last_value, horizon)
        lower, upper = confidence_interval(predictions, artifact.residual_std)

    elif artifact.model_type == "moving_average":
        window = int(artifact.extra.get("window", 7))
        value = float(history_df[artifact.target_column].tail(window).mean())
        predictions = np.repeat(value, horizon)
        lower, upper = confidence_interval(predictions, artifact.residual_std)

    elif artifact.model_type == "machine_learning":
        predictions = forecast_with_ml(artifact, history_df, horizon)
        lower, upper = confidence_interval(predictions, artifact.residual_std)

    elif artifact.model_type == "statsmodels_state":
        forecast_result = artifact.model.get_forecast(steps=horizon)
        predictions = np.maximum(0, np.asarray(forecast_result.predicted_mean, dtype=float))
        intervals = forecast_result.conf_int(alpha=0.05)
        lower = np.maximum(0, np.asarray(intervals.iloc[:, 0], dtype=float))
        upper = np.maximum(0, np.asarray(intervals.iloc[:, 1], dtype=float))

    elif artifact.model_type == "statsmodels_smoothing":
        predictions = np.maximum(0, np.asarray(artifact.model.forecast(horizon), dtype=float))
        lower, upper = confidence_interval(predictions, artifact.residual_std)

    else:
        raise ValueError(f"Unsupported model type: {artifact.model_type}")

    return pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "forecast": predictions,
            "lower_ci": lower,
            "upper_ci": upper,
            "model": artifact.model_name,
            "target": artifact.target_column,
        }
    )


def simple_discharge_forecast(df: pd.DataFrame, discharge_column: str, horizon: int) -> pd.DataFrame:
    """Create a lightweight discharge-demand forecast for the dashboard."""

    dates = future_dates(df, horizon)
    recent = df[discharge_column].astype(float).tail(14)
    prediction = float(recent.mean())
    residual = float(recent.std(ddof=1) or 1.0)
    predictions = np.repeat(prediction, horizon)
    lower, upper = confidence_interval(predictions, residual)
    return pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "forecast": predictions,
            "lower_ci": lower,
            "upper_ci": upper,
            "model": "14-day Moving Average",
            "target": discharge_column,
        }
    )


def detect_surge_risk(
    forecast_df: pd.DataFrame,
    history_df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    increase_threshold: float = 0.10,
) -> dict[str, float | int | str]:
    """Detect surge and capacity risk from forecasted values.

    Rule:
    If forecast values exceed the recent 30-day rolling average by 10% or more,
    the status becomes HIGH CAPACITY RISK.
    """

    target_history = history_df[target_column].astype(float)
    rolling_average = float(target_history.tail(30).mean())
    capacity_threshold = rolling_average * (1 + increase_threshold)

    forecast_values = forecast_df["forecast"].astype(float)
    upper_values = forecast_df["upper_ci"].astype(float)
    risk_days = int(((forecast_values > capacity_threshold) | (upper_values > capacity_threshold)).sum())
    risk_percent = float(risk_days / len(forecast_df) * 100) if len(forecast_df) else 0.0

    if risk_percent >= 50:
        risk_level = "HIGH CAPACITY RISK"
    elif risk_percent > 0:
        risk_level = "MODERATE CAPACITY RISK"
    else:
        risk_level = "LOW CAPACITY RISK"

    surge_dates = forecast_df.loc[forecast_values > capacity_threshold, DATE_COLUMN]
    if surge_dates.empty:
        lead_time = int(len(forecast_df))
    else:
        lead_time = int((pd.to_datetime(surge_dates.iloc[0]) - pd.to_datetime(history_df[DATE_COLUMN].max())).days)

    stability_index = float(max(0.0, 100.0 - (forecast_values.std(ddof=0) / (forecast_values.mean() or 1.0)) * 100))

    return {
        "risk_level": risk_level,
        "rolling_average_30_day": round(rolling_average, 2),
        "capacity_threshold": round(capacity_threshold, 2),
        "risk_days": risk_days,
        "capacity_breach_risk_percent": round(risk_percent, 2),
        "surge_lead_time_days": lead_time,
        "forecast_stability_index": round(stability_index, 2),
    }
