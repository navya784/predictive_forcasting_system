"""Model training and evaluation for all required forecasting models."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.utils import (
    DATE_COLUMN,
    LAG_DAYS,
    MODEL_FILENAMES,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ROLLING_WINDOWS,
    TEST_SIZE_RATIO,
    ForecastArtifact,
    numeric_columns,
    safe_mape,
    save_dataframe,
)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features from the date column."""

    output = df.copy()
    output["day"] = output[DATE_COLUMN].dt.day
    output["week"] = output[DATE_COLUMN].dt.isocalendar().week.astype(int)
    output["month"] = output[DATE_COLUMN].dt.month
    output["day_of_week"] = output[DATE_COLUMN].dt.dayofweek
    output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype(int)
    return output


def create_ml_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    """Create lag, rolling, pressure, and calendar features for ML models."""

    output = df.sort_values(DATE_COLUMN).copy()
    output = add_time_features(output)

    for lag in LAG_DAYS:
        output[f"{target_column}_lag_{lag}"] = output[target_column].shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = output[target_column].shift(1)
        output[f"{target_column}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
        output[f"{target_column}_rolling_std_{window}"] = shifted.rolling(window, min_periods=2).std().fillna(0)

    feature_columns = [
        column
        for column in numeric_columns(output, exclude=[DATE_COLUMN, target_column])
        if column != target_column
    ]

    modeling_df = output[[DATE_COLUMN, target_column] + feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
    return modeling_df, feature_columns


def chronological_split(df: pd.DataFrame, test_size: float = TEST_SIZE_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically with no random shuffling."""

    split_index = int(len(df) * (1 - test_size))
    split_index = min(max(split_index, 30), len(df) - 7)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def calculate_metrics(actual_values: pd.Series | np.ndarray, predicted_values: pd.Series | np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, MAPE, R2, and accuracy percentage."""

    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    predicted = np.maximum(0, predicted)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = safe_mape(actual, predicted)
    r2 = float(r2_score(actual, predicted)) if len(np.unique(actual)) > 1 else 0.0
    accuracy = float(max(0.0, 100.0 - mape))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2_score": r2,
        "accuracy_percent": accuracy,
    }


def residual_std(actual_values: pd.Series | np.ndarray, predicted_values: pd.Series | np.ndarray) -> float:
    """Estimate residual standard deviation for confidence intervals."""

    residuals = np.asarray(actual_values, dtype=float) - np.asarray(predicted_values, dtype=float)
    if len(residuals) <= 1:
        return 1.0
    return float(np.nanstd(residuals, ddof=1) or 1.0)


def naive_forecast(train_series: pd.Series, test_series: pd.Series) -> np.ndarray:
    """Walk-forward naive forecast using yesterday's value."""

    history = list(train_series.astype(float))
    predictions: list[float] = []
    for actual in test_series.astype(float):
        predictions.append(float(history[-1]))
        history.append(float(actual))
    return np.asarray(predictions)


def moving_average_forecast(train_series: pd.Series, test_series: pd.Series, window: int = 7) -> np.ndarray:
    """Walk-forward moving-average forecast."""

    history = list(train_series.astype(float))
    predictions: list[float] = []
    for actual in test_series.astype(float):
        predictions.append(float(np.mean(history[-window:])))
        history.append(float(actual))
    return np.asarray(predictions)


def fit_arima(train_series: pd.Series) -> Any:
    """Fit ARIMA(1,1,1)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(train_series, order=(1, 1, 1), enforce_stationarity=False).fit()


def fit_sarima(train_series: pd.Series) -> Any:
    """Fit weekly SARIMA model."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)


def fit_exponential_smoothing(train_series: pd.Series) -> Any:
    """Fit additive Holt-Winters exponential smoothing."""

    seasonal = "add" if len(train_series) >= 28 else None
    seasonal_periods = 7 if seasonal else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)


def walk_forward_validation(feature_df: pd.DataFrame, feature_columns: list[str], target_column: str, model_name: str) -> float:
    """Evaluate ML models with TimeSeriesSplit walk-forward validation."""

    if len(feature_df) < 80:
        return float("nan")

    splitter = TimeSeriesSplit(n_splits=5)
    rmse_values: list[float] = []

    for train_index, test_index in splitter.split(feature_df):
        train_fold = feature_df.iloc[train_index]
        test_fold = feature_df.iloc[test_index]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_fold[feature_columns])
        x_test = scaler.transform(test_fold[feature_columns])

        if model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
                random_state=RANDOM_STATE,
            )

        model.fit(x_train, train_fold[target_column])
        predictions = model.predict(x_test)
        rmse_values.append(calculate_metrics(test_fold[target_column], predictions)["rmse"])

    return float(np.nanmean(rmse_values))


def save_model(artifact: ForecastArtifact, path: Path) -> Path:
    """Save a model artifact with joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def train_ml_model(
    model_name: str,
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[ForecastArtifact, dict[str, float | str]]:
    """Train Random Forest or Gradient Boosting."""

    train_df, test_df = chronological_split(feature_df)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    x_test = scaler.transform(test_df[feature_columns])

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.035,
            max_depth=3,
            random_state=RANDOM_STATE,
        )

    model.fit(x_train, train_df[target_column])
    predictions = np.maximum(0, model.predict(x_test))
    metrics = calculate_metrics(test_df[target_column], predictions)
    metrics["walk_forward_rmse"] = walk_forward_validation(feature_df, feature_columns, target_column, model_name)

    artifact = ForecastArtifact(
        model_name=model_name,
        model_type="machine_learning",
        target_column=target_column,
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        metrics=metrics,
        residual_std=residual_std(test_df[target_column], predictions),
    )
    return artifact, {"model": model_name, **metrics}


def train_all_models(
    df: pd.DataFrame,
    target_column: str,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train every required model, save artifacts, and return a leaderboard."""

    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    working_df = df.sort_values(DATE_COLUMN).reset_index(drop=True).copy()
    train_df, test_df = chronological_split(working_df)
    train_series = train_df.set_index(DATE_COLUMN)[target_column].asfreq("D")
    test_series = test_df[target_column].astype(float)

    rows: list[dict[str, float | str]] = []
    artifacts: dict[str, ForecastArtifact] = {}
    saved_paths: dict[str, str] = {}

    baseline_specs = [
        ("Naive Forecast", "naive", naive_forecast(train_df[target_column], test_series), {}),
        ("Moving Average", "moving_average", moving_average_forecast(train_df[target_column], test_series), {"window": 7}),
    ]
    for model_name, model_type, predictions, extra in baseline_specs:
        metrics = calculate_metrics(test_series, predictions)
        artifact = ForecastArtifact(
            model_name=model_name,
            model_type=model_type,
            target_column=target_column,
            metrics=metrics,
            residual_std=residual_std(test_series, predictions),
            extra=extra,
        )
        artifacts[model_name] = artifact
        rows.append({"model": model_name, **metrics})
        saved_paths[model_name] = str(save_model(artifact, models_dir / MODEL_FILENAMES[model_name]))

    statistical_specs = [
        ("ARIMA", fit_arima, "statsmodels_state"),
        ("SARIMA", fit_sarima, "statsmodels_state"),
        ("Exponential Smoothing", fit_exponential_smoothing, "statsmodels_smoothing"),
    ]
    for model_name, fit_function, model_type in statistical_specs:
        try:
            model = fit_function(train_series)
            predictions = np.asarray(model.forecast(steps=len(test_df)), dtype=float)
            predictions = np.maximum(0, predictions)
            metrics = calculate_metrics(test_series, predictions)
            artifact = ForecastArtifact(
                model_name=model_name,
                model_type=model_type,
                target_column=target_column,
                model=model,
                metrics=metrics,
                residual_std=residual_std(test_series, predictions),
            )
            artifacts[model_name] = artifact
            rows.append({"model": model_name, **metrics})
            saved_paths[model_name] = str(save_model(artifact, models_dir / MODEL_FILENAMES[model_name]))
        except Exception as exc:
            rows.append({"model": model_name, "error": str(exc)})

    feature_df, feature_columns = create_ml_features(working_df, target_column)
    for model_name in ["Random Forest", "Gradient Boosting"]:
        try:
            artifact, row = train_ml_model(model_name, feature_df, feature_columns, target_column)
            artifacts[model_name] = artifact
            rows.append(row)
            saved_paths[model_name] = str(save_model(artifact, models_dir / MODEL_FILENAMES[model_name]))
            if artifact.scaler is not None:
                joblib.dump(artifact.scaler, models_dir / MODEL_FILENAMES["Scaler"])
                saved_paths["Scaler"] = str(models_dir / MODEL_FILENAMES["Scaler"])
        except Exception as exc:
            rows.append({"model": model_name, "error": str(exc)})

    leaderboard = pd.DataFrame(rows)
    for column in ["mae", "rmse", "mape", "r2_score", "accuracy_percent", "walk_forward_rmse"]:
        if column in leaderboard.columns:
            leaderboard[column] = pd.to_numeric(leaderboard[column], errors="coerce")

    leaderboard = leaderboard.sort_values(["rmse", "mae"], na_position="last").reset_index(drop=True)
    save_dataframe(leaderboard, REPORTS_DIR / "model_leaderboard.csv")

    valid_models = leaderboard.dropna(subset=["rmse"])
    if valid_models.empty:
        raise RuntimeError("No model trained successfully. Check the dataset and installed packages.")

    best_model_name = str(valid_models.iloc[0]["model"])
    best_artifact = artifacts[best_model_name]
    saved_paths["Best Model"] = str(save_model(best_artifact, models_dir / MODEL_FILENAMES["Best Model"]))

    return {
        "leaderboard": leaderboard,
        "artifacts": artifacts,
        "best_model_name": best_model_name,
        "best_artifact": best_artifact,
        "saved_paths": saved_paths,
    }
