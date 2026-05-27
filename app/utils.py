"""Shared utilities, constants, and small helper functions.

This file intentionally keeps beginner-friendly project settings in one place.
If you want to change folder paths, forecast defaults, or target-column names,
start here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DEFAULT_DATASET_PATH = DATA_DIR / "dataset.csv"
CLEANED_DATASET_PATH = DATA_DIR / "cleaned_dataset.csv"


# ---------------------------------------------------------------------------
# Forecasting defaults
# ---------------------------------------------------------------------------

DATE_COLUMN = "date"
DEFAULT_TARGET = "children_in_hhs_care"
DISCHARGE_TARGET = "children_discharged_from_hhs_care"
HORIZON_OPTIONS = [7, 14, 30]
LAG_DAYS = [1, 7, 14]
ROLLING_WINDOWS = [7, 14]
RANDOM_STATE = 42
TEST_SIZE_RATIO = 0.20
MINIMUM_ROWS = 45


CANONICAL_TARGETS = [
    "children_in_hhs_care",
    "children_discharged_from_hhs_care",
    "children_transferred_out_of_cbp_custody",
    "children_in_cbp_custody",
    "children_apprehended_and_placed_in_cbp_custody",
    "net_pressure",
    "intake_discharge_imbalance",
]


DISPLAY_NAMES = {
    "children_apprehended_and_placed_in_cbp_custody": "Children apprehended and placed in CBP custody",
    "children_in_cbp_custody": "Children in CBP custody",
    "children_transferred_out_of_cbp_custody": "Children transferred out of CBP custody",
    "children_in_hhs_care": "Children in HHS Care",
    "children_discharged_from_hhs_care": "Children discharged from HHS Care",
    "net_pressure": "Net pressure",
    "intake_discharge_imbalance": "Intake vs discharge imbalance",
    "capacity_pressure_index": "Capacity pressure index",
}


MODEL_FILENAMES = {
    "Naive Forecast": "naive_forecast_model.pkl",
    "Moving Average": "moving_average_model.pkl",
    "ARIMA": "arima_model.pkl",
    "SARIMA": "sarima_model.pkl",
    "Exponential Smoothing": "exponential_smoothing_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Best Model": "best_model.pkl",
    "Scaler": "scaler.pkl",
}


@dataclass
class ForecastArtifact:
    """Saved model object used for future forecasting.

    A single dataclass keeps statistical models, ML models, and baseline models
    in one consistent format.
    """

    model_name: str
    model_type: str
    target_column: str
    model: Any = None
    scaler: Any = None
    feature_columns: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    residual_std: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


def ensure_project_folders() -> None:
    """Create required folders if they are missing."""

    for folder in [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, OUTPUTS_DIR, CHARTS_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def get_logger() -> logging.Logger:
    """Return a clean terminal logger."""

    logger = logging.getLogger("care_load_forecasting")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def clean_column_name(column_name: Any) -> str:
    """Convert any column name to safe snake_case."""

    cleaned = str(column_name).strip().lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def make_unique(columns: list[str]) -> list[str]:
    """Make duplicate column names unique without losing readability."""

    seen: dict[str, int] = {}
    unique_columns: list[str] = []
    for column in columns:
        if column not in seen:
            seen[column] = 0
            unique_columns.append(column)
        else:
            seen[column] += 1
            unique_columns.append(f"{column}_{seen[column]}")
    return unique_columns


def display_name(column: str) -> str:
    """Return a readable label for charts and dashboard controls."""

    return DISPLAY_NAMES.get(column, column.replace("_", " ").title())


def numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return numeric column names, excluding fields such as the date column."""

    exclude_set = set(exclude or [])
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in exclude_set
    ]


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe as CSV and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save a dictionary as JSON and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
    return output_path


def safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate MAPE while avoiding division by zero."""

    denominator = np.where(np.abs(actual) < 1e-8, 1.0, np.abs(actual))
    return float(np.mean(np.abs((actual - predicted) / denominator)) * 100)

