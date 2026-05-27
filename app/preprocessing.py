"""Data loading, cleaning, validation, feature preparation, and outlier checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.utils import (
    CANONICAL_TARGETS,
    DATE_COLUMN,
    DEFAULT_TARGET,
    DISCHARGE_TARGET,
    MINIMUM_ROWS,
    clean_column_name,
    make_unique,
    numeric_columns,
)


class DataValidationError(ValueError):
    """Friendly error raised when the dataset is not forecast-ready."""


def load_dataset(path_or_buffer: str | Path | Any) -> pd.DataFrame:
    """Load a CSV dataset with common encodings.

    The function supports both local file paths and Streamlit-uploaded files.
    """

    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise DataValidationError("Unable to read CSV. Please upload a UTF-8 CSV file.") from last_error


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Clean and canonicalize dataset column names."""

    cleaned_names = make_unique([clean_column_name(column) for column in df.columns])
    rename_map = dict(zip(df.columns, cleaned_names, strict=False))

    output = df.copy()
    output.columns = cleaned_names

    canonical_map: dict[str, str] = {}
    for column in output.columns:
        canonical_name = infer_canonical_name(column)
        canonical_map[column] = canonical_name

    output = output.rename(columns=canonical_map)
    output.columns = make_unique(list(output.columns))
    final_map = {original: canonical_map.get(cleaned, cleaned) for original, cleaned in rename_map.items()}
    return output, final_map


def infer_canonical_name(column: str) -> str:
    """Map slightly different public-dataset headers to standard names."""

    tokens = set(column.split("_"))

    if "date" in tokens or column == "report_date":
        return DATE_COLUMN

    if "hhs" in tokens and "care" in tokens and any(token.startswith("discharg") for token in tokens):
        return DISCHARGE_TARGET

    if "hhs" in tokens and "care" in tokens and "discharged" not in tokens:
        return DEFAULT_TARGET

    if "transfer" in column and "cbp" in tokens:
        return "children_transferred_out_of_cbp_custody"

    if "apprehend" in column and "cbp" in tokens:
        return "children_apprehended_and_placed_in_cbp_custody"

    if "cbp" in tokens and "custody" in tokens and "transfer" not in column:
        return "children_in_cbp_custody"

    return column


def detect_date_column(df: pd.DataFrame) -> str:
    """Find the date column even when the header is not exactly 'Date'."""

    if DATE_COLUMN in df.columns:
        return DATE_COLUMN

    date_like = [column for column in df.columns if "date" in column]
    if date_like:
        return date_like[0]

    best_column = ""
    best_rate = 0.0
    for column in df.columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        rate = float(parsed.notna().mean())
        if rate > best_rate:
            best_column = column
            best_rate = rate

    if best_rate < 0.50:
        raise DataValidationError("No usable Date column was found.")
    return best_column


def convert_to_numeric(series: pd.Series) -> pd.Series:
    """Safely convert strings like '2,484' into numeric values."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "-": np.nan, ".": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def add_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add net pressure, imbalance, and capacity pressure columns."""

    output = df.copy()
    transfers = "children_transferred_out_of_cbp_custody"
    discharges = DISCHARGE_TARGET
    hhs_care = DEFAULT_TARGET

    if transfers in output.columns and discharges in output.columns:
        output["net_pressure"] = output[transfers] - output[discharges]
        output["intake_discharge_imbalance"] = output[transfers] - output[discharges]

    if hhs_care in output.columns:
        rolling_capacity = output[hhs_care].rolling(window=90, min_periods=7).quantile(0.95)
        rolling_capacity = rolling_capacity.bfill().ffill()
        output["capacity_pressure_index"] = np.where(
            rolling_capacity > 0,
            output[hhs_care] / rolling_capacity,
            0.0,
        )

    return output


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers with the IQR rule for every numeric column."""

    rows: list[dict[str, float | int | str]] = []
    for column in numeric_columns(df, exclude=[DATE_COLUMN]):
        q1 = float(df[column].quantile(0.25))
        q3 = float(df[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[column] < lower) | (df[column] > upper)).sum())
        rows.append(
            {
                "column": column,
                "lower_bound": round(lower, 3),
                "upper_bound": round(upper, 3),
                "outlier_count": count,
            }
        )
    return pd.DataFrame(rows)


def available_targets(df: pd.DataFrame) -> list[str]:
    """Return numeric forecast targets available in the cleaned dataset."""

    numeric = numeric_columns(df, exclude=[DATE_COLUMN])
    preferred = [column for column in CANONICAL_TARGETS if column in numeric]
    return preferred or numeric


def preprocess_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the complete preprocessing workflow."""

    if raw_df.empty:
        raise DataValidationError("The dataset is empty.")

    metadata: dict[str, Any] = {
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "warnings": [],
    }

    df, rename_map = standardize_columns(raw_df)
    metadata["column_mapping"] = rename_map

    df = df.replace(r"^\s*$", np.nan, regex=True)
    before_blank_drop = len(df)
    df = df.dropna(how="all").copy()
    metadata["blank_rows_removed"] = int(before_blank_drop - len(df))

    date_column = detect_date_column(df)
    if date_column != DATE_COLUMN:
        df = df.rename(columns={date_column: DATE_COLUMN})

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    invalid_dates = int(df[DATE_COLUMN].isna().sum())
    df = df.dropna(subset=[DATE_COLUMN])
    metadata["invalid_date_rows_removed"] = invalid_dates

    for column in df.columns:
        if column != DATE_COLUMN:
            df[column] = convert_to_numeric(df[column])

    numeric = numeric_columns(df, exclude=[DATE_COLUMN])
    if not numeric:
        raise DataValidationError("No numeric columns were found for forecasting.")

    before_numeric_drop = len(df)
    df = df.dropna(how="all", subset=numeric)
    metadata["rows_without_numeric_values_removed"] = int(before_numeric_drop - len(df))

    duplicate_rows = int(df.duplicated(subset=[DATE_COLUMN]).sum())
    df = df.groupby(DATE_COLUMN, as_index=False)[numeric].mean()
    metadata["duplicate_date_rows_aggregated"] = duplicate_rows

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    if len(df) < MINIMUM_ROWS:
        raise DataValidationError(f"At least {MINIMUM_ROWS} valid rows are required. Found {len(df)}.")

    full_range = pd.date_range(df[DATE_COLUMN].min(), df[DATE_COLUMN].max(), freq="D")
    missing_dates = len(full_range) - df[DATE_COLUMN].nunique()
    metadata["missing_dates_interpolated"] = int(max(0, missing_dates))

    df = df.set_index(DATE_COLUMN).reindex(full_range)
    df.index.name = DATE_COLUMN
    df[numeric] = df[numeric].interpolate(method="time", limit_direction="both").ffill().bfill()
    df[numeric] = df[numeric].clip(lower=0)
    df = df.reset_index()
    df = add_operational_features(df)

    validate_dataset(df)
    metadata["clean_rows"] = int(df.shape[0])
    metadata["clean_columns"] = int(df.shape[1])
    metadata["date_start"] = str(df[DATE_COLUMN].min().date())
    metadata["date_end"] = str(df[DATE_COLUMN].max().date())
    metadata["available_targets"] = available_targets(df)
    return df, metadata


def load_and_preprocess(path_or_buffer: str | Path | Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a CSV file and preprocess it."""

    raw_df = load_dataset(path_or_buffer)
    return preprocess_data(raw_df)


def validate_dataset(df: pd.DataFrame, target_column: str | None = None) -> None:
    """Validate the cleaned dataset before training models."""

    if DATE_COLUMN not in df.columns:
        raise DataValidationError("The cleaned dataset is missing a date column.")

    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        raise DataValidationError("The date column could not be converted to datetime.")

    if not df[DATE_COLUMN].is_monotonic_increasing:
        raise DataValidationError("Dataset must be sorted by date.")

    targets = available_targets(df)
    if not targets:
        raise DataValidationError("No valid target columns are available.")

    selected = target_column or (DEFAULT_TARGET if DEFAULT_TARGET in targets else targets[0])
    if selected not in targets:
        raise DataValidationError(
            f"Target '{selected}' is not available. Available targets: {', '.join(targets)}"
        )

    if df[selected].isna().any():
        raise DataValidationError(f"Target '{selected}' contains missing values after cleaning.")
