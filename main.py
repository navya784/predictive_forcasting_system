"""Command-line pipeline for the complete forecasting project.

Run from the project root:

    python main.py

This script loads the dataset, preprocesses it, generates EDA charts, trains
all forecasting models, saves model artifacts, creates a future forecast, and
exports stakeholder-ready reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.preprocessing import (
    DataValidationError,
    available_targets,
    detect_outliers,
    load_and_preprocess,
    validate_dataset,
)
from app.utils import (
    CLEANED_DATASET_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_TARGET,
    DISCHARGE_TARGET,
    HORIZON_OPTIONS,
    REPORTS_DIR,
    ensure_project_folders,
    get_logger,
    save_dataframe,
    save_json,
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(description="Predictive Forecasting of Care Load & Placement Demand")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the input CSV dataset. Default: data/dataset.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Cleaned target column name, such as children_in_hhs_care.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=HORIZON_OPTIONS,
        default=30,
        help="Forecast horizon. Choose 7, 14, or 30 days.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip static EDA chart generation.",
    )
    return parser.parse_args()


def choose_target(clean_df, requested_target: str | None) -> str:
    """Select a valid target column."""

    targets = available_targets(clean_df)
    if requested_target:
        if requested_target not in targets:
            raise DataValidationError(
                f"Target '{requested_target}' is not available. Available targets: {', '.join(targets)}"
            )
        return requested_target
    return DEFAULT_TARGET if DEFAULT_TARGET in targets else targets[0]


def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the full production-style pipeline."""

    logger = get_logger()
    ensure_project_folders()

    logger.info("Loading dataset: %s", args.data)
    clean_df, preprocessing_report = load_and_preprocess(args.data)

    target_column = choose_target(clean_df, args.target)
    validate_dataset(clean_df, target_column)

    save_dataframe(clean_df, CLEANED_DATASET_PATH)
    save_json(preprocessing_report, REPORTS_DIR / "preprocessing_report.json")

    outlier_report = detect_outliers(clean_df)
    save_dataframe(outlier_report, REPORTS_DIR / "outlier_report.csv")

    if not args.skip_eda:
        from app.visualizations import generate_eda_charts

        logger.info("Generating EDA charts...")
        chart_paths = generate_eda_charts(clean_df, target_column)
        logger.info("Saved %s EDA charts.", len(chart_paths))

    from app.forecast import detect_surge_risk, forecast_future, simple_discharge_forecast
    from app.model_training import train_all_models

    logger.info("Training baseline, statistical, and machine learning models...")
    training_results = train_all_models(clean_df, target_column)
    best_artifact = training_results["best_artifact"]

    forecast_df = forecast_future(best_artifact, clean_df, horizon=args.horizon)
    forecast_path = REPORTS_DIR / f"forecast_{target_column}_{args.horizon}_days.csv"
    save_dataframe(forecast_df, forecast_path)

    surge_report = detect_surge_risk(forecast_df, clean_df, target_column)
    save_json(surge_report, REPORTS_DIR / "surge_warning_report.json")

    if DISCHARGE_TARGET in clean_df.columns:
        discharge_forecast = simple_discharge_forecast(clean_df, DISCHARGE_TARGET, args.horizon)
        save_dataframe(discharge_forecast, REPORTS_DIR / f"discharge_forecast_{args.horizon}_days.csv")

    print("\nPROJECT RUN COMPLETED SUCCESSFULLY")
    print("----------------------------------")
    print(f"Clean rows              : {len(clean_df):,}")
    print(f"Selected target         : {target_column}")
    print(f"Best model              : {training_results['best_model_name']}")
    print(f"Forecast horizon        : {args.horizon} days")
    print(f"Forecast CSV            : {forecast_path}")
    print(f"Risk level              : {surge_report['risk_level']}")
    print(f"Capacity breach risk    : {surge_report['capacity_breach_risk_percent']}%")
    print("\nOpen the dashboard with:")
    print("streamlit run app/main.py")
    return 0


def main() -> int:
    """Main entry point with friendly error messages."""

    args = parse_arguments()
    try:
        return run_pipeline(args)
    except FileNotFoundError:
        print("\nERROR: Dataset file not found.", file=sys.stderr)
        print("Place your CSV at data/dataset.csv or use --data path/to/file.csv", file=sys.stderr)
        return 1
    except DataValidationError as exc:
        print(f"\nDATA VALIDATION ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print("\nDEPENDENCY ERROR: A required package is missing.", file=sys.stderr)
        print("Run: pip install -r requirements.txt", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}", file=sys.stderr)
        print("Check your dataset format and installed dependencies.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
