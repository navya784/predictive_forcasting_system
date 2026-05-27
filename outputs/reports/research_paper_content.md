# Research Paper Content

## Title

Predictive Forecasting of Care Load and Placement Demand Using Time-Series and Machine Learning Models

## Abstract

This project presents a data-driven forecasting framework for estimating future care load, discharge demand, intake-discharge imbalance, and capacity stress indicators in a healthcare-adjacent child placement context. The system uses automated preprocessing, time-series feature engineering, statistical forecasting, and supervised machine learning models to compare predictive performance across multiple approaches. The project includes a Streamlit dashboard for interactive forecasting, model comparison, confidence interval visualization, and surge-risk detection. The final system supports operational planning by converting historical daily counts into short-term demand forecasts and actionable early-warning indicators.

## Introduction

Healthcare and human-services agencies often rely on daily operational data to plan staffing, placement capacity, discharge coordination, and emergency response. When care-load demand changes quickly, decision-makers need short-term forecasts and clear warning signals. This project focuses on forecasting the number of children in HHS care, discharge demand, and intake pressure using a clean and reproducible machine learning pipeline.

## Objectives

1. Clean and validate daily operational datasets automatically.
2. Forecast the future number of children in HHS care.
3. Forecast future discharge demand.
4. Estimate intake-discharge imbalance using net pressure.
5. Compare baseline, statistical, and machine learning forecasting models.
6. Generate confidence intervals and surge warnings.
7. Deploy the solution through an interactive Streamlit dashboard.

## Methodology

The dataset is first standardized by cleaning column names, converting dates, removing blank rows, aggregating duplicate dates, converting numeric strings, sorting chronologically, and interpolating missing daily dates. Feature engineering then creates lag variables, rolling means, rolling standard deviations, calendar variables, weekend indicators, net pressure, intake-discharge imbalance, and capacity pressure index.

The modeling pipeline uses a strict chronological train-test split to prevent data leakage. Baseline models provide simple reference forecasts. Statistical models capture time-series structure through ARIMA, SARIMA, and Exponential Smoothing. Machine learning models use Random Forest and Gradient Boosting regressors trained on engineered time-series features. Models are evaluated using MAE, RMSE, MAPE, R2 Score, and forecast accuracy percentage. The best model is selected automatically by lowest RMSE.

## Results

The system produces a model leaderboard, saved model artifacts, forecast CSV files, EDA charts, and surge-risk reports. The dashboard presents forecast trends, confidence intervals, KPI cards, discharge predictions, net pressure charts, and capacity risk indicators. Actual numerical results depend on the uploaded dataset and selected forecast target.

## Conclusion

The project demonstrates how time-series forecasting and machine learning can support operational planning for care-load and placement-demand management. By combining automated preprocessing, multiple model families, explainable evaluation metrics, and a user-friendly dashboard, the system provides a practical foundation for short-term demand forecasting and early-warning analysis.

## Future Scope

Future improvements may include XGBoost, hyperparameter optimization, cloud model monitoring, policy-event features, staffing capacity inputs, holiday effects, drift detection, and automated PDF report generation.

