# Predictive Forecasting of Care Load & Placement Demand

## Executive Overview

This project is a complete healthcare analytics and time-series forecasting system for estimating care load, discharge demand, intake-discharge imbalance, and capacity stress indicators for HHS-style daily operational datasets.

It is designed for:

- VS Code execution on Windows
- GitHub upload
- Streamlit dashboard demos
- Resume and portfolio use
- College final-year project submission
- Research or stakeholder presentation

Source context: [HHS](https://www.hhs.gov/)

## Key Features

- Dynamic CSV loading with safe column-name detection.
- Missing-value handling, duplicate removal, and date conversion.
- Daily continuity checks with date interpolation.
- Outlier detection using the IQR method.
- Premium dark UI using the approved government-grade palette: `#2563EB`, `#0F172A`, `#F59E0B`, `#EF4444`, `#F8FAFC`, and `#CBD5E1`.
- Glassmorphism KPI cards, risk alert panels, responsive layout, hover effects, Plotly dark charts, and Streamlit 1.56-compatible APIs.
- Feature engineering for lags, rolling averages, rolling standard deviation, calendar fields, weekend flag, and net pressure.
- Baseline models: Naive Forecast and Moving Average.
- Statistical models: ARIMA, SARIMA, and Exponential Smoothing.
- Machine learning models: Random Forest Regressor and Gradient Boosting Regressor.
- Model evaluation with MAE, RMSE, MAPE, R2 Score, and accuracy percentage.
- Automatic best-model selection by lowest RMSE.
- 7-day, 14-day, and 30-day forecasting.
- Confidence intervals for future predictions.
- Surge detection with capacity-risk banners.
- Professional Streamlit dashboard with upload, model selection, forecast horizon, KPI cards, charts, and CSV download.
- Deployment files for Streamlit Cloud, Render, and Hugging Face Spaces.

## Tech Stack

- Python 3.11+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Statsmodels
- Streamlit
- Joblib

## Folder Structure

```text
Predictive_Forecasting_Project/
|
|-- app/
|   |-- __init__.py
|   |-- main.py                 # Streamlit dashboard
|   |-- forecast.py             # Future forecasting and surge detection
|   |-- preprocessing.py        # Data cleaning and validation
|   |-- utils.py                # Shared constants and helpers
|   |-- visualizations.py       # EDA and Plotly charts
|   |-- model_training.py       # Model training and evaluation
|
|-- data/
|   |-- dataset.csv             # Place your dataset here
|   |-- cleaned_dataset.csv     # Generated after running the pipeline
|
|-- models/
|   |-- saved model files       # Generated after training
|
|-- notebooks/
|   |-- EDA.ipynb
|
|-- outputs/
|   |-- charts/                 # Generated chart images
|   |-- reports/                # Generated CSV and JSON reports
|
|-- .streamlit/
|   |-- config.toml
|
|-- app.py                      # Deployment wrapper
|-- main.py                     # Command-line pipeline
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- Procfile
|-- runtime.txt
|-- packages.txt
|-- run_project.bat
```

## Dataset

Expected columns are similar to:

- Date
- Children apprehended and placed in CBP custody
- Children in CBP custody
- Children transferred out of CBP custody
- Children in HHS Care
- Children discharged from HHS Care

The code dynamically cleans and maps column names. Exact capitalization and punctuation are not required.

## Windows VS Code Execution Guide

### 1. Install Python

1. Open [python.org/downloads](https://www.python.org/downloads/).
2. Download Python 3.11 or newer.
3. During installation, select **Add python.exe to PATH**.
4. Verify in PowerShell:

```powershell
python --version
```

### 2. Open the Project in VS Code

1. Open VS Code.
2. Click **File > Open Folder**.
3. Select:

```text
Predictive_Forecasting_Project
```

4. Open the terminal:

```text
Ctrl + `
```

### 3. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 4. Activate the Virtual Environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

### 5. Install Requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Upload or Place the Dataset

Place your CSV file at:

```text
data/dataset.csv
```

This project already includes the mentioned HHS CSV as `data/dataset.csv`. You can replace it with a new dataset later.

### 7. Run the Full ML Pipeline

```powershell
python main.py
```

Run with a specific horizon:

```powershell
python main.py --horizon 7
python main.py --horizon 14
python main.py --horizon 30
```

Run with a specific target:

```powershell
python main.py --target children_discharged_from_hhs_care --horizon 30
```

### 8. Run the Streamlit Dashboard

```powershell
streamlit run app/main.py
```

Alternative:

```powershell
python -m streamlit run app/main.py
```

### 9. Run Everything with the Batch File

```powershell
.\run_project.bat
```

The batch file creates a virtual environment, installs packages, runs the pipeline, and starts Streamlit.

## How to Test Predictions

1. Run:

```powershell
python main.py --horizon 30
```

2. Confirm these files were created:

```text
outputs/reports/model_leaderboard.csv
outputs/reports/forecast_children_in_hhs_care_30_days.csv
outputs/reports/surge_warning_report.json
models/best_model.pkl
```

3. Start the dashboard:

```powershell
streamlit run app/main.py
```

4. In the sidebar:

- Upload `data/dataset.csv`, or use the default dataset.
- Select forecast target.
- Select model or choose Auto Best.
- Select 7, 14, or 30 days.
- Click **Run Forecast**.

5. Review:

- KPI cards
- Forecast chart
- Confidence interval band
- Capacity risk indicator
- Model comparison table
- Downloadable forecast CSV

## Model Evaluation

The project calculates:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- R2 Score
- Forecast Accuracy %

The best model is automatically selected using the lowest RMSE.

## Surge Detection Logic

The system calculates a recent 30-day rolling average for the selected target. If forecast values or upper confidence interval values exceed that rolling average by 10% or more, the system flags capacity risk.

Risk levels:

- `LOW CAPACITY RISK`
- `MODERATE CAPACITY RISK`
- `HIGH CAPACITY RISK`

## GitHub Upload Guide

Run these commands in Windows PowerShell from the project root:

```powershell
git init
git add .
git commit -m "Initial commit: predictive forecasting project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

Steps:

1. Create a new repository on GitHub.
2. Do not initialize it with a README because this project already has one.
3. Copy the repository URL.
4. Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` in the command above.
5. Run the commands.

## Deployment Guide

### Option 1: Streamlit Cloud

Required files:

- `requirements.txt`
- `app/main.py`
- `.streamlit/config.toml`

The dashboard uses Streamlit 1.56+ features such as `st.iframe` and `width="stretch"`, so keep `streamlit>=1.56.0` in `requirements.txt`.

Steps:

1. Push the project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path:

```text
app/main.py
```

6. Click **Deploy**.

Common fixes:

- If packages fail, confirm `requirements.txt` is in the repository root.
- If the dataset is missing, upload from the dashboard sidebar or commit `data/dataset.csv`.
- If the app sleeps, wake it from Streamlit Cloud dashboard.

### Option 2: Render

Required files:

- `requirements.txt`
- `Procfile`
- `runtime.txt`

Steps:

1. Push the project to GitHub.
2. Go to [Render](https://render.com/).
3. Click **New Web Service**.
4. Connect your repository.
5. Select Python environment.
6. Build command:

```bash
pip install -r requirements.txt
```

7. Start command:

```bash
streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
```

8. Deploy.

Common fixes:

- If `$PORT` fails locally, ignore it locally. Render provides it in deployment.
- If the app cannot find data, upload the CSV in the dashboard or commit `data/dataset.csv`.

### Option 3: Hugging Face Spaces

Required files:

- `requirements.txt`
- `app.py`
- `app/`
- `data/`

Steps:

1. Create a new Space at [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK.
3. Upload all project files.
4. Use the root `app.py` wrapper as the entry point.
5. Confirm `requirements.txt` is at the root.

Common fixes:

- If imports fail, confirm the `app/` folder contains `__init__.py`.
- If packages fail, check Python package versions in `requirements.txt`.
- If data is not included, use the dashboard uploader.

## Troubleshooting

### Python is not recognized

Reinstall Python and select **Add python.exe to PATH**.

### pip fails

Use:

```powershell
python -m pip install -r requirements.txt
```

### Streamlit command not found

Use:

```powershell
python -m streamlit run app/main.py
```

### Dataset validation error

Check that your CSV has:

- A date-like column
- At least one numeric target column
- Enough rows for forecasting

### Model training is slow

The SARIMA and Random Forest models can take longer on large datasets. Start with the default HHS CSV to verify your environment.

### Dashboard says model could not be trained

Choose **Auto Best**. Some statistical models may fail on unusual datasets, but the system will continue with the models that train successfully.

## Screenshots Section

After running the dashboard, capture these screenshots for reports or presentations:

- Dashboard overview
- KPI cards
- Forecast chart with confidence intervals
- Model leaderboard
- Capacity risk indicator
- Discharge prediction panel

Suggested folder:

```text
screenshots/
```

## Future Improvements

- Add XGBoost after confirming deployment environment support.
- Add automated hyperparameter tuning.
- Add holiday, policy-event, and operational-capacity features.
- Add drift monitoring.
- Add Docker deployment.
- Add authenticated dashboard access.
- Add automated report generation.

## License

This project is released under the MIT License for academic, portfolio, and demonstration use.
