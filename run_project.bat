@echo off
setlocal

echo =====================================================
echo Predictive Forecasting of Care Load and Placement Demand
echo =====================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.11 or newer from https://www.python.org/downloads/
    echo IMPORTANT: Check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Requirement installation failed. Check your internet connection and Python version.
    pause
    exit /b 1
)

echo.
echo Running the training pipeline...
python main.py
if errorlevel 1 (
    echo Pipeline failed. Check the error message above.
    pause
    exit /b 1
)

echo.
echo Starting Streamlit dashboard...
python -m streamlit run app\main.py

pause

