@echo off
rem Start the Liquidation Guardian web app.
rem Usage: double-click or run from cmd (Windows)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo python not found - install Python 3.10+ first (check "Add to PATH").
  pause
  exit /b 1
)

python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
