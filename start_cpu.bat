@echo off
title Local AI Agent - CPU Mode
echo ============================================================
echo   Starting Local AI Telegram Agent (CPU Mode)
echo ============================================================
echo.

echo [1/3] Activating virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate

echo.
echo [2/3] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo [3/3] Launching Agent...
:: Force the CPU fallback by hijacking the env var locally for this session
set USE_GPU=false
python main.py

echo.
echo Agent stopped. Press any key to exit.
pause >nul
