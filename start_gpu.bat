@echo off
title Local AI Agent - GPU Mode
echo ============================================================
echo   Starting Local AI Telegram Agent (GPU Mode)
echo ============================================================
echo.

echo [1/4] Activating virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate

echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo [3/4] Checking NVIDIA GPU acceleration...
python -c "from tools.gpu_config import print_gpu_status; print_gpu_status()" 2>nul
if %errorlevel% neq 0 (
    echo [!] Warning: GPU acceleration may not be installed.
    echo     Press any key to run 'setup_gpu.bat' now, or close this window.
    pause
    call setup_gpu.bat
)

echo.
echo [4/4] Launching Agent...
:: Force GPU mode usage
set USE_GPU=true
python main.py

echo.
echo Agent stopped. Press any key to exit.
pause >nul
