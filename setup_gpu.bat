@echo off
setlocal EnableDelayedExpansion
title GPU Setup - Local AI Telegram Agent
echo.
echo ============================================================
echo   NVIDIA GPU Setup for Local AI Telegram Agent
echo ============================================================
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [!] Virtual environment not found. Creating one...
    python -m venv .venv
)

:: Activate virtual environment
call .venv\Scripts\activate

:: Step 1: Check for NVIDIA GPU
echo [1/4] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] nvidia-smi not found!
    echo    No NVIDIA GPU detected or drivers not installed.
    echo    Please install NVIDIA drivers from: https://www.nvidia.com/drivers
    echo.
    echo    Falling back to CPU mode.
    pause
    exit /b 1
)

echo [OK] NVIDIA GPU detected:
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo.

:: Step 2: Check for CUDA
echo [2/4] Checking for CUDA Toolkit...
where nvcc >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] CUDA Toolkit nvcc not found in PATH.
    echo    The pre-built wheels may still work with your GPU driver.
    echo    For best results, install CUDA Toolkit 12.x from:
    echo    https://developer.nvidia.com/cuda-downloads
    echo.
) else (
    echo [OK] CUDA Toolkit found in PATH.
    echo.
)

:: Step 3: Uninstall CPU-only llama-cpp-python
echo [3/4] Removing CPU-only llama-cpp-python build...
pip uninstall llama-cpp-python -y >nul 2>&1
echo [OK] Cleaned up old installation.
echo.

:: Step 4: Install CUDA-enabled llama-cpp-python
echo [4/4] Installing CUDA-accelerated llama-cpp-python...
echo    Trying pre-built CUDA 12.4/12.2/12.1 wheels...
echo.

:: We use --index-url (instead of extra) to prevent pip from falling back to CPU version on PyPI
pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --force-reinstall --no-cache-dir

if !errorlevel! neq 0 (
    echo.
    echo [!] CUDA 12.1 wheel failed. Trying 12.2 wheel...
    pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 --force-reinstall --no-cache-dir
)

if !errorlevel! neq 0 (
    echo.
    echo [!] Pre-built wheels failed. Building from source with CUDA...
    echo    This requires CMake and a C++ compiler - Visual Studio Build Tools.
    echo.
    set "CMAKE_ARGS=-DGGML_CUDA=ON"
    pip install llama-cpp-python --force-reinstall --no-cache-dir
)

if !errorlevel! neq 0 (
    echo.
    echo ============================================================
    echo [FAILED] Could not install CUDA-enabled llama-cpp-python.
    echo.
    echo Troubleshooting:
    echo   1. Ensure NVIDIA drivers are up to date
    echo   2. Install CUDA Toolkit 12.x
    echo   3. Install Visual Studio Build Tools for source build
    echo   4. Try manually:
    echo      pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] CUDA-accelerated llama-cpp-python installed!
echo ============================================================
echo.

:: Verify GPU access from Python
echo Verifying GPU access...
python -c "from tools.gpu_config import print_gpu_status; print_gpu_status()"

echo.
echo Setup complete! Run 'python main.py' or 'start.bat' to launch the bot.
echo.
pause
endlocal
