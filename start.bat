@echo off
echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate

echo Running main.py...
python main.py

echo.
echo Agent stopped. Press any key to exit.
pause >nul