@echo off
setlocal
cd /d "%~dp0"

echo =========================================
echo  VR Eyebrow Tracker - Run Script
echo =========================================

echo Checking for virtual environment...
if exist "venv_gpu\Scripts\activate.bat" (
    echo Activating venv_gpu...
) else (
    echo venv_gpu not found. Creating venv_gpu...
    python -m venv "venv_gpu"
    if errorlevel 1 (
        echo ERROR: Failed to create venv_gpu. Make sure Python is installed and on PATH.
        echo.
        pause
        exit /b 1
    )
)
call "venv_gpu\Scripts\activate.bat"
echo.

echo Ensuring required modules (cv2)...
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    python -m pip install --upgrade pip
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements. Check your network and Python environment.
        echo.
        pause
        exit /b 1
    )
)

if not exist "gui.py" (
    echo ERROR: gui.py not found. Cannot launch GUI.
    echo.
    pause
    exit /b 1
)

echo.
echo Launching GUI...
python gui.py

echo.
pause
