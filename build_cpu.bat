@echo off
setlocal
cd /d "%~dp0"

echo =========================================
echo  VR Eyebrow Tracker - CPU Build Script
echo =========================================

echo Checking for CPU virtual environment...
if exist "venv_cpu\Scripts\activate.bat" (
    echo Activating venv_cpu...
) else (
    echo venv_cpu not found. Creating venv_cpu...
    python -m venv "venv_cpu"
    if errorlevel 1 (
        echo ERROR: Failed to create venv_cpu. Make sure Python is installed and on PATH.
        echo.
        pause
        exit /b 1
    )
)
call "venv_cpu\Scripts\activate.bat"
echo.

echo Ensuring required modules (cv2, onnxruntime)...
python -c "import cv2; import onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo Installing CPU-only requirements...
    python -m pip install --upgrade pip
    python -m pip install onnxruntime-directml
    rem Install remaining requirements except torch/torchvision/torchaudio (not needed for CPU inference build)
    if exist "requirements_cpu.txt" del /q "requirements_cpu.txt"
    findstr /i /v "^torch" requirements.txt > requirements_cpu.txt
    python -m pip install -r requirements_cpu.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements. Check your network and Python environment.
        echo.
        pause
        exit /b 1
    )
)

:check_dependencies
echo Checking for required modules (cv2, onnxruntime)...
python -c "import cv2; import onnxruntime; print(cv2.__version__)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python modules 'cv2' and/or 'onnxruntime' were not found in the active environment.
    echo Install the missing dependencies in this environment, then rerun this script.
    echo.
    pause
    exit /b 1
)
echo cv2 / onnxruntime OK.
echo.

echo Building GUI using PyInstaller (CPU build)...
if exist "build" rmdir /s /q "build"
if exist "dist\gui" rmdir /s /q "dist\gui"
pyinstaller --noconfirm --clean gui.spec

echo.
echo CPU Build finished! Check the dist\gui folder.
if exist "requirements_cpu.txt" del /q "requirements_cpu.txt"
echo.
pause
