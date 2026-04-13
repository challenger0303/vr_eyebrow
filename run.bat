@echo off
setlocal
cd /d "%~dp0"

echo =========================================
echo  VR Eyebrow Tracker
echo =========================================

:: Find Python environment
set "PY="
if exist "venv_gpu\Scripts\python.exe" set "PY=venv_gpu\Scripts\python.exe"
if not defined PY if exist "venv_cpu\Scripts\python.exe" set "PY=venv_cpu\Scripts\python.exe"

if not defined PY (
    echo No virtual environment found. Creating venv_gpu...
    python -m venv "venv_gpu"
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Make sure Python is installed and on PATH.
        pause
        exit /b 1
    )
    set "PY=venv_gpu\Scripts\python.exe"
)

echo Using: %PY%

:: Check dependencies
"%PY%" -c "import cv2; import onnxruntime; import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install onnxruntime-directml
    "%PY%" -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements.
        pause
        exit /b 1
    )
)

:: Auto-export ONNX model if only .pth exists
if not exist "eyebrow_model.onnx" (
    if exist "tinybrownet_best.pth" (
        echo Exporting ONNX model from tinybrownet_best.pth...
        "%PY%" -c "from onnx_inference import export_pth_to_onnx; export_pth_to_onnx('tinybrownet_best.pth', 'eyebrow_model.onnx')"
    )
)

:: Verify ONNX Runtime provider
echo.
"%PY%" -c "import onnxruntime as ort; ps=ort.get_available_providers(); print('ONNX Providers:', ps); assert 'DmlExecutionProvider' in ps, 'DirectML not available!'"
if errorlevel 1 (
    echo WARNING: DirectML not available. Will fall back to CPU.
    echo.
)

:: Launch
echo.
echo Starting VR Eyebrow Tracker...
"%PY%" gui.py

pause