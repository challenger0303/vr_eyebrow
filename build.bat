@echo off
setlocal
cd /d "%~dp0"

echo =========================================
echo  VR Eyebrow Tracker - Build Script
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

:check_cv2
echo Checking for required modules (cv2)...
python -c "import cv2; print(cv2.__version__)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python module 'cv2' not found in the active environment.
    echo Install OpenCV in this environment, then rerun this script.
    echo.
    pause
    exit /b 1
)
echo cv2 OK.
echo.

:menu
echo Select an option to build:
echo [1] Build GUI Application (gui.spec)
echo [2] Build Inference Standalone (build_exe.py)
echo [3] Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto build_gui
if "%choice%"=="2" goto build_infer
if "%choice%"=="3" goto end

echo Invalid choice. Try again.
echo.
goto menu

:build_gui
echo.
echo Cleaning old build directories...
if exist "build" rmdir /s /q "build"
if exist "dist\gui" rmdir /s /q "dist\gui"
echo Building GUI using PyInstaller...
pyinstaller --noconfirm --clean gui.spec
echo.
echo GUI Build finished! Check the dist\gui folder.
goto end

:build_infer
echo.
echo Running build_exe.py...
python build_exe.py
echo.
goto end

:end
echo.
pause
