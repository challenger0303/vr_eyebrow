@echo off
echo ============================================
echo  Visual C++ Redistributable Installer
echo ============================================
echo.
echo This installs Microsoft Visual C++ 2015-2022 Redistributable (x64),
echo which is required by ONNX Runtime and PyQt5.
echo.
echo If VR Eyebrow Tracker fails to start with a "DLL load failed" error,
echo run this installer and try again.
echo.
pause

if not exist "%~dp0vc_redist.x64.exe" (
    echo ERROR: vc_redist.x64.exe not found in this folder.
    pause
    exit /b 1
)

"%~dp0vc_redist.x64.exe" /install /passive /norestart
echo.
if errorlevel 1 (
    echo Installation may have failed or required a reboot.
) else (
    echo Installation complete.
)
echo.
pause
