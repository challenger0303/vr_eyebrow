@echo off
setlocal
cd /d "%~dp0"

:: Unblock all files (remove Mark of the Web from downloaded ZIP)
:: This fixes "DLL load failed" errors caused by Windows blocking pyd/dll files
powershell -NoProfile -Command "Get-ChildItem -Path '%~dp0' -Recurse -File | Unblock-File" >nul 2>&1

:: Check if VC++ 2015-2022 Redistributable (x64) is installed
:: Registry key: HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64 - Installed = 1
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if errorlevel 1 (
    goto :need_redist
)

:: Also check version - require 14.30+ (VS2022)
for /f "tokens=3" %%a in ('reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Major 2^>nul ^| findstr Major') do set MAJOR=%%a
for /f "tokens=3" %%a in ('reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Minor 2^>nul ^| findstr Minor') do set MINOR=%%a
if "%MAJOR%"=="" goto :need_redist
if %MAJOR% LSS 14 goto :need_redist
if %MAJOR% EQU 14 if %MINOR% LSS 30 goto :need_redist

goto :launch

:need_redist
echo ============================================
echo  Visual C++ Redistributable Required
echo ============================================
echo.
echo This app needs Microsoft Visual C++ 2015-2022
echo Redistributable (x64) to run.
echo.
echo Installing now (silent, may take 30-60 seconds)...
echo.
if not exist "%~dp0vc_redist.x64.exe" (
    echo ERROR: vc_redist.x64.exe not found in this folder.
    echo Please reinstall VR Eyebrow Tracker.
    pause
    exit /b 1
)
"%~dp0vc_redist.x64.exe" /install /passive /norestart
if errorlevel 3010 (
    echo.
    echo Installation complete. A reboot may be recommended.
    echo Continuing anyway...
    timeout /t 3 >nul
) else if errorlevel 1 (
    echo.
    echo Installation may have failed. Please run install_vc_redist.bat manually.
    pause
    exit /b 1
) else (
    echo Installation complete.
    timeout /t 2 >nul
)

:launch
start "" "%~dp0VR Eyebrow Tracker.exe"
exit /b 0
