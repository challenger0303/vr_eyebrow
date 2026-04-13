@echo off
echo ============================================
echo  VR Eyebrow Tracker - Uninstaller
echo ============================================
echo.
echo This will remove:
echo   1. Training Python environment (%APPDATA%\VREyebrowTracker\training_python)
echo   2. App data and settings      (%APPDATA%\VREyebrowTracker)
echo.
echo The application folder itself will NOT be deleted.
echo.
set /p CONFIRM="Are you sure? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Removing training Python environment...
if exist "%APPDATA%\VREyebrowTracker\training_python" (
    rmdir /s /q "%APPDATA%\VREyebrowTracker\training_python"
    echo   Done.
) else (
    echo   Not found, skipping.
)

echo Removing app data and settings...
if exist "%APPDATA%\VREyebrowTracker" (
    rmdir /s /q "%APPDATA%\VREyebrowTracker"
    echo   Done.
) else (
    echo   Not found, skipping.
)

echo.
echo ============================================
echo  Uninstall complete.
echo  You can now delete this application folder.
echo ============================================
pause
