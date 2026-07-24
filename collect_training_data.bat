@echo off
setlocal EnableExtensions
title YOLO Training Data Collector
cd /d "%~dp0"

if /I "%~1"=="--check" (
    echo Collector launcher OK
    exit /b 0
)

echo ============================================================
echo YOLO Training Data Collector
echo ============================================================
echo.

set "PYTHON_EXE="
set "PYTHON_ARGS="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    )
)
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE goto :python_missing

echo Opening button-based review window...
"%PYTHON_EXE%" %PYTHON_ARGS% tools\review_training_data.py --result-root Result --manifest review_manifest.csv
if errorlevel 1 goto :failed

echo.
echo Review window closed. Every decision was saved immediately.
goto :done

:python_missing
echo ERROR: Python was not found.
echo Install the project environment or place .venv beside this file.
goto :done_error

:failed
echo.
echo ERROR: Collection failed. The console will remain open.

:done_error
pause
exit /b 1

:done
pause
exit /b 0
