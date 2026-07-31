@echo off
setlocal EnableExtensions
title YOLO Model Acceptance

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Inference project directory was not found: %~dp0
    exit /b 1
)
set "YOLO_CONFIG_DIR=%~dp0data\.ultralytics"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"

set "PYTHON_EXE="
if defined YOLO11_PYTHON if exist "%YOLO11_PYTHON%" set "PYTHON_EXE=%YOLO11_PYTHON%"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda\envs\anomalib_env\python.exe" set "PYTHON_EXE=D:\miniconda\envs\anomalib_env\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda\envs\yolo_anomalib\python.exe" set "PYTHON_EXE=D:\miniconda\envs\yolo_anomalib\python.exe"
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE (
    echo ERROR: Python was not found.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import cv2, onnxruntime, yaml; from PyQt5 import QtCore; from core.detection_system import DetectionSystem"
if errorlevel 1 (
    echo ERROR: Model acceptance runtime check failed.
    echo Python: %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m app.acceptance.main
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo ERROR: Model acceptance tool exited with code %APP_EXIT_CODE%.
    pause
)
exit /b %APP_EXIT_CODE%
