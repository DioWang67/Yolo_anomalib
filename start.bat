@echo off
setlocal EnableExtensions

set "CHECK_MODE=0"
if /I "%~1"=="--check" set "CHECK_MODE=1"

if defined YOLO11_CONDA (
    set "CONDA_PATH=%YOLO11_CONDA%"
) else (
    set "CONDA_PATH=D:\miniconda"
)

if defined YOLO11_PYTHON (
    set "PYTHON_EXE=%YOLO11_PYTHON%"
) else (
    set "PYTHON_EXE=%CONDA_PATH%\envs\yolo_anomalib\python.exe"
)

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Inference project directory was not found: %~dp0
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment was not found: %PYTHON_EXE%
    echo Set YOLO11_CONDA or YOLO11_PYTHON to a valid environment.
    exit /b 1
)

if not defined YOLO11_PYTHON (
    if not exist "%CONDA_PATH%\Scripts\activate.bat" (
        echo ERROR: Conda activation script was not found: %CONDA_PATH%
        exit /b 1
    )
    call "%CONDA_PATH%\Scripts\activate.bat" yolo_anomalib
    if errorlevel 1 (
        echo ERROR: Conda environment activation failed: yolo_anomalib
        exit /b 1
    )
)

"%PYTHON_EXE%" tools\check_runtime_environment.py
if errorlevel 1 (
    echo ERROR: Python runtime environment check failed: %PYTHON_EXE%
    exit /b 1
)

"%PYTHON_EXE%" GUI.py --check-onnxruntime
if errorlevel 1 (
    echo ERROR: ONNX Runtime check failed: %PYTHON_EXE%
    exit /b 1
)

if "%CHECK_MODE%"=="1" (
    echo Inference launcher OK
    echo Python: %PYTHON_EXE%
    exit /b 0
)

"%PYTHON_EXE%" GUI.py %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo ERROR: Inference application exited with code %APP_EXIT_CODE%.
    pause
)
exit /b %APP_EXIT_CODE%
