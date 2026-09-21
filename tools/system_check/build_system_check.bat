@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM ==========================================================================
REM  Build the standalone deployment preflight checker.
REM  Output: dist\system_check\system_check.exe  (copy the whole folder)
REM
REM  Override the interpreter with:  set YOLO11_PYTHON=D:\...\python.exe
REM ==========================================================================

pushd "%~dp0..\.."
set "SOURCE_PATH=%CD%"

set "DEFAULT_PYTHON=D:\miniconda\envs\yolo_anomalib\python.exe"
if not "%YOLO11_PYTHON%"=="" (
    set "ENV_PYTHON=%YOLO11_PYTHON%"
) else (
    set "ENV_PYTHON=%DEFAULT_PYTHON%"
)

if not exist "%ENV_PYTHON%" (
    echo [ERROR] Python not found: %ENV_PYTHON%
    echo Set YOLO11_PYTHON to a valid interpreter.
    popd
    if not defined CI pause
    exit /b 1
)

"%ENV_PYTHON%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller is not installed. Run: pip install pyinstaller
    popd
    if not defined CI pause
    exit /b 1
)

echo [INFO] Python : %ENV_PYTHON%
echo [INFO] Source : %SOURCE_PATH%
echo.

"%ENV_PYTHON%" -m PyInstaller --clean --noconfirm ^
    --distpath "%SOURCE_PATH%\dist" ^
    --workpath "%SOURCE_PATH%\build\system_check" ^
    "%SOURCE_PATH%\tools\system_check\system_check.spec"
if errorlevel 1 (
    echo [ERROR] Build failed.
    popd
    if not defined CI pause
    exit /b 1
)

echo.
echo [INFO] Smoke test:
"%SOURCE_PATH%\dist\system_check\system_check.exe" --version
if errorlevel 1 (
    echo [ERROR] The built executable did not start.
    popd
    if not defined CI pause
    exit /b 1
)

echo.
echo [OK] Built: %SOURCE_PATH%\dist\system_check\
echo      Copy the whole system_check folder to the target machine.
popd
if not defined CI pause
exit /b 0
