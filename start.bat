@echo off
REM 指定 Miniconda 路徑（可用環境變數 YOLO11_CONDA / YOLO11_PYTHON 覆蓋）
if "%YOLO11_CONDA%"=="" (set "CONDA_PATH=D:\miniconda") else (set "CONDA_PATH=%YOLO11_CONDA%")
if "%YOLO11_PYTHON%"=="" (set "PYTHON_EXE=%CONDA_PATH%\envs\yolo_anomalib\python.exe") else (set "PYTHON_EXE=%YOLO11_PYTHON%")

REM 設定環境變數 PATH
set PATH=%CONDA_PATH%;%CONDA_PATH%\Scripts;%CONDA_PATH%\Library\bin;%PATH%

REM 啟動 Conda 並激活環境
call %CONDA_PATH%\Scripts\activate.bat
call conda activate yolo_anomalib

REM 切換到本腳本所在目錄（不依賴 repo 在固定磁碟位置）
cd /d "%~dp0"

REM 確認 Python 環境和 torch 模組
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found: %PYTHON_EXE%
    exit /b 1
)

"%PYTHON_EXE%" --version
"%PYTHON_EXE%" -c "import torch; print(torch.__version__)"

REM 執行主控腳本並保持視窗開啟
"%PYTHON_EXE%" GUI.py %*
cmd /k
