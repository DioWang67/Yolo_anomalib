@echo off
setlocal enabledelayedexpansion

REM --- 強制使用 UTF-8 code page，避免中文訊息顯示亂碼 ---
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"


REM ==========================================================================
REM  yolo11_inference  ─  PyInstaller 打包腳本
REM  用法: 直接雙擊執行，或在命令列執行 build_exe.bat
REM  如需覆蓋 Python 路徑: set YOLO11_PYTHON=D:\...python.exe && build_exe.bat
REM ==========================================================================

REM --- 自動偵測腳本所在目錄（不依賴 cwd）---
pushd "%~dp0"
set "SOURCE_PATH=%CD%"
popd

REM --- Python 環境設定（可透過 YOLO11_PYTHON 環境變數覆蓋）---
set "DEFAULT_PYTHON=D:\miniconda\envs\yolo_anomalib\python.exe"
if not "%YOLO11_PYTHON%"=="" (
    set "ENV_PYTHON=%YOLO11_PYTHON%"
) else (
    set "ENV_PYTHON=%DEFAULT_PYTHON%"
)

REM --- 驗證 Python 存在 ---
if not exist "%ENV_PYTHON%" (
    echo [ERROR] 找不到 Python: %ENV_PYTHON%
    echo 請修改腳本中的 DEFAULT_PYTHON，或設定環境變數 YOLO11_PYTHON。
    pause & exit /b 1
)

echo [INFO] 使用 Python: %ENV_PYTHON%
"%ENV_PYTHON%" -V
echo [INFO] 原始碼路徑: %SOURCE_PATH%
echo.

REM --- 確認 PyInstaller 已安裝 ---
"%ENV_PYTHON%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller 未安裝，請執行: pip install pyinstaller
    pause & exit /b 1
)

REM --- 輸出設定 ---
set "BUILD_NAME=yolo11_inference"
set "OUTPUT_PATH=%SOURCE_PATH%\dist"
set "WORK_PATH=%SOURCE_PATH%\build"
set "SPEC_PATH=%SOURCE_PATH%"

REM --- 清理上次輸出 ---
echo [INFO] 清理舊輸出目錄...
if exist "%OUTPUT_PATH%\%BUILD_NAME%" rd /s /q "%OUTPUT_PATH%\%BUILD_NAME%"
if exist "%WORK_PATH%" rd /s /q "%WORK_PATH%"

echo [INFO] 開始打包，這需要幾分鐘...
echo.

REM ==========================================================================
REM  PyInstaller 打包指令
REM  --onedir   : 輸出為資料夾（比 onefile 啟動快，DLL 相容性更好）
REM  --console  : 保留主控台視窗（方便看 log，可改 --noconsole 隱藏）
REM  注意: core/, app/, camera/ 不加 --add-data，PyInstaller 會透過 import
REM        分析自動處理；只有非 Python 資源才需要 --add-data
REM ==========================================================================
REM --- 執行期需要的資料檔（非 Python 程式碼）、隱藏 import、子模組收集與 metadata 保留 ---
REM --- 注意：在 ^ 續行的指令區塊中不可插入 REM 註解，否則可能導致參數被截斷或解析失敗 ---
"%ENV_PYTHON%" -m PyInstaller ^
  --noconfirm ^
  --onedir ^
  --console ^
  --name "%BUILD_NAME%" ^
  --distpath "%OUTPUT_PATH%" ^
  --workpath "%WORK_PATH%" ^
  --specpath "%SPEC_PATH%" ^
  ^
  --add-data "%SOURCE_PATH%\Runtime;Runtime" ^
  --add-data "%SOURCE_PATH%\MvImport;MvImport" ^
  --add-data "%SOURCE_PATH%\timm_cache;timm_cache" ^
  ^
  --hidden-import torch ^
  --hidden-import torch.nn.functional ^
  --hidden-import torchvision ^
  --hidden-import cv2 ^
  --hidden-import numpy ^
  --hidden-import scipy ^
  --hidden-import scipy.special._ufuncs ^
  --hidden-import PIL ^
  --hidden-import PIL._tkinter_finder ^
  --hidden-import kornia ^
  --hidden-import anomalib ^
  --hidden-import lightning ^
  --hidden-import ultralytics ^
  --hidden-import pandas ^
  --hidden-import openpyxl ^
  --hidden-import openpyxl.cell._writer ^
  --hidden-import yaml ^
  --hidden-import pydantic ^
  --hidden-import tqdm ^
  --hidden-import timm ^
  --hidden-import einops ^
  --hidden-import FrEIA ^
  --hidden-import imgaug ^
  --hidden-import PyQt5 ^
  --hidden-import PyQt5.sip ^
  --hidden-import PyQt5.QtCore ^
  --hidden-import PyQt5.QtGui ^
  --hidden-import PyQt5.QtWidgets ^
  --hidden-import pkg_resources ^
  --hidden-import importlib.metadata ^
  --hidden-import jsonargparse ^
  ^
  --collect-submodules anomalib ^
  --collect-submodules anomalib.models ^
  --collect-submodules ultralytics ^
  --collect-submodules lightning ^
  --collect-submodules timm ^
  --collect-submodules PyQt5 ^
  --collect-all kornia ^
  --collect-all jsonargparse ^
  --collect-data anomalib ^
  --collect-data open_clip ^
  --collect-data ultralytics ^
  ^
  --copy-metadata torch ^
  --copy-metadata ultralytics ^
  --copy-metadata anomalib ^
  --copy-metadata lightning ^
  ^
  "%SOURCE_PATH%\GUI.py"

if errorlevel 1 (
    echo.
    echo [ERROR] 打包失敗！請檢查上方錯誤訊息。
    pause & exit /b 1
)

echo.
echo [INFO] 複製外部資料到輸出目錄...
copy /Y "%SOURCE_PATH%\config.yaml" "%OUTPUT_PATH%\%BUILD_NAME%\config.yaml" >nul
copy /Y "%SOURCE_PATH%\config.example.yaml" "%OUTPUT_PATH%\%BUILD_NAME%\config.example.yaml" >nul
if exist "%OUTPUT_PATH%\%BUILD_NAME%\models" rd /s /q "%OUTPUT_PATH%\%BUILD_NAME%\models"
xcopy /E /I /Q "%SOURCE_PATH%\models" "%OUTPUT_PATH%\%BUILD_NAME%\models" >nul
echo [INFO] config.yaml, config.example.yaml, models/ 已複製。

echo.
echo [INFO] 打包完成，執行後置驗證...
echo.

REM --- 執行驗證腳本 ---
"%ENV_PYTHON%" "%SOURCE_PATH%\verify_build.py" "%OUTPUT_PATH%\%BUILD_NAME%"
if errorlevel 1 (
    echo [WARNING] 驗證有問題，請確認上方報告。
) else (
    echo [OK] 驗證通過。
)

echo.
echo 輸出目錄: %OUTPUT_PATH%\%BUILD_NAME%
echo 執行程式: %OUTPUT_PATH%\%BUILD_NAME%\%BUILD_NAME%.exe
echo.
pause
endlocal
