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
set "SPEC_FILE=%SOURCE_PATH%\yolo11_inference.spec"

if not exist "%SPEC_FILE%" (
    echo [ERROR] 找不到打包規格檔: %SPEC_FILE%
    pause & exit /b 1
)

REM --- 清理上次輸出 ---
echo [INFO] 清理舊輸出目錄...
if exist "%OUTPUT_PATH%\%BUILD_NAME%" rd /s /q "%OUTPUT_PATH%\%BUILD_NAME%"
if exist "%WORK_PATH%" rd /s /q "%WORK_PATH%"

echo [INFO] 開始打包，這需要幾分鐘...
echo.

REM ==========================================================================
REM  PyInstaller 打包指令（單一真相來源：yolo11_inference.spec）
REM  所有設定（onedir/console/noupx、hidden-import、collect-*、copy-metadata、
REM  資料檔 Runtime/MvImport/timm_cache）都定義在 spec 內，並以 SPECPATH 相對解析，
REM  換機器/換路徑都不會爆。此處只負責輸出位置與覆寫。
REM  注意: 從 spec 打包時，PyInstaller 會忽略 --name/--add-data/--hidden-import
REM        等選項；要改打包內容請編輯 yolo11_inference.spec。
REM ==========================================================================
"%ENV_PYTHON%" -m PyInstaller ^
  --noconfirm ^
  --distpath "%OUTPUT_PATH%" ^
  --workpath "%WORK_PATH%" ^
  "%SPEC_FILE%"

if not exist "%OUTPUT_PATH%\%BUILD_NAME%\%BUILD_NAME%.exe" (
    echo.
    echo [ERROR] 打包失敗！請檢查上方錯誤訊息。
    pause & exit /b 1
)

echo.
echo [INFO] Running build postprocess...
"%ENV_PYTHON%" "%SOURCE_PATH%\tools\postprocess_build.py" "%SOURCE_PATH%" "%OUTPUT_PATH%\%BUILD_NAME%"
if errorlevel 1 (
    echo [ERROR] Build postprocess failed.
    pause & exit /b 1
)

copy /Y "%SOURCE_PATH%\tools\diagnostics\diagnose_camera.bat" "%OUTPUT_PATH%\%BUILD_NAME%\diagnose_camera.bat" >nul
"%ENV_PYTHON%" "%SOURCE_PATH%\tools\packaging\write_runtime_manifest.py" "%OUTPUT_PATH%\%BUILD_NAME%" --output "%OUTPUT_PATH%\%BUILD_NAME%\runtime_manifest_20260528.txt"
if errorlevel 1 (
    echo [ERROR] Runtime manifest generation failed.
    pause & exit /b 1
)

echo.
echo [INFO] Build complete. Running verification...
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
