@echo off
setlocal

REM === 基本路徑設定 ===
set "SOURCE_PATH=D:\Git\robotlearning\yolo11_inference"
set "ENV_PYTHON=D:\miniconda\envs\yolo_anomalib\python.exe"
set "OUTPUT_PATH=D:\Git\robotlearning\build_exe"

REM === 顯示目前使用的 Python，確保是正確 env ===
echo Using Python:
"%ENV_PYTHON%" -V
echo Python executable:
echo %ENV_PYTHON%
echo.

REM === 清理輸出資料夾 ===
if exist "%OUTPUT_PATH%" rd /s /q "%OUTPUT_PATH%"
mkdir "%OUTPUT_PATH%"

REM === 執行 PyInstaller 打包（一定要用指定 env 的 python） ===
"%ENV_PYTHON%" -m PyInstaller ^
  --noconfirm ^
  --onefile ^
  --console ^
  --add-data "%SOURCE_PATH%\config.yaml;." ^
  --add-data "%SOURCE_PATH%\Runtime;Runtime/" ^
  --add-data "%SOURCE_PATH%\MvImport;MvImport/" ^
  --add-data "%SOURCE_PATH%\models;models/" ^
  --add-data "%SOURCE_PATH%\core;core/" ^
  --add-data "%SOURCE_PATH%\camera;camera/" ^
  --add-data "%SOURCE_PATH%\app;app/" ^
  --add-data "%SOURCE_PATH%\GUI.py;." ^
  --add-data "%SOURCE_PATH%\README.md;." ^
  --hidden-import torch ^
  --hidden-import torchvision ^
  --hidden-import cv2 ^
  --hidden-import scipy ^
  --hidden-import numpy ^
  --hidden-import torch.nn.functional ^
  --hidden-import kornia ^
  --hidden-import anomalib ^
  --hidden-import lightning ^
  --collect-submodules anomalib.models ^
  --collect-all kornia ^
  --collect-data anomalib ^
  --collect-data open_clip ^
  --collect-all jsonargparse ^
  --distpath "%OUTPUT_PATH%" ^
  --workpath "%OUTPUT_PATH%\build" ^
  --specpath "%OUTPUT_PATH%\specs" ^
  "%SOURCE_PATH%\GUI.py"


echo.
echo 打包完成，輸出目錄: %OUTPUT_PATH%
pause
endlocal
