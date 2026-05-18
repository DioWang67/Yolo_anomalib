@echo off
setlocal

set "PYTHON_EXE=D:\miniconda\envs\yolo_anomalib\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" tools\pcba_pilot.py %*
) else (
    python tools\pcba_pilot.py %*
)
