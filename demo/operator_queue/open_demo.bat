@echo off
setlocal
cd /d "%~dp0..\.."
python -m tools.open_training_batch_demo
if errorlevel 1 pause
exit /b %errorlevel%
