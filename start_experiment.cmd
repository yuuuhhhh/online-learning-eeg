@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup_environment.cmd
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" run_experiment.py
if errorlevel 1 pause
endlocal
