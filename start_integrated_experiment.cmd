@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup_environment.cmd
    if errorlevel 1 exit /b 1
)

start "EEG Recorder" ".venv\Scripts\python.exe" run_experiment.py
start "" "web_experiment\index.html"
endlocal
