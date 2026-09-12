@echo off
setlocal
cd /d "%~dp0"
set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" -m pip --version >nul 2>nul
    if not errorlevel 1 goto check_python_version
    echo Incomplete local environment detected. Rebuilding .venv ...
    rmdir /s /q ".venv"
)

if exist ".venv" (
    echo Incomplete local environment detected. Rebuilding .venv ...
    rmdir /s /q ".venv"
)

echo Creating a local Python environment in .venv ...
set "VENV_CREATED="
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv .venv
    if not errorlevel 1 set "VENV_CREATED=1"
)

if defined VENV_CREATED goto check_python_version

where python >nul 2>nul
if errorlevel 1 goto python_not_found
python -m venv .venv
if errorlevel 1 goto create_failed

:check_python_version
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 goto python_too_old

:install_requirements
echo Installing project dependencies ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto install_failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

echo.
echo Setup completed. You can now run start_integrated_experiment.cmd.
exit /b 0

:python_not_found
echo.
echo Python was not found. Install Python 3.10 or newer, then run this file again.
pause
exit /b 1

:create_failed
echo.
echo Failed to create the local Python environment.
pause
exit /b 1

:python_too_old
echo.
echo Python 3.10 or newer is required. Please update Python and run this file again.
pause
exit /b 1

:install_failed
echo.
echo Failed to install dependencies. Check the network connection and try again.
pause
exit /b 1
