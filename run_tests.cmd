@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run setup_environment.cmd first.
  popd
  exit /b 1
)

".venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py"
if errorlevel 1 (
  popd
  exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
  echo [WARN] Node.js not found; Python tests passed, browser static checks skipped.
  popd
  exit /b 0
)

node tests\test_browser_v21.js
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
