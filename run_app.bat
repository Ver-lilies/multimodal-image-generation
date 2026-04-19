@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Optional: uncomment and set to force Python path
rem set "PYTHON=C:\path\to\python.exe"

if not defined PYTHON (
  if exist "%~dp0..\.venv-gpu\Scripts\python.exe" set "PYTHON=%~dp0..\.venv-gpu\Scripts\python.exe"
)
if not defined PYTHON (
  if exist "%~dp0.venv310\Scripts\python.exe" set "PYTHON=%~dp0.venv310\Scripts\python.exe"
)
if not defined PYTHON (
  where python >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
  echo Python not found. Edit run_app.bat and set PYTHON= to your python.exe path.
  pause
  exit /b 1
)

echo Using: "%PYTHON%"
echo Server: http://127.0.0.1:8000  ^(index.html + API on same port^)
echo.

start "multimodal-api" /D "%~dp0" "%PYTHON%" "%~dp0api_server.py"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000"

echo Browser launched. Close the window titled "multimodal-api" to stop the server.
pause
