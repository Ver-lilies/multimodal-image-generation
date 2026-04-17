@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if exist "%~dp0..\.venv-gpu\Scripts\python.exe" (
  set "PY=%~dp0..\.venv-gpu\Scripts\python.exe"
) else if exist "%~dp0.venv310\Scripts\python.exe" (
  set "PY=%~dp0.venv310\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" scripts\download_models_stepwise.py --python "%PY%" --group large --timeout 7200
pause
