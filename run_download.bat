@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM 依赖安装与 install_deps.bat 完全相同，本脚本只多一步：下载 Hugging Face 模型（webapp 分组）
call "%~dp0install_deps.bat"
if errorlevel 1 (
  pause
  exit /b 1
)

if exist "%~dp0..\.venv-gpu\Scripts\python.exe" (
  set "PY=%~dp0..\.venv-gpu\Scripts\python.exe"
) else if exist "%~dp0.venv310\Scripts\python.exe" (
  set "PY=%~dp0.venv310\Scripts\python.exe"
) else (
  set "PY=python"
)

echo.
echo === Hugging Face 模型 ^(group webapp^) ===
echo 需 .env 中 HF_TOKEN；体积大、耗时长，请耐心等待。
echo.
"%PY%" scripts\download_models_stepwise.py --python "%PY%" --group webapp --timeout 7200
pause
