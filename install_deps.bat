@echo off
REM 只装 Python 环境：CUDA 版 PyTorch + requirements.txt（不下载 HF 模型）
REM 若还要预下载模型，请运行 run_download.bat（会先执行本脚本再下载）
setlocal EnableExtensions
cd /d "%~dp0"

if exist "%~dp0..\.venv-gpu\Scripts\python.exe" (
  set "PY=%~dp0..\.venv-gpu\Scripts\python.exe"
) else if exist "%~dp0.venv310\Scripts\python.exe" (
  set "PY=%~dp0.venv310\Scripts\python.exe"
) else (
  set "PY=python"
)

echo Using: %PY%

REM 先装官方 CUDA 版 PyTorch，避免 pip 从 PyPI 装 CPU 版并卸掉你已有的 +cu 轮子
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
echo [1/2] PyTorch GPU ^(torch torchvision torchaudio^) from %TORCH_INDEX%
echo      若需其他 CUDA 版本，请改下方 TORCH_INDEX，见 https://pytorch.org
"%PY%" -m pip install torch torchvision torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 (
  echo PyTorch GPU install failed.
  exit /b 1
)

echo.
echo [2/2] requirements.txt ^(transformers, diffusers, controlnet-aux, fastapi, ...^)
"%PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo pip install failed.
  exit /b 1
)
echo OK. GPU 版 PyTorch 已优先安装，其余包不会替换为 CPU torch。
exit /b 0
