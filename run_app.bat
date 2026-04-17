@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem 可选：在此行下面取消注释并填写，可强制指定解释器
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
  echo 未找到 Python。请在 run_app.bat 顶部设置 PYTHON= 完整路径。
  pause
  exit /b 1
)

echo 使用: "%PYTHON%"
echo 后端 + 页面: http://127.0.0.1:8000  ^(index.html 与 API 同端口^)
echo.

start "multimodal-api" /D "%~dp0" "%PYTHON%" "%~dp0api_server.py"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000"

echo 已打开浏览器；标题为 multimodal-api 的窗口为服务进程，关闭即停止。
pause
