@echo off
chcp 65001 >nul
title Brain Visualizer Server - test5_fast
cd /d "%~dp0"

rem ==================================================
rem  自动定位可用的 Python（按优先级）：
rem   1) conda env_torch 环境（含 numpy/torch/networkx）
rem   2) Anaconda 基环境
rem   3) PATH 中的 python
rem  每个候选都会做一次 numpy/torch 依赖自检。
rem ==================================================
set "PY="

call :pick "%~dp0..\Anaconda\envs\env_torch\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import numpy, torch, networkx" >nul 2>nul
if %errorlevel%==0 goto run

echo.
echo [错误] 未找到可用的 Python 环境。
echo   选中的解释器: %PY%
echo   缺少依赖: numpy / torch / networkx
echo.
echo 请先创建并激活 conda 环境 env_torch，然后安装依赖：
echo   conda activate env_torch
echo   pip install torch numpy networkx python-louvain
echo.
echo 之后重新双击 start_server.bat 启动。
echo.
pause
exit /b 1

:run
echo ============================================
echo  test5_fast 贪吃蛇脑活动可视化 - 后端服务
echo ============================================
echo 使用 Python: %PY%
echo 启动中，请稍候（首次约 5~15 秒加载模型与布局）...
echo.
"%PY%" brain_visualizer.py --no-browser
echo.
echo 服务已停止。按任意键关闭窗口...
pause >nul
exit /b 0

:pick
if defined PY goto :eof
if not exist "%~1" goto :eof
"%~1" -c "import numpy, torch" >nul 2>nul
if %errorlevel%==0 (
  set "PY=%~1"
)
goto :eof