@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist logs	est7h_run103_153.log del logs	est7h_run103_153.log
Anacondanvsnv_torch\python.exe -X utf8 -u test7h.py --gens 153 2>&1 | "D:\Program Files\Git\usrin	ee.exe" logs	est7h_run103_153.log
echo.
echo === 训练结束，窗口可关闭 ===
pause
