@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist logs\test7h_run103_153.log del logs\test7h_run103_153.log
python -X utf8 -u test7h.py --gens 153 2>&1 | "D:\Program Files\Git\usr\bin\tee.exe" logs\test7h_run103_153.log
echo.
echo === 训练结束，窗口可关闭 ===
pause
