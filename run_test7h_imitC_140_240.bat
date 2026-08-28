@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist logs\test7h_imitC_run140_240.log del logs\test7h_imitC_run140_240.log
python -X utf8 -u test7h.py --gens 240 --te-elite 32 --imitation-w 2.0 --resume-pop test7h_imit_C.pth --name imitC 2>&1 | "D:\Program Files\Git\usr\bin\tee.exe" logs\test7h_imitC_run140_240.log
echo.
echo === 训练结束，窗口可关闭 ===
pause
