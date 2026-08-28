@echo off
chcp 65001 >nul
REM =====================================================================
REM test11 A/B 一键实验链：锦标赛淘汰制 vs 现行两阶段快速筛选（独立系列）
REM
REM ★ 必须等 test7h 103-153 训练结束（GPU 空闲）后再启动本脚本 ★
REM
REM 阶段：
REM   [1/5] Phase0 筛选保真度 — test7h 收敛种群（约 1.5 小时）
REM   [2/5] Phase0 筛选保真度 — test7g 中期种群，次要（约 45 分钟）
REM   [3/5] A 臂 two_stage  从零 20 代（约 1.5 小时）
REM   [4/5] B 臂 tourn_k3   从零 20 代（约 1.5 小时）
REM   [5/5] 终局验收：40 库配对复评 + 预注册判定表（约 25 分钟）
REM
REM Phase0 门判定不过（选中集真值差 <-0.5 分）会自动停止训练臂。
REM 两臂启动前会删除旧产物，保证从零同起点（--seed 42 + 同 CRN）。
REM 跑完后把 logs\test11_console.log 尾部 / results\test11_*.json 交给 ZCode 分析。
REM =====================================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=python
set TEE="D:\Program Files\Git\usr\bin\tee.exe"
if not exist logs mkdir logs

echo [1/5] Phase0 筛选保真度 — test7h 收敛种群 ...
%PY% -X utf8 -u experiments\test11_selection_fidelity.py --checkpoint test7h_econ_checkpoint.pth --tag t7h --trials 8 --gt-banks 24 > logs\test11_fidelity_t7h.log 2>&1
type logs\test11_fidelity_t7h.log
if errorlevel 2 goto gate_stop

echo [2/5] Phase0 筛选保真度 — test7g 中期种群（次要）...
%PY% -X utf8 -u experiments\test11_selection_fidelity.py --checkpoint "test7g_econ_checkpoint copy.pth" --tag t7g --trials 4 --gt-banks 16 > logs\test11_fidelity_t7g.log 2>&1
type logs\test11_fidelity_t7g.log
if errorlevel 2 goto gate_stop

echo [3/5] A 臂 two_stage 从零 20 代 ...
del /q test11_econ_two_stage_checkpoint.pth test11_econ_two_stage_best_model.pth test11_econ_two_stage_latest_gen_best.pth test11_econ_two_stage_history.json 2>nul
%PY% -X utf8 -u test11.py --gens 20 --seed 42 --selection two_stage --tag two_stage 2>&1 | %TEE% logs\test11_armA.log

echo [4/5] B 臂 tournament K=3 从零 20 代 ...
del /q test11_econ_tourn_k3_checkpoint.pth test11_econ_tourn_k3_best_model.pth test11_econ_tourn_k3_latest_gen_best.pth test11_econ_tourn_k3_history.json 2>nul
%PY% -X utf8 -u test11.py --gens 20 --seed 42 --selection tournament --tourn-k 3 --tag tourn_k3 2>&1 | %TEE% logs\test11_armB.log

echo [5/5] 终局验收 ...
%PY% -X utf8 -u experiments\test11_ab_verdict.py --gens 20 > logs\test11_verdict.log 2>&1
type logs\test11_verdict.log

echo.
echo === test11 A/B 全链完成 ===
echo === 请通知 ZCode 分析：results\test11_ab_verdict.json + logs\test11_*.log ===
pause
exit /b 0

:gate_stop
echo.
echo === Phase0 门判定 STOP：锦标赛选中集真值显著差于两阶段（配对差 < -0.5 分），训练臂未启动 ===
echo === 请通知 ZCode 分析 results\test11_selection_fidelity_*.json ===
pause
exit /b 1
