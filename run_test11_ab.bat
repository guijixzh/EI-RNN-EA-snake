@echo off
rem =====================================================================
rem test11 A/B chain: tournament elimination vs current two-stage selection
rem (independent series; test7g/test7h serve as external references only)
rem
rem !! Start only when GPU is free !!
rem
rem Steps:
rem   [1/5] Phase0 selection fidelity - test7h converged population (~1.5h)
rem   [2/5] Phase0 selection fidelity - test7g mid population, secondary (~45min)
rem   [3/5] Arm A two_stage  from scratch 20 gens (~1.5h)
rem   [4/5] Arm B tourn_k3   from scratch 20 gens (~1.5h)
rem   [5/5] Final verdict: paired 40-bank re-eval + pre-registered table (~25min)
rem
rem Phase0 gate: fidelity script exit code 2 (paired diff < -0.5) => STOP,
rem training arms not started. Any nonzero exit also stops the chain.
rem Arms delete their old products first, guaranteeing same from-zero start
rem (--seed 42 + same CRN seed).
rem When finished, hand results/test11_ab_verdict.json + logs/test11_*.log
rem to ZCode for analysis.
rem =====================================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=python
set TEE="D:\Program Files\Git\usr\bin\tee.exe"
if not exist logs mkdir logs

echo [1/5] Phase0 selection fidelity - test7h converged population ...
del /q logs\test11_fidelity_t7h.log 2>nul
%PY% -X utf8 -u experiments\test11_selection_fidelity.py --checkpoint test7h_econ_checkpoint.pth --tag t7h --trials 8 --gt-banks 24 > logs\test11_fidelity_t7h.log 2>&1
set RC=%errorlevel%
type logs\test11_fidelity_t7h.log
if not "%RC%"=="0" goto gate_stop

echo [2/5] Phase0 selection fidelity - test7g mid population (secondary) ...
del /q logs\test11_fidelity_t7g.log 2>nul
%PY% -X utf8 -u experiments\test11_selection_fidelity.py --checkpoint "test7g_econ_checkpoint copy.pth" --tag t7g --trials 4 --gt-banks 16 > logs\test11_fidelity_t7g.log 2>&1
set RC=%errorlevel%
type logs\test11_fidelity_t7g.log
if not "%RC%"=="0" goto gate_stop

echo [3/5] Arm A two_stage from scratch, 20 gens ...
del /q test11_econ_two_stage_checkpoint.pth test11_econ_two_stage_best_model.pth test11_econ_two_stage_latest_gen_best.pth test11_econ_two_stage_history.json 2>nul
%PY% -X utf8 -u test11.py --gens 20 --seed 42 --selection two_stage --tag two_stage 2>&1 | %TEE% logs\test11_armA.log

echo [4/5] Arm B tournament K=3 from scratch, 20 gens ...
del /q test11_econ_tourn_k3_checkpoint.pth test11_econ_tourn_k3_best_model.pth test11_econ_tourn_k3_latest_gen_best.pth test11_econ_tourn_k3_history.json 2>nul
%PY% -X utf8 -u test11.py --gens 20 --seed 42 --selection tournament --tourn-k 3 --tag tourn_k3 2>&1 | %TEE% logs\test11_armB.log

echo [5/5] Final verdict ...
%PY% -X utf8 -u experiments\test11_ab_verdict.py --gens 20 > logs\test11_verdict.log 2>&1
type logs\test11_verdict.log

echo.
echo === test11 A/B chain finished ===
echo === Notify ZCode to analyze: results\test11_ab_verdict.json + logs\test11_*.log ===
pause
exit /b 0

:gate_stop
echo.
echo === Phase0 gate STOP: fidelity gate failed (see logs above), ===
echo === training arms NOT started. Notify ZCode to analyze ===
echo === results\test11_selection_fidelity_*.json ===
pause
exit /b 1
