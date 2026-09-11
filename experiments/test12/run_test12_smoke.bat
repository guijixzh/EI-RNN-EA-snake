@echo off
rem test12 冒烟验证：自检 + 3 代小规模端到端（新食物绝对方位编码 + 孤岛惩罚）
cd /d "%~dp0..\.."
if not defined PY set PY=python
%PY% -X utf8 -u experiments\test12\test12.py --selfcheck 2>&1 | tee logs\test12_selfcheck.log
%PY% -X utf8 -u experiments\test12\test12.py --smoke 2>&1 | tee logs\test12_smoke.log
pause
