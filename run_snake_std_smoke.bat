@echo off
setlocal enableextensions
rem snake_std 标准实现冒烟验证：自检（18 组）+ 3 代小规模端到端 + 三观测环境 + 研究选项
cd /d "%~dp0"
if not exist logs mkdir logs

if defined PY goto run
for /f "delims=" %%C in ('where conda 2^>nul') do call :try "%%~dpC..\envs\env_torch\python.exe"
call :try "python"
call :try "py"
call :try "%USERPROFILE%\anaconda3\envs\env_torch\python.exe"
call :try "%USERPROFILE%\miniconda3\envs\env_torch\python.exe"
call :try "%LOCALAPPDATA%\anaconda3\envs\env_torch\python.exe"
call :try "%LOCALAPPDATA%\miniconda3\envs\env_torch\python.exe"
call :try "%ProgramData%\anaconda3\envs\env_torch\python.exe"
call :try "%ProgramData%\miniconda3\envs\env_torch\python.exe"
if not defined PY (
  echo [错误] 未找到含 torch 的 Python。请先激活 conda 环境，或设置 PY 环境变量后重试。
  pause
  exit /b 1
)

:run
echo 使用 Python: %PY%
echo [1/8] selfcheck（适应度/CRN/稀疏等价/观测等价/激素/鲁棒/池/固定地图/K帧消融/w_ii自连/E泄漏积分器/w_ee）...
%PY% -X utf8 -u snake_std.py --selfcheck > logs\snake_std_selfcheck.log 2>&1
type logs\snake_std_selfcheck.log
echo [2/8] smoke 默认（=16b 行为，obs40）...
%PY% -X utf8 -u snake_std.py --smoke --name sm_default > logs\snake_std_smoke_default.log 2>&1
echo [3/8] smoke --obs 32ego ...
%PY% -X utf8 -u snake_std.py --smoke --name sm_ego32 --obs 32ego > logs\snake_std_smoke_ego32.log 2>&1
echo [4/8] smoke --obs 32proj ...
%PY% -X utf8 -u snake_std.py --smoke --name sm_proj32 --obs 32proj > logs\snake_std_smoke_proj32.log 2>&1
echo [5/8] smoke 激素+轮换 ...
%PY% -X utf8 -u snake_std.py --smoke --name sm_hormone --train-hormone --cycle-pattern G1,G2,G3 > logs\snake_std_smoke_hormone.log 2>&1
echo [6/8] smoke K帧首帧输入+尾帧读出（test17）...
%PY% -X utf8 -u snake_std.py --smoke --name sm_kft --kframe-input first --kframe-read tail > logs\snake_std_smoke_kft.log 2>&1
echo [7/8] smoke w_ii+自连（test17A）...
%PY% -X utf8 -u snake_std.py --smoke --name sm_a2 --allow-self-conn --train-wii > logs\snake_std_smoke_a2.log 2>&1
echo [8/8] smoke E泄漏积分器+w_ii+w_ee（test17B'）...
%PY% -X utf8 -u snake_std.py --smoke --name sm_b2 --euler-leak --train-wii --train-wee > logs\snake_std_smoke_b2.log 2>&1
echo 全部完成：自检与 8 项冒烟日志在 logs\ 下。
pause
exit /b 0

:try
if defined PY goto :eof
set "_cand=%~1"
if exist "%_cand%" goto :tryrun
where %_cand% >nul 2>nul || goto :eof
:tryrun
"%_cand%" -c "import torch" >nul 2>nul
if %errorlevel%==0 set "PY=%_cand%"
goto :eof
