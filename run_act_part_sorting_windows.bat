@echo off
setlocal

set "PROJECT=D:\guedr\Projects\GHRC2026"
set "ISAAC_SIM=D:\guedr\Downloads\isaac-sim-standalone-5.1.0-windows-x86_64"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%PROJECT%\.windows-act-dependencies;%PROJECT%;%PYTHONPATH%"

cd /d "%PROJECT%"
call "%ISAAC_SIM%\python.bat" "%PROJECT%\run_act_part_sorting_windows.py" --duration 240 --keep-open %*
exit /b %ERRORLEVEL%
