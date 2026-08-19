@echo off
setlocal

rem PROJECT is this script's own directory (portable across machines/checkouts).
set "PROJECT=%~dp0"
if "%PROJECT:~-1%"=="\" set "PROJECT=%PROJECT:~0,-1%"

rem ISAAC_SIM must point at your local Isaac Sim standalone install. Set it as
rem an environment variable before calling this script, e.g.:
rem   set "ISAAC_SIM=C:\path\to\isaac-sim-standalone-5.1.0-windows-x86_64"
if not defined ISAAC_SIM (
    echo ERROR: ISAAC_SIM environment variable is not set.
    echo Set it to your local Isaac Sim standalone install directory, e.g.:
    echo   set "ISAAC_SIM=C:\path\to\isaac-sim-standalone-5.1.0-windows-x86_64"
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%PROJECT%\.windows-act-dependencies;%PROJECT%;%PYTHONPATH%"

cd /d "%PROJECT%"
call "%ISAAC_SIM%\python.bat" "%PROJECT%\run_act_packing_box_windows.py" --keep-open %*
exit /b %ERRORLEVEL%
