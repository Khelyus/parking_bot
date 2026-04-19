@echo off
setlocal

cd /d "%~dp0"
title Parking Vision Bot

if not exist "runtime" mkdir "runtime" >nul 2>&1

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
    echo Created .env from .env.example
)

findstr /B /C:"TELEGRAM_BOT_TOKEN=replace-with-your-token" ".env" >nul
if %errorlevel%==0 (
    echo Set TELEGRAM_BOT_TOKEN in .env before starting the bot.
    exit /b 1
)

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and $_.CommandLine -like '*-m parking_bot*' } | Select-Object -ExpandProperty ProcessId"`) do (
    echo Stopping previous bot process %%I...
    powershell -NoProfile -Command "Stop-Process -Id %%I -Force" >nul
)

echo Starting bot in this window.
echo Keep this window open while the bot should work.
echo.

python -m parking_bot --config config/cameras.yaml --log-level INFO 2>> runtime\parking_bot_live_all.err.log
set EXIT_CODE=%errorlevel%

echo.
echo Bot stopped with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
