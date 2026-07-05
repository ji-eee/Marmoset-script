@echo off
rem Capture Front/Back And Bake - Marmoset plugin installer launcher.
rem Double-click this file to install the plugin (runs install.ps1 with
rem ExecutionPolicy Bypass so no policy change is needed).
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
    echo.
    echo Install failed. See the messages above.
)
echo.
pause
