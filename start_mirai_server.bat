@echo off
setlocal

cd /d "%~dp0"

if not exist "mirai_server.py" (
    echo [ERROR] mirai_server.py was not found in:
    echo         %CD%
    pause
    exit /b 1
)

uv --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv could not be executed.
    echo         Install uv, then run the official Irodori-TTS setup first.
    pause
    exit /b 1
)

echo ============================================================
echo  MiRaiHead Irodori-TTS v4 Server
echo  URL: http://127.0.0.1:5000
echo  Voices: %CD%\voices
echo  Preload: enabled
echo  Model revision: e4aaac4df355ff560dcd35e0dae272c3a759317b
echo ============================================================
echo.

set "MIRAI_IRODORI_PRELOAD=true"
set "MIRAI_IRODORI_REVISION=e4aaac4df355ff560dcd35e0dae272c3a759317b"
uv run --no-sync python mirai_server.py
set "MIRAI_SERVER_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%MIRAI_SERVER_EXIT_CODE%"=="0" (
    echo [ERROR] Server exited with code %MIRAI_SERVER_EXIT_CODE%.
) else (
    echo Server stopped.
)
pause
exit /b %MIRAI_SERVER_EXIT_CODE%
