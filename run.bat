@echo off
cd /d "%~dp0"

echo ============================================================
echo    YOLOv5 Detection System
echo ============================================================
echo.

set "PYTHON_EXE="

:: 1. Read configured Python path from python_path.txt (fastest)
if exist "python_path.txt" (
    set /p PYTHON_EXE=<"python_path.txt"
)

:: 2. Validate the path
if defined PYTHON_EXE (
    if not exist "%PYTHON_EXE%" (
        echo [Warn] Configured Python not found, using smart launcher...
        set "PYTHON_EXE="
    )
)

:: 3. If python_path.txt works, start directly
if defined PYTHON_EXE (
    echo [Info] Python: %PYTHON_EXE%
    echo [Info] Starting application...
    echo [Info] First launch may take 10-30 seconds, please wait...
    echo [Info] Do NOT close this window while loading...
    echo.
    "%PYTHON_EXE%" main.py
    if errorlevel 1 (
        echo.
        echo [Error] Program exited with code %errorlevel%
        echo.
        pause
    )
    exit /b 0
)

:: 4. Fallback: use smart launcher (auto-detect environment)
echo [Info] No python_path.txt, using smart launcher...
echo [Info] The launcher will auto-detect Python/Conda and install dependencies.
echo.

where python >nul 2>&1
if %errorlevel%==0 (
    python launcher.py
) else (
    echo.
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.8 - 3.11:
    echo   https://www.python.org/downloads/
    echo.
    echo Or install Miniconda:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo [Error] Launcher exited with code %errorlevel%
    echo.
    pause
)
