@echo off
TITLE Pripix Launcher
CLS

ECHO ======================================================
ECHO        PRIPIX - Local AI Photo Storage
ECHO ======================================================
ECHO.

:: 1. Check for Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python is not installed!
    ECHO Please install Python 3.10+ from python.org and check "Add to PATH".
    PAUSE
    EXIT
)

:: 2. Create Virtual Environment
IF NOT EXIST "venv" (
    ECHO [1/3] Creating local environment...
    python -m venv venv
)

:: 3. Activate Environment
CALL venv\Scripts\activate

:: 4. Install Dependencies
:: CHANGED: We check for the NEW library to ensure upgrades happen
pip show reverse_geocoder >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [2/3] Installing New Libraries...
    pip install -r requirements.txt
) ELSE (
    ECHO [2/3] Dependencies up to date.
)

:: 5. Start the App
ECHO.
ECHO [3/3] Starting Pripix Server...
ECHO [INFO] Opening browser...
START index.html

ECHO [NOTE] Do not close this black window.
uvicorn backend:app --host 0.0.0.0 --port 8000

PAUSE