@echo off
TITLE Pripix Launcher
CLS

ECHO ======================================================
ECHO        PRIPIX - Local AI Photo Storage
ECHO ======================================================
ECHO.

:: 1. Check for Python
:: We check if 'python' command exists.
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python is not installed!
    ECHO.
    ECHO Please ask your friend to install Python 3.10+ from python.org.
    ECHO IMPORTANT: They must check the box "Add Python to PATH" during install.
    ECHO.
    PAUSE
    EXIT
)

:: 2. Create Virtual Environment (If not exists)
:: This keeps the installation clean and isolated.
IF NOT EXIST "venv" (
    ECHO [1/3] Creating local environment... (One time only)
    python -m venv venv
)

:: 3. Activate Environment
:: This tells Windows to use the 'pripix' python, not the system python.
CALL venv\Scripts\activate

:: 4. Install Dependencies
:: We check if uvicorn is installed. If not, we run the heavy installation.
pip show uvicorn >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [2/3] Installing AI Libraries...
    ECHO        (This takes time depending on internet speed. Please wait.)
    pip install -r requirements.txt
) ELSE (
    ECHO [2/3] Libraries already installed. Skipping...
)

:: 5. Start the App
ECHO.
ECHO [3/3] Starting Pripix Server...
ECHO.
ECHO [SUCCESS] The app is running!
ECHO [INFO] Opening your browser...

:: Open the HTML file directly in default browser
START index.html

ECHO [NOTE] Do not close this black window while using the app.
ECHO.

:: Run the server
uvicorn backend:app --host 0.0.0.0 --port 8000

PAUSE