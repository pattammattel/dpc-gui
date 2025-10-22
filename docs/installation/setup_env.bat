@echo off
REM Auto-create a venv and install dependencies on Windows

setlocal
set PROJECT_DIR=%~dp0\..
cd /d %PROJECT_DIR%

echo Creating virtual environment in .venv...
python -m venv .venv

echo Activating environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo 💡 Done. To activate later:
echo     call .venv\Scripts\activate.bat
echo.
pause
