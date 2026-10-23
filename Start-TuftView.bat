@echo off
cd /d "%~dp0"
if exist .venv-unet\Scripts\python.exe (set PY=.venv-unet\Scripts\python.exe) else if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=py -3)
%PY% -m pip install -q -r Requirements.txt -r Requirements-dashboard.txt
if errorlevel 1 (echo Install failed. Is Python 3 installed? & pause & exit /b 1)
%PY% -m dashboard
pause
