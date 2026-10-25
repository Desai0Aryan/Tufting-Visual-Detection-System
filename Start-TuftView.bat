@echo off
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (where python >nul 2>nul && (set PY=python) || (echo Python 3 is not installed. Get it from python.org, then double-click this again. & pause & exit /b 1))
%PY% launch.py
pause
