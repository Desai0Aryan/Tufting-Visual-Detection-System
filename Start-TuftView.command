#!/bin/bash
# Double-click on a Mac to start TuftView. First run installs what it needs (needs internet once).
cd "$(dirname "$0")" || exit 1
if [ -d .venv ]; then source .venv/bin/activate; fi
PY=python3
$PY -m pip install -q -r Requirements.txt -r Requirements-dashboard.txt || { echo "Install failed. Is Python 3 installed?"; read -r -p "Press Return to close"; exit 1; }
$PY -m dashboard
read -r -p "TuftView stopped. Press Return to close."
