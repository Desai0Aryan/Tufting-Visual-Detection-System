#!/bin/bash
# Double-click on a Mac to start TuftView. The first run installs everything it needs (internet needed once).
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "Python 3 is not installed. Get it from python.org, then double-click this again."; read -r -p "Press Return to close"; exit 1; fi
"$PY" launch.py
read -r -p "TuftView stopped. Press Return to close."
