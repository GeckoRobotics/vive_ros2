@echo off

echo Creating virtual environment... && call python -m venv .venv && echo Activating virtual environment... && call .venv/Scripts/activate && echo Installing dependencies in environment... && pip install -r requirements.txt

