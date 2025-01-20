@echo off
cd /d "%~dp0"
call "f:/ai-tk-ui/ai-toolkit/venv/Scripts/activate.bat"
python gui.py
pause