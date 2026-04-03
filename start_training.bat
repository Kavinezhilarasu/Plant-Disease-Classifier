@echo off
echo Activating Virtual Environment and Starting Training...
call .\venv\Scripts\activate.bat
python train.py
pause
