@echo off
echo Activating Virtual Environment and Starting Web App...
call .\venv\Scripts\activate.bat
streamlit run app\app.py
pause
