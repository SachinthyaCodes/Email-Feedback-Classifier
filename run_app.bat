@echo off
echo Starting Feedback Sentiment Analyzer...
echo.
echo 1. Starting FastAPI backend (http://127.0.0.1:8000)
start cmd /k "cd /d %~dp0 && call env\Scripts\activate && uvicorn app.api:app --reload"
echo.
echo 2. Starting Streamlit frontend (http://localhost:8501)
start cmd /k "cd /d %~dp0 && call env\Scripts\activate && streamlit run app/streamlit_app.py"
echo.
echo Both applications should start in separate windows.
echo Visit http://localhost:8501 in your browser to use the Streamlit app.
echo.
pause
