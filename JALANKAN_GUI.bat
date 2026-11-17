@echo off
REM ARM Experiment Runner - klik dua kali file ini untuk membuka GUI di browser.
REM Memakai Python yang sama dengan yang dipakai untuk menjalankan eksperimen.
cd /d "%~dp0"
title ARM Experiment Runner
REM Hentikan server GUI lama yang masih berjalan di port 8765 (misalnya dari folder lain)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do taskkill /F /PID %%p >nul 2>&1
where python >nul 2>nul
if %errorlevel%==0 (
    python gui\app_server.py
) else (
    py -3 gui\app_server.py
)
echo.
echo Server berhenti. Tekan tombol apa saja untuk menutup jendela ini.
pause >nul
