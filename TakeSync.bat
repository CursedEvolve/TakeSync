@echo off
setlocal
title TakeSync
cd /d "%~dp0"

rem ---- find Python (the "py" launcher comes with the python.org / winget installer)
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :nopython

rem ---- first run: private Python environment with TakeSync's packages
if not exist ".venv\installed.ok" (
    echo First start: setting up TakeSync. This takes a minute or two...
    if not exist ".venv\Scripts\python.exe" (
        %PY% -m venv .venv
        if errorlevel 1 goto :nopython
    )
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
    if errorlevel 1 goto :pipfail
    echo ok> ".venv\installed.ok"
)

rem ---- FFmpeg: on PATH, or in TakeSync\ffmpeg\bin
if exist "ffmpeg\bin\ffmpeg.exe" goto :run
where ffmpeg >nul 2>nul
if errorlevel 1 goto :noffmpeg

:run
".venv\Scripts\python.exe" -m takesync
if errorlevel 1 (
    echo.
    echo TakeSync closed with an error. The message above explains what went wrong.
    pause
)
exit /b 0

:nopython
echo.
echo Python 3 was not found.
echo Open PowerShell and run:   winget install -e --id Python.Python.3.12
echo Then close PowerShell and double-click TakeSync.bat again.
pause
exit /b 1

:pipfail
echo.
echo Installing TakeSync's Python packages failed (see the messages above).
echo Check your internet connection and double-click TakeSync.bat again.
pause
exit /b 1

:noffmpeg
echo.
echo FFmpeg was not found.
echo Open PowerShell and run:   winget install -e --id Gyan.FFmpeg
echo Then close PowerShell, open it again, and double-click TakeSync.bat again.
echo (Or put ffmpeg.exe and ffprobe.exe in a folder named ffmpeg\bin inside the TakeSync folder.)
pause
exit /b 1
