@echo off
where py >nul 2>nul
if %errorlevel%==0 (
    py edit-image.py %*
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python edit-image.py %*
    ) else (
        echo Error: Python is not found. Make sure it's installed and added to PATH.
    )
)