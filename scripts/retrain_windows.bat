@echo off
setlocal

REM Run from the repository root so relative paths remain stable.
pushd "%~dp0.."

if "%~1"=="" (
    python src\retrain.py
) else (
    python src\retrain.py %*
)

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
