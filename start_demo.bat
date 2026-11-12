@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Tomato Leaf Detector - Local Demo

set "HOST=127.0.0.1"
set "PORT=8000"
set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%\src"
set "REBUILD=0"
if /I "%~1"=="--rebuild" set "REBUILD=1"

echo ============================================================
echo   TOMATO LEAF DETECTOR - LOCAL DEMO
echo ============================================================
echo.

rem ------------------------------------------------------------
rem 1. Kiem tra Python venv
rem ------------------------------------------------------------
if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay .venv\Scripts\python.exe
    echo.
    echo Hay tao moi truong theo README:
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\Activate.ps1
    echo   python -m pip install -r requirements.txt
    echo   python -m pip install -e . --no-deps
    echo.
    pause
    exit /b 1
)

"%PYTHON%" -c "import fastapi, uvicorn, tomato_det" >nul 2>&1
if errorlevel 1 (
    echo [LOI] Thieu thu vien backend hoac package tomato_det chua duoc cai.
    echo.
    echo Chay:
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo   .venv\Scripts\python.exe -m pip install -e . --no-deps
    echo.
    pause
    exit /b 1
)

rem ------------------------------------------------------------
rem 2. Build frontend neu can
rem ------------------------------------------------------------
if not exist "frontend\dist\index.html" set "REBUILD=1"

if "%REBUILD%"=="1" (
    where npm >nul 2>&1
    if errorlevel 1 (
        echo [CANH BAO] Khong tim thay npm; se chi chay API, khong co giao dien.
        echo Hay cai Node.js 22 LTS roi chay:
        echo   cd frontend
        echo   npm install
        echo   npm run build
        echo.
    ) else (
        echo [1/2] Dang build frontend...
        pushd frontend
        call npm run build
        if errorlevel 1 (
            popd
            echo.
            echo [LOI] Build frontend that bai.
            pause
            exit /b 1
        )
        popd
    )
) else (
    echo [1/2] Dung frontend build co san: frontend\dist
)

rem ------------------------------------------------------------
rem 3. Kiem tra checkpoint
rem ------------------------------------------------------------
if exist "weights\yolo11s_coco\best.pt" (
    echo [2/2] Checkpoint: weights\yolo11s_coco\best.pt
) else (
    echo [2/2] CANH BAO: Chua co weights\yolo11s_coco\best.pt
    echo        UI se hien model "chua san sang"; upload/webcam khong co du doan.
    echo        Day la che do graceful degradation, khong phai loi khoi dong.
)
echo.

rem ------------------------------------------------------------
rem 4. Tim port trong (8000 - 8010)
rem ------------------------------------------------------------
:find_port
netstat -ano | findstr /R /C:":!PORT! .*LISTENING" >nul
if not errorlevel 1 (
    set /a PORT+=1
    if !PORT! GTR 8010 (
        echo [LOI] Khong tim thay port trong tu 8000 den 8010.
        echo Hay tat ung dung dang dung port roi chay lai.
        pause
        exit /b 1
    )
    goto :find_port
)

set "TDET_HOST=%HOST%"
set "TDET_PORT=%PORT%"

rem ------------------------------------------------------------
rem 5. Mo trinh duyet khi server san sang
rem ------------------------------------------------------------
start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest -UseBasicParsing 'http://%HOST%:%PORT%/api/health' -TimeoutSec 1 | Out-Null; Start-Process 'http://%HOST%:%PORT%'; break}catch{Start-Sleep -Milliseconds 500}}"

echo ============================================================
echo   Server: http://%HOST%:%PORT%
echo   API:    http://%HOST%:%PORT%/api/health
echo.
echo   Nhan Ctrl+C de dung server.
echo ============================================================
echo.

"%PYTHON%" -m uvicorn tomato_det.serving.main:app --host %HOST% --port %PORT% --log-level info
set "EXITCODE=%ERRORLEVEL%"

echo.
echo Server da dung (exit code %EXITCODE%).
pause
exit /b %EXITCODE%
