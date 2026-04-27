@echo off
rem ══════════════════════════════════════════════════════════
rem  FedMorph Liver Segmentation — Start Client
rem ══════════════════════════════════════════════════════════
rem
rem  Usage:
rem    run_liver_client.bat CLIENT_ID [SERVER_ADDRESS]
rem
rem  Examples:
rem    run_liver_client.bat 0                        (uses config default)
rem    run_liver_client.bat 0 192.168.1.100:9000     (explicit server)
rem    run_liver_client.bat 1 192.168.1.100:9000
rem    run_liver_client.bat 2 192.168.1.100:9000
rem ══════════════════════════════════════════════════════════

set USE_CASE=liver_segmentation

if "%1"=="" (
    echo.
    echo  Usage: run_liver_client.bat CLIENT_ID [SERVER_ADDRESS]
    echo.
    echo  CLIENT_ID      : 0, 1, or 2
    echo  SERVER_ADDRESS  : e.g. 192.168.1.100:9000
    echo.
    pause
    exit /b 1
)

set CLIENT_ID=%1
set SERVER_ADDR=%2

rem Move to project root
cd /d "%~dp0\.."

echo ============================================================
echo  FedMorph Liver Segmentation Client %CLIENT_ID%
echo ============================================================
echo.

rem ── Override paths for this PC ──
rem Uncomment and edit the lines below for each Windows PC:
rem set FEDMORPH_DATA_DIR=D:\data\combined
rem set FEDMORPH_CLIENT_DATA_DIR=D:\data\fl_clients

if "%SERVER_ADDR%"=="" (
    echo Starting client %CLIENT_ID% (server address from config) ...
    uv run python src/use_cases/%USE_CASE%/main_client.py --client-id %CLIENT_ID%
) else (
    echo Starting client %CLIENT_ID% connecting to %SERVER_ADDR% ...
    uv run python src/use_cases/%USE_CASE%/main_client.py --client-id %CLIENT_ID% --server-address %SERVER_ADDR%
)

pause
