@echo off
rem Delayed expansion stays OFF: with it on, '!' in arguments is silently stripped.
setlocal disabledelayedexpansion

rem Run from the script's folder (start.ps1 and docker-compose.yml use relative paths)
pushd "%~dp0"

rem ================================================================================================
rem app.cmd - Windows Command Prompt bootstrap script for XAI Playground
rem Equivalent of app.sh and app.ps1 for Windows CMD environments.
rem ================================================================================================

rem Initialize ANSI color escapes
set "ESC="
for /f %%a in ('powershell -NoProfile -Command "[char]27"') do set "ESC=%%a"

set "IS_SLURM=0"
set "IS_DOCKER=0"
set "QUEUES=0"
set "USER_NAME="
set "KEY_TYPE=ed25519"
set "CHOICE="

rem Same as app.sh: ${API_PORT:-8000}, ${UI_PORT:-8501}
if not defined API_PORT set "API_PORT=8000"
if not defined UI_PORT set "UI_PORT=8501"

rem --- Parse CLI Arguments ---
:parse_loop
if "%~1"=="" goto after_parse

if /i "%~1"=="-s" ( set "IS_SLURM=1" & shift & goto parse_loop )
if /i "%~1"=="--slurm" ( set "IS_SLURM=1" & shift & goto parse_loop )
if /i "%~1"=="/s" ( set "IS_SLURM=1" & shift & goto parse_loop )

if /i "%~1"=="-d" ( set "IS_DOCKER=1" & shift & goto parse_loop )
if /i "%~1"=="--docker" ( set "IS_DOCKER=1" & shift & goto parse_loop )
if /i "%~1"=="/d" ( set "IS_DOCKER=1" & shift & goto parse_loop )

if /i "%~1"=="-q" ( set "QUEUES=1" & shift & goto parse_loop )
if /i "%~1"=="--queues" ( set "QUEUES=1" & shift & goto parse_loop )
if /i "%~1"=="/q" ( set "QUEUES=1" & shift & goto parse_loop )

rem Note: CMD treats an unquoted '=' as a separator, so --user=tspop arrives as two tokens:
rem "--user" and "tspop". The quoted form "--user=tspop" is handled via substring below.
set "ARG=%~1"
if /i "%ARG:~0,7%"=="--user=" (
    call set "USER_NAME=%%ARG:~7%%"
    shift
    goto parse_loop
)
if /i "%~1"=="--user" (
    set "USER_NAME=%~2"
    shift & shift
    goto parse_loop
)
if /i "%~1"=="-u" (
    set "USER_NAME=%~2"
    shift & shift
    goto parse_loop
)

if /i "%ARG:~0,6%"=="--key=" (
    call set "KEY_TYPE=%%ARG:~6%%"
    shift
    goto parse_loop
)
if /i "%~1"=="--key" (
    set "KEY_TYPE=%~2"
    shift & shift
    goto parse_loop
)
if /i "%~1"=="-k" (
    set "KEY_TYPE=%~2"
    shift & shift
    goto parse_loop
)

if /i "%~1"=="--help" ( goto show_usage_exit0 )
if /i "%~1"=="-h" ( goto show_usage_exit0 )
if /i "%~1"=="/?" ( goto show_usage_exit0 )

echo %ESC%[31m[ERROR] Unknown argument: %~1%ESC%[0m
call :show_usage
exit /b 1

:show_usage_exit0
call :show_usage
exit /b 0

:after_parse

rem Mutual exclusion check
if "%IS_SLURM%"=="1" if "%IS_DOCKER%"=="1" (
    echo %ESC%[31m[ERROR] --slurm and --docker are mutually exclusive.%ESC%[0m
    exit /b 1
)

set "START_ARGS="
if "%QUEUES%"=="1" set "START_ARGS=-Queues"

rem --- Dispatch ---
if "%IS_SLURM%"=="1" goto run_slurm
if "%IS_DOCKER%"=="1" goto run_docker
goto run_local

rem ================================================================================================
rem Case 1: Slurm Mode
rem ================================================================================================
:run_slurm
if "%USER_NAME%"=="" (
    echo %ESC%[31m[ERROR] Slurm flag set, but user argument not set.%ESC%[0m
    echo Usage: app.cmd --slurm --user=^<HPC_username^> [--key=^<SSH_key_type^>]
    exit /b 1
)

rem Validate username pattern: ^[a-zA-Z_][a-zA-Z0-9_-]*$
rem Passed through the environment, never spliced into code (avoids quote injection)
powershell -NoProfile -Command "if ($env:USER_NAME -cnotmatch '^[a-zA-Z_][a-zA-Z0-9_-]*$') { exit 1 }"
if %errorlevel% neq 0 (
    rem [CHANGE] Input is deliberately NOT echoed: a '&' or ')' in it would be executed/parsed by CMD.
    echo %ESC%[31m[ERROR] Invalid username. Must start with a letter/underscore and contain only alphanumeric, '_', or '-'.%ESC%[0m
    exit /b 1
)

rem Validate key type
set "KEY_VALID=0"
for %%K in (ecdsa ecdsa-sk ed25519 ed25519-sk rsa) do (
    if /i "%KEY_TYPE%"=="%%K" set "KEY_VALID=1"
)
if "%KEY_VALID%"=="0" (
    echo %ESC%[31m[ERROR] Invalid key type. Must be one of: ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa%ESC%[0m
    exit /b 1
)

set "SSH_DIR=%USERPROFILE%\.ssh"
set "SSH_CONFIG=%SSH_DIR%\config"

if not exist "%SSH_DIR%" mkdir "%SSH_DIR%"
if not exist "%SSH_CONFIG%" type nul > "%SSH_CONFIG%"

rem Check if Host hpc.utcn is present
rem Exact line match, like app.sh's grep "^Host hpc.utcn$"
findstr /r /c:"^Host hpc\.utcn *$" "%SSH_CONFIG%" >nul 2>nul
if not errorlevel 1 (
    echo %ESC%[33mHost 'hpc.utcn' already exists in ~/.ssh/config. Keeping it [--user/--key ignored].%ESC%[0m
    goto ssh_config_done
)
rem Redirect placed FIRST: "echo ...8501>>f" would be parsed as handle redirect "1>>"
>>"%SSH_CONFIG%" echo.
>>"%SSH_CONFIG%" echo Host hpc.utcn
>>"%SSH_CONFIG%" echo     Hostname 193.226.17.41
>>"%SSH_CONFIG%" echo     Port 51225
>>"%SSH_CONFIG%" echo     ServerAliveInterval 60
>>"%SSH_CONFIG%" echo     User %USER_NAME%
>>"%SSH_CONFIG%" echo     IdentityFile ~/.ssh/id_%KEY_TYPE%
>>"%SSH_CONFIG%" echo     LocalForward %UI_PORT% localhost:%UI_PORT%
>>"%SSH_CONFIG%" echo     LocalForward %API_PORT% localhost:%API_PORT%
echo %ESC%[32m[+] Added 'hpc.utcn' to %SSH_CONFIG%%ESC%[0m
:ssh_config_done

where ssh >nul 2>nul
if errorlevel 1 (
    echo %ESC%[31m[ERROR] ssh not found. Enable the Windows OpenSSH Client feature.%ESC%[0m
    exit /b 1
)

echo %ESC%[34m[*] Updating repository on the HPC login node...%ESC%[0m
ssh hpc.utcn "if [ ! -d ~/XAI-Playground ]; then git clone https://github.com/tudorpop1904/XAI-Playground.git ~/XAI-Playground; fi && cd ~/XAI-Playground && git pull origin main && chmod +x start.sh"
if %errorlevel% neq 0 (
    echo %ESC%[31m[ERROR] Failed to connect or update repo on HPC cluster.%ESC%[0m
    exit /b %errorlevel%
)

echo.
echo %ESC%[32m[+] Access URLs once running:%ESC%[0m
echo     Streamlit UI: http://localhost:%UI_PORT%
echo     FastAPI Docs: http://localhost:%API_PORT%/docs
echo.

set "REMOTE_START_ARGS="
if "%QUEUES%"=="1" set "REMOTE_START_ARGS=-q"

ssh -t hpc.utcn "cd ~/XAI-Playground && tmux new-session -A -s xai 'bash start.sh %REMOTE_START_ARGS%'"
exit /b %errorlevel%

rem ================================================================================================
rem Case 2: Docker Mode
rem ================================================================================================
:run_docker
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo %ESC%[31m[ERROR] Docker command not found.%ESC%[0m
    goto docker_fallback
)

docker info >nul 2>nul
if %errorlevel% neq 0 (
    echo %ESC%[31m[ERROR] Docker daemon is not running.%ESC%[0m
    goto docker_fallback
)

echo %ESC%[32m[+] Docker Engine is available. Launching containers...%ESC%[0m
docker compose -f docker-compose.yml up -d --build --remove-orphans
if errorlevel 1 (
    echo %ESC%[31m[ERROR] docker compose failed.%ESC%[0m
    exit /b 1
)
echo.
echo %ESC%[32m[+] Containers started:%ESC%[0m
echo     Streamlit UI: http://localhost:%UI_PORT%
echo     FastAPI Docs: http://localhost:%API_PORT%/docs
exit /b 0

:docker_fallback
set /p "CHOICE=Docker is unavailable. Continue by launching locally? (Default: yes) [y/n]: "
rem Same semantics as app.sh: only y/Y/empty means yes
if "%CHOICE%"=="" goto run_local
if /i "%CHOICE%"=="y" goto run_local
echo %ESC%[31mConsent for local build denied. Stopping...%ESC%[0m
exit /b 1

rem ================================================================================================
rem Case 3: Local Mode
rem ================================================================================================
:run_local
echo %ESC%[34m[*] Launching local application via PowerShell start.ps1...%ESC%[0m
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %START_ARGS%
exit /b %errorlevel%

rem ================================================================================================
rem Usage
rem ================================================================================================
:show_usage
echo ========================================================================================
echo %ESC%[32mUsage for Slurm : app.cmd --slurm --user=^<HPC_username^> [--key=^<SSH_key_type^>]%ESC%[0m
echo %ESC%[32m                  app.cmd -s -u ^<HPC_username^> [-k ^<SSH_key_type^>]%ESC%[0m
echo ========================================================================================
echo %ESC%[36mUsage for Docker: app.cmd --docker (or app.cmd -d)%ESC%[0m
echo ========================================================================================
echo %ESC%[33mUsage for Local : app.cmd [--queues] (or app.cmd -q)%ESC%[0m
echo ========================================================================================
echo.
echo Note: Runs with both Docker and Slurm flags are not supported.
echo Valid SSH key types: ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa (default: ed25519)
exit /b 0
