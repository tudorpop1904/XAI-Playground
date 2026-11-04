@echo off
rem Delayed expansion stays OFF: with it on, '!' in arguments is silently stripped.
setlocal disabledelayedexpansion
chcp 65001 >nul

rem ================================================================================================
rem app.cmd - Windows Command Prompt bootstrap script for XAI Playground
rem Equivalent of app.sh and app.ps1 for Windows CMD environments (pure CMD: never calls PowerShell).
rem ================================================================================================

rem This script lives in <root>\scripts\bootstrap; docker-compose.yml, storage\, app.py and .venv are
rem relative to the project root, so run from there.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
pushd "%PROJECT_ROOT%"

rem Initialize ASCII 27 (Escape Character)
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%b"

set "IS_SLURM=0"
set "IS_DOCKER=0"
set "QUEUES=0"
set "USER_NAME="
set "KEY_TYPE=ed25519"

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
    if "%~2"=="" goto missing_value
    set "USER_NAME=%~2"
    shift & shift
    goto parse_loop
)
if /i "%~1"=="-u" (
    if "%~2"=="" goto missing_value
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
    if "%~2"=="" goto missing_value
    set "KEY_TYPE=%~2"
    shift & shift
    goto parse_loop
)
if /i "%~1"=="-k" (
    if "%~2"=="" goto missing_value
    set "KEY_TYPE=%~2"
    shift & shift
    goto parse_loop
)

if /i "%~1"=="--help" ( goto show_usage_exit0 )
if /i "%~1"=="-h" ( goto show_usage_exit0 )
if /i "%~1"=="/?" ( goto show_usage_exit0 )

echo.
echo %ESC%[1;31m✖ ERROR! Unknown argument: "%~1"%ESC%[0m
goto fail_hint

:missing_value
echo.
echo %ESC%[1;31m✖ ERROR! Missing value for "%~1".%ESC%[0m
goto fail_hint

:show_usage_exit0
call :cue info
call :show_usage
exit /b 0

:after_parse

rem Mutual exclusion check
if "%IS_SLURM%"=="1" if "%IS_DOCKER%"=="1" (
    echo.
    echo %ESC%[1;31m✖ ERROR! --slurm and --docker are mutually exclusive.%ESC%[0m
    goto fail_hint
)

set "START_ARGS="
if "%QUEUES%"=="1" set "START_ARGS=-q"

rem --- Dispatch ---
if "%IS_SLURM%"=="1" goto run_slurm
if "%IS_DOCKER%"=="1" goto run_docker
goto run_local

rem ================================================================================================
rem Case 1: Slurm Mode
rem ================================================================================================
:run_slurm
if "%USER_NAME%"=="" (
    echo.
    echo %ESC%[1;31m✖ ERROR! Slurm flag set, but user argument not set. Specify --user ^<HPC_username^>.%ESC%[0m
    goto fail_hint
)

rem Validate username: ^[a-zA-Z_][a-zA-Z0-9_-]*$
rem "set VAR" prints the value as data, so a hostile name can never be executed (unlike "echo %VAR%")
set USER_NAME | findstr /R /C:"^USER_NAME=[a-zA-Z_][a-zA-Z0-9_-]*$" >nul
if errorlevel 1 (
    echo.
    echo %ESC%[1;31m✖ ERROR! Invalid username. Must start with a letter/underscore and contain only alphanumeric, '_', or '-'.%ESC%[0m
    goto fail_hint
)

rem Validate key type
set "KEY_VALID=0"
for %%K in (ecdsa ecdsa-sk ed25519 ed25519-sk rsa) do (
    if /i "%KEY_TYPE%"=="%%K" set "KEY_VALID=1"
)
if "%KEY_VALID%"=="0" (
    echo.
    echo %ESC%[1;31m✖ ERROR! Invalid SSH key type. Must be one of: ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa.%ESC%[0m
    goto fail_hint
)

where ssh >nul 2>nul
if errorlevel 1 (
    echo.
    echo %ESC%[1;31m✖ ERROR! Command 'ssh' not found. Enable the Windows OpenSSH Client in Settings, Optional features.%ESC%[0m
    goto die
)

set "SSH_DIR=%USERPROFILE%\.ssh"
set "SSH_CONFIG=%SSH_DIR%\config"

rem Add the HPC host entry as per the HPC tutorial, unless already present
findstr /r /c:"^[ ]*Host[ ][ ]*hpc\.utcn[ ]*$" "%SSH_CONFIG%" >nul 2>nul
if not errorlevel 1 (
    echo %ESC%[33mHost 'hpc.utcn' already exists in ~/.ssh/config. Keeping existing config [--user/--key ignored].%ESC%[0m
    goto ssh_config_done
)
echo %ESC%[33mThe 'hpc.utcn' SSH host is not configured yet.%ESC%[0m
call :confirm "Add it to ~/.ssh/config (User %USER_NAME%, key ~/.ssh/id_%KEY_TYPE%, forwards %UI_PORT%/%API_PORT%)?" y
if not "%CONFIRM_YES%"=="1" (
    echo.
    echo %ESC%[1;31m✖ ERROR! Consent denied. The 'hpc.utcn' entry is required to connect.%ESC%[0m
    goto die
)
if not exist "%SSH_DIR%\id_%KEY_TYPE%" echo %ESC%[33m⚠ SSH key ~/.ssh/id_%KEY_TYPE% not found. Create one with 'ssh-keygen -t %KEY_TYPE%' and register the public key on the cluster.%ESC%[0m
if not exist "%SSH_DIR%" mkdir "%SSH_DIR%"
if not exist "%SSH_CONFIG%" type nul > "%SSH_CONFIG%"
>>"%SSH_CONFIG%" echo.
>>"%SSH_CONFIG%" echo Host hpc.utcn
>>"%SSH_CONFIG%" echo     Hostname 193.226.17.41
>>"%SSH_CONFIG%" echo     Port 51225
>>"%SSH_CONFIG%" echo     ServerAliveInterval 60
>>"%SSH_CONFIG%" echo     User %USER_NAME%
>>"%SSH_CONFIG%" echo     IdentityFile ~/.ssh/id_%KEY_TYPE%
>>"%SSH_CONFIG%" echo     LocalForward %UI_PORT% 127.0.0.1:%UI_PORT%
>>"%SSH_CONFIG%" echo     LocalForward %API_PORT% 127.0.0.1:%API_PORT%
echo %ESC%[32m✓ Added 'hpc.utcn' entry to ~/.ssh/config.%ESC%[0m
:ssh_config_done

echo %ESC%[1;34m⚡ Instantiating HPC cluster session...%ESC%[0m
set "BANNER_TITLE=XAI Playground Remote Forwarding Configured"
call :banner

set "REMOTE_START_ARGS="
if "%QUEUES%"=="1" set "REMOTE_START_ARGS=-q"

set "REMOTE_CMD="
set "REMOTE_CMD=%REMOTE_CMD%set -euo pipefail; "
set "REMOTE_CMD=%REMOTE_CMD%[ -d ~/XAI-Playground ] || git clone https://github.com/tudorpop1904/XAI-Playground.git ~/XAI-Playground; "
set "REMOTE_CMD=%REMOTE_CMD%cd ~/XAI-Playground; "
set "REMOTE_CMD=%REMOTE_CMD%git merge --abort 2>/dev/null || true; "
set "REMOTE_CMD=%REMOTE_CMD%git fetch origin main 2>/dev/null || true; "
set "REMOTE_CMD=%REMOTE_CMD%git reset --hard origin/main 2>/dev/null || true; "
set "REMOTE_CMD=%REMOTE_CMD%chmod +x scripts/bootstrap/app.sh; "
set "REMOTE_CMD=%REMOTE_CMD%tmux new-session -A -s xai 'bash scripts/bootstrap/app.sh %REMOTE_START_ARGS%; exec bash'"

ssh -t hpc.utcn "%REMOTE_CMD%"
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo %ESC%[1;31m✖ ERROR! Remote cluster session exited with code %RC%.%ESC%[0m
    goto die
)
echo %ESC%[33mRemote session closed.%ESC%[0m
call :cue shutdown wait
exit /b 0

rem ================================================================================================
rem Case 2: Docker Mode
rem ================================================================================================
:run_docker
echo %ESC%[1;33mChecking Docker availability...%ESC%[0m
where docker >nul 2>nul
if not errorlevel 1 goto docker_have_cli

call :cue warning
echo %ESC%[1;33m⚠ WARNING! Docker Engine not found.%ESC%[0m
call :confirm "Install Docker Desktop with winget now? (administrator approval needed; a restart may follow)" n
if not "%CONFIRM_YES%"=="1" (
    echo %ESC%[33mConsent denied. Skipping the Docker installation.%ESC%[0m
    goto docker_unavailable
)
where winget >nul 2>nul
if errorlevel 1 (
    echo %ESC%[33m⚠ winget not found. Install Docker Desktop manually: https://docs.docker.com/desktop/setup/install/windows-install/%ESC%[0m
    goto docker_unavailable
)
echo %ESC%[34mInstalling Docker Desktop, approve the administrator prompt...%ESC%[0m
winget install -e --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
set "PATH=%ProgramFiles%\Docker\Docker\resources\bin;%PATH%"
where docker >nul 2>nul
if errorlevel 1 (
    echo %ESC%[33m⚠ Docker was installed but is not on PATH yet. Restart your terminal or Windows, then re-run.%ESC%[0m
    goto docker_unavailable
)
echo %ESC%[32m✓ Docker Desktop installed.%ESC%[0m

:docker_have_cli
docker info >nul 2>nul
if not errorlevel 1 goto docker_ready

call :cue warning
echo %ESC%[1;33m⚠ WARNING! Docker daemon is not running.%ESC%[0m
call :confirm "Start it now?" y
if not "%CONFIRM_YES%"=="1" (
    echo %ESC%[33mConsent denied. Not starting the daemon.%ESC%[0m
    goto docker_unavailable
)
echo %ESC%[1;32mConsent granted. Starting Docker...%ESC%[0m
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
) else (
    net start docker >nul 2>nul
)
set "TRIES=0"
:docker_wait
docker info >nul 2>nul
if not errorlevel 1 goto docker_up
set /a TRIES+=1
if %TRIES% GTR 90 goto docker_wait_failed
<nul set /p "=%ESC%[34m.%ESC%[0m"
ping -n 3 127.0.0.1 >nul
goto docker_wait

:docker_wait_failed
echo.
echo %ESC%[33m⚠ Docker did not become ready in time.%ESC%[0m
goto docker_unavailable

:docker_up
echo.

:docker_ready
set "COMPOSE=docker compose"
docker compose version >nul 2>nul
if errorlevel 1 (
    where docker-compose >nul 2>nul
    if errorlevel 1 (
        echo.
        echo %ESC%[1;31m✖ ERROR! Docker Compose not found. Install it: https://docs.docker.com/compose/install/%ESC%[0m
        goto die
    )
    set "COMPOSE=docker-compose"
)
echo %ESC%[1;32m✓ Docker Engine is running.%ESC%[0m
echo %ESC%[1;34mBuilding and launching containers...%ESC%[0m
%COMPOSE% -f docker-compose.yml up -d --build --remove-orphans
if errorlevel 1 (
    echo.
    echo %ESC%[1;31m✖ ERROR! Docker Compose failed.%ESC%[0m
    goto die
)
call :cue startup
echo %ESC%[1;32m✓ Containers started.%ESC%[0m
set "BANNER_TITLE=XAI Playground Containers Ready"
call :banner
exit /b 0

:docker_unavailable
call :cue warning
echo %ESC%[1;33m⚠ WARNING! Docker is not available.%ESC%[0m
call :confirm "Continue by launching locally without Docker?" y
if "%CONFIRM_YES%"=="1" (
    echo %ESC%[1;32mConsent granted. Launching locally...%ESC%[0m
    goto run_local
)
echo.
echo %ESC%[1;31m✖ ERROR! Consent denied. Exiting.%ESC%[0m
goto die

rem ================================================================================================
rem Case 3: Local Mode (native CMD)
rem ================================================================================================
:run_local
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

if "%QUEUES%"=="1" (
    set "RABBITMQ_ENABLED=true"
    echo %ESC%[33mRabbitMQ message queue enabled.%ESC%[0m
) else (
    set "RABBITMQ_ENABLED=false"
)

if not exist "storage\logs"     mkdir "storage\logs"
if not exist "storage\models"   mkdir "storage\models"
if not exist "storage\datasets" mkdir "storage\datasets"
if not exist "storage\images"   mkdir "storage\images"

echo %ESC%[32m📁 Project root set: %PROJECT_ROOT%%ESC%[0m
echo %ESC%[34m💻 Service active on node: %COMPUTERNAME%%ESC%[0m

call :ensure_uv || exit /b 1
call :ensure_port_free %API_PORT% API_PORT || exit /b 1
call :ensure_port_free %UI_PORT% UI_PORT || exit /b 1

if exist ".venv" (
    echo %ESC%[34mVirtual environment exists. Syncing dependencies...%ESC%[0m
) else (
    echo %ESC%[33mVirtual environment doesn't exist. Initializing and syncing...%ESC%[0m
)
uv sync
if errorlevel 1 (
    echo.
    echo %ESC%[1;31m✖ ERROR! uv sync failed.%ESC%[0m
    goto die
)

echo %ESC%[34mChecking CUDA hardware acceleration...%ESC%[0m
uv run python -c "import torch; c = torch.cuda.is_available(); print('  -> CUDA Available:', c, '| Device:', torch.cuda.get_device_name(0) if c else 'CPU Only')"
if errorlevel 1 (
    echo.
    echo %ESC%[1;31m✖ ERROR! CUDA check failed ^(is torch installed?^).%ESC%[0m
    goto die
)

rem Start FastAPI in the background (same console, no extra window)
echo %ESC%[34m⚡ Starting FastAPI on port %API_PORT%...%ESC%[0m
start "" /B uv run uvicorn api.main:app --host 0.0.0.0 --port %API_PORT%

rem Wait (up to ~60 s) until FastAPI is actually listening
set "TRIES=0"
:api_wait
netstat -ano | findstr /R /C:":%API_PORT% .*LISTENING" >nul
if not errorlevel 1 goto api_up
set /a TRIES+=1
if %TRIES% GTR 60 goto api_failed
<nul set /p "=%ESC%[34m.%ESC%[0m"
ping -n 2 127.0.0.1 >nul
goto api_wait

:api_failed
echo.
echo %ESC%[1;31m✖ ERROR! FastAPI did not start listening on port %API_PORT% within 60 s. Check the output above.%ESC%[0m
call :stop_services
goto die

:api_up
echo.
call :cue startup
set "BANNER_TITLE=XAI Playground Web Services Ready"
call :banner
echo %ESC%[2mTip: press Ctrl+C, then N at the "Terminate batch job" prompt, to stop cleanly.%ESC%[0m
echo.

rem Run Streamlit in the foreground
set "API_BASE_URL=http://localhost:%API_PORT%"
echo %ESC%[1;34m🚀 Launching Streamlit UI on port %UI_PORT%...%ESC%[0m
uv run streamlit run app.py --server.port %UI_PORT% --server.address 0.0.0.0

call :stop_services
exit /b 0

rem ================================================================================================
rem Usage
rem ================================================================================================
:show_usage
echo.
echo %ESC%[1;30;47m╔══════════════════════════════════════════════════════════════════════════════════════════════╗%ESC%[0m
echo %ESC%[1;30;47m║                                                                                              ║%ESC%[0m
echo %ESC%[1;30;47m║                 Windows Command Prompt bootstrap script for XAI Playground                   ║%ESC%[0m
echo %ESC%[1;30;47m║                                         Usage Manual                                         ║%ESC%[0m
echo %ESC%[1;30;47m║                                                                                              ║%ESC%[0m
echo %ESC%[1;30;47m╚══════════════════════════════════════════════════════════════════════════════════════════════╝%ESC%[0m
set "DUMMY=" & set /p "DUMMY=Press <Enter> to continue..."

echo %ESC%[1m┌────────────────────────────────────────────────────────────────────────────┐%ESC%[0m
echo %ESC%[1m│                                  General                                   │%ESC%[0m
echo %ESC%[1m│%ESC%[0m                          (Available for all three)                         %ESC%[1m│%ESC%[0m
echo %ESC%[1m│ Options:                                                                   │%ESC%[0m
echo %ESC%[1m│%ESC%[0m   --queues,  -q,  /q                - Enable RabbitMQ                      %ESC%[1m│%ESC%[0m
echo %ESC%[1m│%ESC%[0m   --help,    -h,  /?                - Show this help message               %ESC%[1m│%ESC%[0m
echo %ESC%[1m│                                                                            │%ESC%[0m
echo %ESC%[1m│   Examples:                                                                │%ESC%[0m
echo %ESC%[1m│%ESC%[0m   app.cmd [ --queues ] [ --help ]                                          %ESC%[1m│%ESC%[0m
echo %ESC%[1m│%ESC%[0m   app.cmd [ -q ] [ -h ]                                                    %ESC%[1m│%ESC%[0m
echo %ESC%[1m│%ESC%[0m   app.cmd [ /q ] [ /? ]                                                    %ESC%[1m│%ESC%[0m
echo %ESC%[1m└────────────────────────────────────────────────────────────────────────────┘%ESC%[0m
set "DUMMY=" & set /p "DUMMY=Press <Enter> to continue..."

echo %ESC%[1;34m┌───────────────────────────────────────────────────────────────────────────────────┐%ESC%[0m
echo %ESC%[1;34m│                                       Slurm                                       │%ESC%[0m
echo %ESC%[1;34m│ Options:                                                                          │%ESC%[0m
echo %ESC%[1;34m│%ESC%[0m%ESC%[34m   --slurm,  -s,  /s            - Run on HPC cluster                               %ESC%[1;34m│%ESC%[0m
echo %ESC%[1;34m│%ESC%[0m%ESC%[34m   --user,   -u ^<HPC_username^>  - HPC username                                     %ESC%[1;34m│%ESC%[0m
echo %ESC%[1;34m│%ESC%[0m%ESC%[34m   --key,    -k ^<SSH_key_type^>  - SSH key type (ed25519 or rsa)                    %ESC%[1;34m│%ESC%[0m
echo %ESC%[1;34m│                                                                                   │%ESC%[0m
echo %ESC%[1;34m│ Examples:                                                                         │%ESC%[0m
echo %ESC%[1;34m│%ESC%[0m%ESC%[34m   app.cmd --slurm --user=^<HPC_username^> [ --key=^<SSH_key_type^> ] [ --queues ]     %ESC%[1;34m│%ESC%[0m
echo %ESC%[1;34m│%ESC%[0m%ESC%[34m   app.cmd --slurm --user ^<HPC_username^> [ --key ^<SSH_key_type^> ] [ --queues ]     %ESC%[1;34m│%ESC%[0m
echo %ESC%[1;34m│%ESC%[0m%ESC%[34m   app.cmd -s -u ^<HPC_username^> [ -k ^<SSH_key_type^> ] [ -q ]                       %ESC%[1;34m│%ESC%[0m
echo %ESC%[1;34m│                                                                                   │%ESC%[0m
echo %ESC%[1;34m│ Runs the app via an SSH session in the university's HPC cluster using Slurm.      │%ESC%[0m
echo %ESC%[1;34m└───────────────────────────────────────────────────────────────────────────────────┘%ESC%[0m
set "DUMMY=" & set /p "DUMMY=Press <Enter> to continue..."

echo %ESC%[1;36m┌────────────────────────────────────────────────────────────────────┐%ESC%[0m
echo %ESC%[1;36m│                               Docker                               │%ESC%[0m
echo %ESC%[1;36m│ Options:                                                           │%ESC%[0m
echo %ESC%[1;36m│%ESC%[0m%ESC%[36m   --docker, -d, /d                                                 %ESC%[1;36m│%ESC%[0m
echo %ESC%[1;36m│                                                                    │%ESC%[0m
echo %ESC%[1;36m│ Examples:                                                          │%ESC%[0m
echo %ESC%[1;36m│%ESC%[0m%ESC%[36m   app.cmd --docker [ --queues ]                                    %ESC%[1;36m│%ESC%[0m
echo %ESC%[1;36m│%ESC%[0m%ESC%[36m   app.cmd -d [ -q ]                                                %ESC%[1;36m│%ESC%[0m
echo %ESC%[1;36m│%ESC%[0m%ESC%[36m   app.cmd /d [ /q ]                                                %ESC%[1;36m│%ESC%[0m
echo %ESC%[1;36m│                                                                    │%ESC%[0m
echo %ESC%[1;36m│ Builds the Docker image locally and runs the app in a container.   │%ESC%[0m
echo %ESC%[1;36m└────────────────────────────────────────────────────────────────────┘%ESC%[0m
set "DUMMY=" & set /p "DUMMY=Press <Enter> to continue..."

echo %ESC%[1;33m┌─────────────────────────────────────────────────────────────────────────┐%ESC%[0m
echo %ESC%[1;33m│                                  Local                                  │%ESC%[0m
echo %ESC%[1;33m│ Options:                                                                │%ESC%[0m
echo %ESC%[1;33m│%ESC%[0m%ESC%[33m   ONLY --queues, -q, or /q                                              %ESC%[1;33m│%ESC%[0m
echo %ESC%[1;33m│%ESC%[0m%ESC%[33m   NO --slurm, -s, /s, --docker, -d, or /d                               %ESC%[1;33m│%ESC%[0m
echo %ESC%[1;33m│                                                                         │%ESC%[0m
echo %ESC%[1;33m│ Examples:                                                               │%ESC%[0m
echo %ESC%[1;33m│%ESC%[0m%ESC%[33m   app.cmd [ --queues ] (or app.cmd [ -q ] / app.cmd [ /q ])             %ESC%[1;33m│%ESC%[0m
echo %ESC%[1;33m│%ESC%[0m%ESC%[33m       (Default mode, runs locally without Docker or Slurm)              %ESC%[1;33m│%ESC%[0m
echo %ESC%[1;33m└─────────────────────────────────────────────────────────────────────────┘%ESC%[0m
set "DUMMY=" & set /p "DUMMY=Press <Enter> to continue..."

echo %ESC%[35mNote: Runs with both Docker and Slurm flags are not supported.%ESC%[0m
echo.
echo %ESC%[36mValid SSH key types: ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa (default: ed25519).%ESC%[0m
echo.
echo %ESC%[2mSet XAI_MUTE=1 to silence the audio cues.%ESC%[0m
echo.
exit /b 0

rem ================================================================================================
rem Failure exits and helper subroutines (always reached with "goto" / "call", never by fall-through)
rem ================================================================================================
:fail_hint
echo %ESC%[0;31mRefer to the user manual by running: %ESC%[1;33mapp.cmd --help%ESC%[0m
echo.

:die
call :cue error wait
exit /b 1

:cue
rem usage: call :cue NAME [wait]   (best effort, never fails; see audio\play.cmd)
call "%SCRIPT_DIR%audio\play.cmd" %*
exit /b 0

:confirm
rem usage: call :confirm "Question?" y|n   (2nd argument = default answer) -> CONFIRM_YES=1 or 0
rem Without a terminal, "set /p" returns at once and the default is used, so unattended runs never hang.
set "CONFIRM_YES=0"
set "REPLY="
if /i "%~2"=="y" ( set "HINT=Y/n" ) else ( set "HINT=y/N" )
set /p "REPLY=%ESC%[33m%~1 %ESC%[2m[%HINT%]%ESC%[0m: "
if not defined REPLY goto confirm_default
rem Strip quotes so the comparison below can never be broken or abused
set "REPLY=%REPLY:"=%"
if /i "%REPLY%"=="y"   set "CONFIRM_YES=1"
if /i "%REPLY%"=="yes" set "CONFIRM_YES=1"
exit /b 0
:confirm_default
if /i "%~2"=="y" set "CONFIRM_YES=1"
exit /b 0

:banner
rem Service card (61 columns wide). Only ASCII text is padded, so the emoji/bullet widths cannot skew it.
rem usage: set "BANNER_TITLE=..." & call :banner
set "BT=%BANNER_TITLE%                                                                "
call set "BT=%%BT:~0,55%%"
set "BU=http://localhost:%UI_PORT%                                                     "
call set "BU=%%BU:~0,41%%"
set "BA=http://localhost:%API_PORT%/docs                                               "
call set "BA=%%BA:~0,41%%"
echo %ESC%[1;34m┌─────────────────────────────────────────────────────────────┐%ESC%[0m
echo %ESC%[1;32m│  🌐  %BT%│%ESC%[0m
echo %ESC%[1;34m├─────────────────────────────────────────────────────────────┤%ESC%[0m
echo %ESC%[32m│  • Streamlit UI :  %BU%│%ESC%[0m
echo %ESC%[32m│  • FastAPI Docs :  %BA%│%ESC%[0m
echo %ESC%[1;34m└─────────────────────────────────────────────────────────────┘%ESC%[0m
echo.
exit /b 0

:ensure_uv
set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 (
    echo %ESC%[32m✓ uv found.%ESC%[0m
    exit /b 0
)
call :cue warning
echo %ESC%[1;33m⚠ WARNING! uv ^(Python package manager^) was not found.%ESC%[0m
call :confirm "Install it with winget now? (user-level, no administrator rights)" y
if not "%CONFIRM_YES%"=="1" (
    echo.
    echo %ESC%[1;31m✖ ERROR! uv is required to run locally. Install it manually: https://docs.astral.sh/uv/%ESC%[0m
    goto die
)
where winget >nul 2>nul
if errorlevel 1 (
    echo.
    echo %ESC%[1;31m✖ ERROR! winget not found. Install uv manually: https://docs.astral.sh/uv/%ESC%[0m
    goto die
)
echo %ESC%[34mInstalling uv...%ESC%[0m
winget install -e --id astral-sh.uv --accept-package-agreements --accept-source-agreements
set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo %ESC%[1;31m✖ ERROR! uv was installed but is not on PATH yet. Open a new terminal and re-run.%ESC%[0m
    goto die
)
echo %ESC%[32m✓ uv installed.%ESC%[0m
exit /b 0

:port_pids
rem usage: call :port_pids PORT   -> PORT_PIDS = space-separated unique PIDs listening on PORT
set "PORT_PIDS="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%~1 .*LISTENING"') do call :add_pid %%P
exit /b 0

:add_pid
echo  %PORT_PIDS% | find " %1 " >nul
if errorlevel 1 set "PORT_PIDS=%PORT_PIDS% %1"
exit /b 0

:show_pid
set "PNAME=unknown"
for /f "tokens=1 delims=," %%N in ('tasklist /FI "PID eq %~1" /NH /FO CSV 2^>nul') do set "PNAME=%%~N"
echo %PNAME% | findstr /B /C:"INFO" >nul && set "PNAME=unknown"
echo     PID %~1 ^(%PNAME%^)
exit /b 0

:ensure_port_free
rem usage: call :ensure_port_free PORT ENV_NAME
rem Never starts on top of something else, and never kills a process without asking.
call :port_pids %~1
if not defined PORT_PIDS exit /b 0
call :cue warning
echo %ESC%[1;33m⚠ WARNING! Port %~1 ^(%~2^) is already in use by:%ESC%[0m
for %%P in (%PORT_PIDS%) do call :show_pid %%P
call :confirm "Stop it so the app can use port %~1?" n
if not "%CONFIRM_YES%"=="1" (
    echo.
    echo %ESC%[1;31m✖ ERROR! Port %~1 is busy. Free it, or set the %~2 environment variable to another port.%ESC%[0m
    goto die
)
for %%P in (%PORT_PIDS%) do taskkill /F /T /PID %%P >nul 2>nul
ping -n 2 127.0.0.1 >nul
call :port_pids %~1
if defined PORT_PIDS (
    echo.
    echo %ESC%[1;31m✖ ERROR! Could not free port %~1 ^(try an elevated terminal^).%ESC%[0m
    goto die
)
echo %ESC%[32m✓ Port %~1 freed.%ESC%[0m
exit /b 0

:stop_services
rem Safe: :ensure_port_free guaranteed both ports were free (or freed with consent) at startup,
rem so anything listening now is ours.
echo.
echo %ESC%[33m[!] Shutting down. Performing cleanup...%ESC%[0m
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%API_PORT% .*LISTENING"') do taskkill /F /T /PID %%P >nul 2>nul
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%UI_PORT% .*LISTENING"') do taskkill /F /T /PID %%P >nul 2>nul
echo %ESC%[33mStreamlit and FastAPI/Uvicorn stopped. Cleanup complete.%ESC%[0m
call :cue shutdown wait
exit /b 0
