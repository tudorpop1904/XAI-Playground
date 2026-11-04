#!/usr/bin/env bash
set -euo pipefail
####################################################################################################
# PROTOTYPE 3 ---------- ./app.sh; To be revised.
#
# Parametric script to run on local Unix machine for bootstrapping the app.
#
# Flags & Arguments:
#
# --slurm: Handles Slurm logic. If set, connect to HPC via SSH (then build & run). 
#          If not, check Docker flag.
#
# -s: Short form equivalent of --slurm.
#
# --docker. Handles Docker logic. If not set, build & run locally. If set, build & run container. 
#           If -s is set, ignore (no Docker daemon on the HPC machine).
# 
# -d: Short form equivalent of --docker.
#
# --user=(str): HPC Username argument (Slurm only - ignored otherwise). 
#               Mandatory if connecting via Slurm. Also: --user <str>, -u <str>.
#
# --key=(str): SSH Key type argument (Slurm only - ignored otherwise).
#              One of (ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa). 
#              Optional, defaults to ed25519. Also: --key <str>, -k <str>.
#
# --queues: Enable RabbitMQ job queues. Off by default.
#
# -q: Short form equivalent of --queues.
#
# --help, -h: Print out the usage manual.
####################################################################################################

# Resolve the project root (this script lives in <root>/scripts/bootstrap) and run from there:
# docker-compose.yml, storage/, app.py, .venv ... all use paths relative to the root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Audio cues (best effort, never fatal; see audio/play.sh)
if [ -f "$SCRIPT_DIR/audio/play.sh" ]; then
    # shellcheck source=audio/play.sh
    . "$SCRIPT_DIR/audio/play.sh"
else
    play_sound() { :; }
fi
cue() { play_sound "$@" || true; }

# Text formatting functions
red()    { printf "\e[0;31m%s\e[0m\n" "$*"; }
green()  { printf "\e[0;32m%s\e[0m\n" "$*"; }
yellow() { printf "\e[0;33m%s\e[0m\n" "$*"; }
blue()   { printf "\e[0;34m%s\e[0m\n" "$*"; }

# Text background formatting functions
bg_red()    { printf "\e[0;41m%s\e[0m\n" "$*"; }
bg_green()  { printf "\e[0;42m%s\e[0m\n" "$*"; }
bg_yellow() { printf "\e[0;43m%s\e[0m\n" "$*"; }
bg_blue()   { printf "\e[0;44m%s\e[0m\n" "$*"; }

# Raw ANSI codes for the box-drawing helpers
ESC=$'\e'
RESET="${ESC}[0m"
BOLD="${ESC}[1m"
NOBOLD="${ESC}[22m"

# List membership check: contains <needle> <item>...
# Returns 0 if <needle> is one of the items, 1 otherwise.
contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

# —————— Box-drawing helpers (mirror Write-Formatted / Show-Usage in app.ps1) ——————
# box_edge <ansi_color> <inner_width> <left> <fill> <right>
box_edge() {
    local color="$1" width="$2" left="$3" fill="$4" right="$5" bar
    printf -v bar '%*s' "$width" ''
    bar="${bar// /$fill}"
    printf '%s%s%s%s%s%s\n' "${ESC}[${color}m" "$BOLD" "$left" "$bar" "$right" "$RESET"
}

# box_row <ansi_color> <inner_width> <side> <text> [bold] [center]
# Borders are always bold; the text is bold only if the 5th argument is "bold".
box_row() {
    local color="${ESC}[$1m" width="$2" side="$3" text="$4" weight="" left=0 right
    [ "${5:-}" = "bold" ] && weight="$BOLD"
    if [ "${6:-}" = "center" ]; then
        left=$(( (width - ${#text}) / 2 ))
    fi
    right=$(( width - ${#text} - left ))
    printf '%s%s%s%s%s%*s%s%*s%s%s%s\n' \
        "$color" "$BOLD" "$side" "$NOBOLD" "$weight" "$left" '' "$text" "$right" '' \
        "$BOLD" "$side" "$RESET"
}

# Pause between manual pages (skipped when stdin is not a terminal)
pause_page() {
    if [ -t 0 ]; then
        read -r -p "Press <Enter> to continue..." _ || true
    fi
}

# Usage manual function
usage() {
    local W=37 B=34 C=36 Y=33 T="30;47"
    echo ""
    box_edge "$T" 94 "╔" "═" "╗"
    box_row  "$T" 94 "║" "" bold
    box_row  "$T" 94 "║" "Bash bootstrap script for XAI Playground" bold center
    box_row  "$T" 94 "║" "Usage Manual" bold center
    box_row  "$T" 94 "║" "" bold
    box_edge "$T" 94 "╚" "═" "╝"
    pause_page

    box_edge "$W" 76 "┌" "─" "┐"
    box_row  "$W" 76 "│" "General" bold center
    box_row  "$W" 76 "│" "(Available for all three)" "" center
    box_row  "$W" 76 "│" " Options:" bold
    box_row  "$W" 76 "│" "   --queues,   -q                  - Enable RabbitMQ"
    box_row  "$W" 76 "│" "   --help,     -h                  - Show this help message"
    box_row  "$W" 76 "│" ""
    box_row  "$W" 76 "│" " Examples:" bold
    box_row  "$W" 76 "│" "   ./app.sh [ --queues ] [ --help ]"
    box_row  "$W" 76 "│" "   ./app.sh [ -q ] [ -h ]"
    box_edge "$W" 76 "└" "─" "┘"
    pause_page

    box_edge "$B" 83 "┌" "─" "┐"
    box_row  "$B" 83 "│" "Slurm" bold center
    box_row  "$B" 83 "│" " Options:" bold
    box_row  "$B" 83 "│" "   --slurm,  -s                   - Run on HPC cluster"
    box_row  "$B" 83 "│" "   --user,   -u <HPC_username>    - HPC username"
    box_row  "$B" 83 "│" "   --key,    -k <SSH_key_type>    - SSH key type (ed25519 or rsa)"
    box_row  "$B" 83 "│" ""
    box_row  "$B" 83 "│" " Examples:" bold
    box_row  "$B" 83 "│" "   ./app.sh --slurm --user=<HPC_username> [ --key=<SSH_key_type> ] [ --queues ]"
    box_row  "$B" 83 "│" "   ./app.sh --slurm --user <HPC_username> [ --key <SSH_key_type> ] [ --queues ]"
    box_row  "$B" 83 "│" "   ./app.sh -s -u <HPC_username> [ -k <SSH_key_type> ] [ -q ]"
    box_row  "$B" 83 "│" ""
    box_row  "$B" 83 "│" " Runs the app via an SSH session in the university's HPC cluster using Slurm." bold
    box_edge "$B" 83 "└" "─" "┘"
    pause_page

    box_edge "$C" 68 "┌" "─" "┐"
    box_row  "$C" 68 "│" "Docker" bold center
    box_row  "$C" 68 "│" " Options:" bold
    box_row  "$C" 68 "│" "   --docker, -d"
    box_row  "$C" 68 "│" ""
    box_row  "$C" 68 "│" " Examples:" bold
    box_row  "$C" 68 "│" "   ./app.sh --docker [ --queues ]"
    box_row  "$C" 68 "│" "   ./app.sh -d [ -q ]"
    box_row  "$C" 68 "│" ""
    box_row  "$C" 68 "│" " Builds the Docker image locally and runs the app in a container." bold
    box_edge "$C" 68 "└" "─" "┘"
    pause_page

    box_edge "$Y" 73 "┌" "─" "┐"
    box_row  "$Y" 73 "│" "Local" bold center
    box_row  "$Y" 73 "│" " Options:" bold
    box_row  "$Y" 73 "│" "   ONLY --queues or -q"
    box_row  "$Y" 73 "│" "   NO --slurm, -s, --docker, or -d"
    box_row  "$Y" 73 "│" ""
    box_row  "$Y" 73 "│" " Examples:" bold
    box_row  "$Y" 73 "│" "   ./app.sh [ --queues ] (or ./app.sh [ -q ])"
    box_row  "$Y" 73 "│" "     (Default mode, runs locally without Docker or Slurm)" bold
    box_edge "$Y" 73 "└" "─" "┘"
    pause_page

    printf '\e[35m%s\e[0m\n\n' "Note: Runs with both Docker and Slurm flags are not supported."
    printf '\e[36m%s\e[0m\n\n' "Valid SSH key types: ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa (default: ed25519)."
    printf '\e[2m%s\e[0m\n\n' "Set XAI_MUTE=1 to silence the audio cues."
}

# Fatal runtime error (no usage hint)
die() {
    cue error
    printf '\n\e[1;31m✖ ERROR! %s\e[0m\n\n' "$*"
    exit 1
}

# Fatal argument error: same, plus a pointer to the manual
fail() {
    cue error
    printf '\n\e[1;31m✖ ERROR! %s\e[0m\n' "$*"
    printf '\e[0;31mRefer to the user manual by running: \e[1;33m./app.sh --help\e[0m\n\n'
    exit 1
}

# Ask for consent: confirm "<question>" y|n   (2nd argument = default answer; 0 = yes, 1 = no).
# Without a terminal the default is used, so unattended runs never hang on a prompt.
confirm() {
    local prompt="$1" def="${2:-y}" hint reply=""
    if [ "$def" = y ]; then hint="Y/n"; else hint="y/N"; fi
    if [ ! -t 0 ]; then
        printf '\e[0;33m%s \e[2m[%s]\e[0;33m (no terminal: using default)\e[0m\n' "$prompt" "$hint"
        [ "$def" = y ]
        return
    fi
    printf '\e[0;33m%s \e[2m[%s]\e[0m: ' "$prompt" "$hint"
    read -r reply || reply=""
    case "$reply" in
        "")              [ "$def" = y ] ;;
        y|Y|yes|YES|Yes) return 0 ;;
        *)               return 1 ;;
    esac
}

# Service card (61 columns wide; the globe emoji is 2 columns wide)
banner() {
    local title="$1"
    blue  "┌─────────────────────────────────────────────────────────────┐"
    green "$(printf '│  🌐  %s%*s│' "$title" $((55 - ${#title})) '')"
    blue  "├─────────────────────────────────────────────────────────────┤"
    green "$(printf '│  • Streamlit UI :  %-41s│' "http://localhost:${UI_PORT}")"
    green "$(printf '│  • FastAPI Docs :  %-41s│' "http://localhost:${API_PORT}/docs")"
    blue  "└─────────────────────────────────────────────────────────────┘"
    echo ""
}

# —————— Ports ——————
port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }

# PIDs listening on a TCP port (best effort: needs lsof or ss)
port_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | sort -u || true
    elif command -v ss >/dev/null 2>&1; then
        ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
    fi
    return 0
}

# Never starts on top of something else, and never kills a process without asking
ensure_port_free() {   # ensure_port_free <port> <ENV_NAME>
    local port="$1" env_name="$2" pids pid
    port_in_use "$port" || return 0
    cue warning
    yellow "⚠ WARNING! Port ${port} (${env_name}) is already in use."
    pids="$(port_pids "$port" || true)"
    if [ -z "$pids" ]; then
        die "Port ${port} is busy and its owner could not be identified. Free it, or set ${env_name} to another port."
    fi
    for pid in $pids; do
        printf '    PID %s (%s)\n' "$pid" "$(ps -o comm= -p "$pid" 2>/dev/null || echo unknown)"
    done
    confirm "Stop it so the app can use port ${port}?" n \
        || die "Port ${port} is busy. Free it, or set ${env_name} to another port."
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    if port_in_use "$port"; then
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
    if port_in_use "$port"; then die "Could not free port ${port} (it may belong to another user)."; fi
    green "✓ Port ${port} freed."
}

# —————— uv ——————
ensure_uv() {
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv >/dev/null 2>&1; then
        green "✓ uv found."
        return 0
    fi
    cue warning
    yellow "⚠ WARNING! uv (Python package manager) was not found."
    confirm "Install it from astral.sh now? (user-level, no sudo)" y \
        || die "uv is required to run locally. Install it manually: https://docs.astral.sh/uv/"
    blue "Installing uv..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv installation failed. Install it manually: https://docs.astral.sh/uv/"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh || die "uv installation failed. Install it manually: https://docs.astral.sh/uv/"
    else
        die "Neither curl nor wget is available to download uv. Install it manually: https://docs.astral.sh/uv/"
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installation failed. Install it manually: https://docs.astral.sh/uv/"
    green "✓ uv installed."
}

# Set ports
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

# Parse arguments
IS_SLURM=0
IS_DOCKER=0
IS_LOCAL=0
QUEUES=0
user=""
key="ed25519"
keys=(ecdsa ecdsa-sk ed25519 ed25519-sk rsa)
printf -v keys_list '%s, ' "${keys[@]}"
keys_list="${keys_list%, }"

# Same accepted forms as app.ps1 / app.cmd: --user=x, --user x, -u x (likewise for --key / -k)
while [ $# -gt 0 ]; do
    case "$1" in
    -s|--slurm)
        IS_SLURM=1
        ;;
    -d|--docker)
        IS_DOCKER=1
        ;;
    -q|--queues)
        QUEUES=1
        ;;
    -h|--help)
        cue info
        usage
        exit 0
        ;;
    --user=*)
        user="${1#*=}"
        ;;
    -u|--user)
        [ $# -ge 2 ] || fail "Missing value for '$1'."
        user="$2"
        shift
        ;;
    --key=*)
        key="${1#*=}"
        ;;
    -k|--key)
        [ $# -ge 2 ] || fail "Missing value for '$1'."
        key="$2"
        shift
        ;;
    *)
        fail "Unknown argument: '$1'."
        ;;
    esac
    shift
done

# Flags forwarded to the remote app.sh (RabbitMQ is opt-in)
START_ARGS=""
if [ "$QUEUES" = 1 ]; then
    START_ARGS="-q"
fi

# If both flags are set, stop
if [ "$IS_SLURM" = 1 ] && [ "$IS_DOCKER" = 1 ]; then
    fail "--slurm and --docker are mutually exclusive."
fi

# If neither Slurm nor Docker are set, run locally
if [ "$IS_SLURM" = 0 ] && [ "$IS_DOCKER" = 0 ]; then
    IS_LOCAL=1
fi


####################################################################################################
# Case 1: Slurm (connect to the HPC login node, update the repo, run app.sh (local mode) in tmux)
####################################################################################################
run_slurm() {
    [ -n "$user" ] || fail "Slurm flag set, but user argument not set. Specify --user=<HPC_username>."

    [[ "$user" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]] \
        || fail "Username '$user' is invalid. Must start with a letter/underscore and contain only alphanumeric, '_', or '-'."

    contains "$key" "${keys[@]}" || fail "Invalid SSH key type '$key'. Must be one of: ${keys_list}."

    command -v ssh >/dev/null 2>&1 \
        || die "Command 'ssh' not found. Install an OpenSSH client (e.g. 'sudo apt install openssh-client')."

    # Add the HPC host entry as per the HPC tutorial, unless already present
    if [ -f "$HOME/.ssh/config" ] && grep -Eq '^[[:space:]]*Host[[:space:]]+hpc\.utcn([[:space:]]|$)' "$HOME/.ssh/config"; then
        yellow "Host 'hpc.utcn' already exists in ~/.ssh/config. Keeping existing config (--user/--key ignored)."
    else
        yellow "The 'hpc.utcn' SSH host is not configured yet."
        confirm "Add it to ~/.ssh/config (User ${user}, key ~/.ssh/id_${key}, forwards ${UI_PORT}/${API_PORT})?" y \
            || die "Consent denied. The 'hpc.utcn' entry is required to connect."
        if [ ! -f "$HOME/.ssh/id_${key}" ]; then
            yellow "⚠ SSH key ~/.ssh/id_${key} not found. Create one with 'ssh-keygen -t ${key}' and register the public key on the cluster."
        fi
        mkdir -p ~/.ssh
        chmod 700 ~/.ssh
        touch ~/.ssh/config
        chmod 600 ~/.ssh/config
        {
            echo ""
            echo "Host hpc.utcn"
            echo "    Hostname 193.226.17.41"
            echo "    Port 51225"
            echo "    ServerAliveInterval 60"
            echo "    User $user"
            echo "    IdentityFile ~/.ssh/id_$key"
            echo "    LocalForward ${UI_PORT} 127.0.0.1:${UI_PORT}"
            echo "    LocalForward ${API_PORT} 127.0.0.1:${API_PORT}"
        } >> ~/.ssh/config
        green "✓ Added 'hpc.utcn' entry to ~/.ssh/config."
    fi

    blue "⚡ Instantiating HPC cluster session..."
    banner "XAI Playground Remote Forwarding Configured"

    REMOTE_SETUP=$(cat <<EOF
set -euo pipefail
[ -d ~/XAI-Playground ] || git clone https://github.com/tudorpop1904/XAI-Playground.git ~/XAI-Playground
cd ~/XAI-Playground
git merge --abort 2>/dev/null || true
git fetch origin main 2>/dev/null || true
git reset --hard origin/main 2>/dev/null || true
chmod +x scripts/bootstrap/app.sh
tmux new-session -A -s xai 'bash scripts/bootstrap/app.sh ${START_ARGS}; exec bash'
EOF
)
    local rc=0
    ssh -t hpc.utcn "$REMOTE_SETUP" || rc=$?
    if [ "$rc" -ne 0 ]; then die "Remote cluster session exited with code ${rc}."; fi
    yellow "Remote session closed."
    cue shutdown
}

####################################################################################################
# Case 2: Docker (install / start the engine with consent, build and run the compose stack)
####################################################################################################
DOCKER=(docker)

# 0 = usable, 2 = daemon is up but this user may not use the socket, 1 = daemon not reachable
docker_state() {
    local out
    if out="$("${DOCKER[@]}" info 2>&1)"; then return 0; fi
    case "$out" in *"ermission denied"*) return 2 ;; esac
    return 1
}

# Wait for the daemon to come up (any state except "not reachable")
wait_for_docker() {
    local t=0 max="${1:-120}" spin='|/-\' st
    while [ "$t" -lt "$max" ]; do
        st=0; docker_state || st=$?
        if [ "$st" != 1 ]; then printf '\r\e[K'; return 0; fi
        printf '\r\e[0;34m%s Waiting for the Docker daemon... %ss\e[0m' "${spin:t%4:1}" "$t"
        sleep 1; t=$((t + 1))
    done
    printf '\r\e[K'
    return 1
}

install_docker() {
    local distro_ids=""
    if [ "$(uname -s)" = Darwin ]; then
        if command -v brew >/dev/null 2>&1; then
            confirm "Install Docker Desktop with Homebrew (brew install --cask docker)?" n || return 1
            brew install --cask docker
            return
        fi
        yellow "Homebrew not found. Install Docker Desktop manually: https://docs.docker.com/desktop/setup/install/mac-install/"
        return 1
    fi
    if [ -r /etc/os-release ]; then
        distro_ids="$(. /etc/os-release; echo "${ID:-} ${ID_LIKE:-}")"
    fi
    if [[ " $distro_ids " =~ [[:space:]](ubuntu|debian|fedora|centos|rhel)[[:space:]] ]]; then
        green "✓ Distro is supported (Ubuntu, Debian, Fedora, CentOS, RHEL)."
        yellow "Docker's official convenience script (https://get.docker.com) will run with sudo, and '${USER:-$(id -un)}' will be added to the 'docker' group."
        confirm "Install Docker Engine now?" n || return 1
        command -v curl >/dev/null 2>&1 || { yellow "curl is required to download the installer."; return 1; }
        curl -fsSL https://get.docker.com | sudo sh || return 1
        sudo usermod -aG docker "${USER:-$(id -un)}" || true
        return 0
    fi
    yellow "Automatic installation is not available here (supported: Ubuntu, Debian, Fedora, CentOS, RHEL, macOS). See https://docs.docker.com/engine/install/"
    return 1
}

start_docker_daemon() {
    if [ "$(uname -s)" = Darwin ]; then
        open -a Docker || return 1
    elif command -v systemctl >/dev/null 2>&1; then
        sudo systemctl start docker || return 1
    elif command -v service >/dev/null 2>&1; then
        sudo service docker start || return 1
    else
        return 1
    fi
    wait_for_docker 120
}

run_docker() {
    yellow "Checking Docker availability..."
    local st=1

    if ! command -v docker >/dev/null 2>&1; then
        cue warning
        yellow "⚠ WARNING! Docker Engine not found."
        if install_docker; then
            green "✓ Docker installed."
        else
            yellow "Docker was not installed."
        fi
    fi

    if command -v docker >/dev/null 2>&1; then
        st=0; docker_state || st=$?
        if [ "$st" = 1 ]; then
            cue warning
            yellow "⚠ WARNING! Docker daemon is not running."
            if confirm "Start it now?" y; then
                green "Consent granted. Starting Docker..."
                start_docker_daemon || yellow "⚠ Docker did not become ready."
                st=0; docker_state || st=$?
            else
                yellow "Consent denied. Not starting the daemon."
            fi
        fi
        if [ "$st" = 2 ]; then
            cue warning
            yellow "⚠ WARNING! This user cannot access the Docker socket (not in the 'docker' group yet; a re-login applies it)."
            if confirm "Run Docker through sudo for this session?" n; then
                DOCKER=(sudo docker)
                st=0; docker_state || st=$?
            fi
        fi
    fi

    if [ "$st" = 0 ]; then
        local compose=("${DOCKER[@]}" compose)
        if ! "${compose[@]}" version >/dev/null 2>&1; then
            if command -v docker-compose >/dev/null 2>&1; then
                if [ "${DOCKER[0]}" = sudo ]; then compose=(sudo docker-compose); else compose=(docker-compose); fi
            else
                die "Docker Compose not found. Install it: https://docs.docker.com/compose/install/"
            fi
        fi
        green "✓ Docker Engine is running."
        blue "Building and launching containers..."
        "${compose[@]}" -f docker-compose.yml up -d --build --remove-orphans || die "Docker Compose failed."
        cue startup
        green "✓ Containers started."
        banner "XAI Playground Containers Ready"
    else
        cue warning
        yellow "⚠ WARNING! Docker is not available."
        if confirm "Continue by launching locally without Docker?" y; then
            green "Consent granted. Launching locally..."
            run_local
        else
            die "Consent denied. Exiting."
        fi
    fi
}


####################################################################################################
# Case 3: Local (formerly start.sh; also what the HPC tmux session runs)
####################################################################################################
API_PID=""

# Stops Uvicorn and, inside tmux, kills the session so SSH drops back to the local machine
cleanup() {
    local code=$?
    trap - INT TERM EXIT
    printf '\n\e[0;33m%s\e[0m\n' "[!] Shutting down. Performing cleanup..."
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null || true
    fi
    printf '\e[0;33m%s\e[0m\n' "Streamlit and FastAPI/Uvicorn stopped. Cleanup complete."
    cue shutdown
    if [ -n "${TMUX:-}" ]; then
        tmux kill-session -t xai 2>/dev/null || true
    fi
    exit "$code"
}

run_local() {
    export PROJECT_ROOT

    # Enable or disable RabbitMQ
    RABBITMQ_ENABLED=false
    if [ "$QUEUES" = 1 ]; then
        RABBITMQ_ENABLED=true
        yellow "RabbitMQ message queue enabled."
    fi
    export RABBITMQ_ENABLED

    # Create storage directories (including logs for sbatch)
    mkdir -p storage/models storage/datasets storage/logs storage/images

    # Project root & node info
    green "📁 Project root set: ${PROJECT_ROOT}"
    blue "💻 Service active on node: ${SLURMD_NODENAME:-$(hostname)} (Slurm Job ID: ${SLURM_JOB_ID:-local})"

    ensure_uv
    ensure_port_free "$API_PORT" API_PORT
    ensure_port_free "$UI_PORT" UI_PORT

    # Sync venv (uv sync creates .venv if missing)
    if [ -d ".venv" ]; then
        blue "Virtual environment exists. Syncing dependencies..."
    else
        yellow "Virtual environment doesn't exist. Initializing and syncing..."
    fi
    uv sync || die "uv sync failed."

    # Check CUDA acceleration
    blue "Checking CUDA hardware acceleration..."
    uv run python -c "import torch; c = torch.cuda.is_available(); print('  -> CUDA Available:', c, '| Device:', torch.cuda.get_device_name(0) if c else 'CPU Only')" \
        || die "CUDA check failed (is torch installed?)."

    trap cleanup INT TERM EXIT

    # Run API in background
    blue "⚡ Starting FastAPI on port ${API_PORT}..."
    uv run uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
    API_PID=$!

    # Wait (up to 30 s) until FastAPI is actually listening; bail out if it died
    local tries=0 spin='|/-\'
    while [ "$tries" -lt 60 ]; do
        if ! kill -0 "$API_PID" 2>/dev/null; then
            printf '\r\e[K'
            die "FastAPI exited during startup. Check the output above."
        fi
        if port_in_use "$API_PORT"; then break; fi
        printf '\r\e[0;34m%s Waiting for FastAPI...\e[0m' "${spin:tries%4:1}"
        sleep 0.5
        tries=$((tries + 1))
    done
    printf '\r\e[K'
    port_in_use "$API_PORT" || die "FastAPI did not start listening on port ${API_PORT} within 30 s."

    cue startup
    banner "XAI Playground Web Services Ready"

    # Run Streamlit in foreground
    export API_BASE_URL="http://localhost:${API_PORT}"
    blue "🚀 Launching Streamlit UI on port ${UI_PORT}..."
    uv run streamlit run app.py --server.port "$UI_PORT" --server.address 0.0.0.0
}


# Dispatch
if [ "$IS_SLURM" = 1 ]; then
    run_slurm
elif [ "$IS_DOCKER" = 1 ]; then
    run_docker
elif [ "$IS_LOCAL" = 1 ]; then
    run_local
else
    die "Invalid state."
fi

exit 0
