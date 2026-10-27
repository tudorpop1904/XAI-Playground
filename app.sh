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
#               Mandatory if connecting via Slurm.
#
# --key=(str): SSH Key type argument (Slurm only - ignored otherwise).
#              One of (ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa). 
#              Optional, defaults to ed25519.
#
# --queues: Enable RabbitMQ job queues (forwarded to start.sh as -q). Off by default.
#
# -q: Short form equivalent of --queues.
####################################################################################################

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

# Usage manual function
usage() {
    echo -e "========================================================================================"
    echo -e "Usage for \e[0;32mSlurm\e[0m: $0 \e[0;32m--slurm \e[0m(\e[0;32mor -s\e[0m)\e[0;32m --user=\e[0m<HPC_username> [ \e[0;32m--key=\e[0m<SSH_key_type> ]"
    echo -e "========================================================================================"
    echo -e "Usage for \e[0;34mDocker\e[0m: $0 \e[0;34m--docker \e[0m(\e[0;34mor -d\e[0m)"
    echo -e "========================================================================================\n"
    echo -e "========================================================================================"
    echo -e "\e[1mNote\e[0m: Since the \e[0;34mDocker\e[0m daemon is not available on the HPC machine, \e[0;31mruns with both \e[0;34mDocker\e[0;31m and \e[0;32mSlurm\e[0;34m flags are not supported.\e[0m"
    echo -e "========================================================================================\n"
    echo -e "========================================================================================"
    echo -e "For the \e[0;32m--user\e[0m argument (\e[0;32mSlurm\e[0m):"
    echo -e "* It must begin with a letter (a-z, A-Z), or with an underscore (_)."
    echo -e "* It can only contain alphanumeric characters (a-z, A-Z, 0-9), underscores (_), and hyphens (-)."
    echo -e "* Any other characters (spaces, hashtags, asterisks, etc.) are \e[0;31mnot permitted\e[0m.\n"
    echo -e "========================================================================================\n"
    echo -e "========================================================================================"
    echo -e "For the \e[0;32m--key\e[0m argument (\e[0;32mSlurm\e[0m):"
    echo -e "* It must be one of ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa, or empty."
    echo -e "* CAN BE EMPTY, in which case, it defaults to \e[0;32med25519\e[0m."
    echo -e "* Although there is no check for this, the key type \e[0;32mmust correspond to your SSH key of choice\e[0m, such that the script can invoke the \e[0;32mid_<key-type>\e[0m file from your ~/.ssh directory."
    echo -e "* Anything else is deemed \e[0;31minvalid\e[0m."
    echo -e "========================================================================================"
}

# Print usage, an error message, then stop
fail() {
    usage
    echo
    red "ERROR! $*"
    exit 1
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

for arg in "$@"; do
    case "$arg" in
    -s|--slurm)
        IS_SLURM=1
        ;;
    -d|--docker)
        IS_DOCKER=1
        ;;
    -q|--queues)
        QUEUES=1
        ;;
    --user=*)
        user="${arg#*=}"
        ;;
    --key=*)
        key="${arg#*=}"
        ;;
    *)
        fail "Unknown argument: '$arg'."
        ;;
    esac
done

# start.sh flags that get forwarded (RabbitMQ is opt-in)
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
# Case 1: Slurm (connect to the HPC login node, update the repo, run start.sh in tmux)
####################################################################################################
run_slurm() {
    # If username is empty, stop
    if [ -z "$user" ]; then
        fail "Slurm flag set, but user argument not set. Stopping..."
    fi

    # If username doesn't pass the validity check (RegEx pattern), stop
    if [[ ! "$user" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]]; then
        fail "Username doesn't respect the valid pattern. Stopping..."
    fi

    # If the SSH key is invalid (not on the specified list), stop
    if ! contains "$key" "${keys[@]}"; then
        fail "Invalid SSH key type. Stopping..."
    fi

    # Prepare ~/.ssh/config (private to the owner, as ssh requires)
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    touch ~/.ssh/config
    chmod 600 ~/.ssh/config

    # Add the HPC host entry as per the HPC tutorial, unless already present
    if grep -q "^Host hpc.utcn$" ~/.ssh/config; then
        yellow "Host 'hpc.utcn' already exists in ~/.ssh/config. Keeping it (--user/--key ignored)."
    else
        {
            echo ""
            echo "Host hpc.utcn"
            echo "    Hostname 193.226.17.41"
            echo "    Port 51225"
            echo "    ServerAliveInterval 60"
            echo "    User $user"
            echo "    IdentityFile ~/.ssh/id_$key"
            echo "    LocalForward ${UI_PORT} localhost:${UI_PORT}"
            echo "    LocalForward ${API_PORT} localhost:${API_PORT}"
        } >> ~/.ssh/config
        green "✓ Added 'hpc.utcn' to ~/.ssh/config."
    fi

    # Step 1 (non-interactive): clone or update the repo on the cluster
    blue "Updating the repository on the HPC login node..."
    ssh hpc.utcn 'bash -s' <<'EOF'
set -e
if [ ! -d "$HOME/XAI-Playground" ]; then
    git clone https://github.com/tudorpop1904/XAI-Playground.git "$HOME/XAI-Playground"
fi
cd "$HOME/XAI-Playground"
git pull origin main
chmod +x start.sh
EOF

    # Step 2 (interactive, needs a TTY): run start.sh inside tmux. The SSH session stays open,
    # which keeps the LocalForward tunnel alive. Detach with Ctrl+B then D; re-running this
    # script re-attaches to the same session.
    green "🌐 Once started: Streamlit http://localhost:${UI_PORT} | FastAPI http://localhost:${API_PORT}/docs"
    ssh -t hpc.utcn "cd ~/XAI-Playground && tmux new-session -A -s xai 'bash start.sh ${START_ARGS}'"
}


####################################################################################################
# Case 2: Docker (install the engine if needed, build and run the compose stack)
####################################################################################################
run_docker() {
    local distro_ids=""
    if [ -r /etc/os-release ]; then
        distro_ids="$(. /etc/os-release; echo "${ID:-} ${ID_LIKE:-}")"
    fi

    if [[ " $distro_ids " =~ [[:space:]](ubuntu|debian|fedora|centos|rhel)[[:space:]] ]]; then

        green "✓ Distro is valid (One of Ubuntu, Debian, Fedora, CentOS, RHEL)."
        yellow "Checking Docker..."

        if ! command -v docker >/dev/null 2>&1; then
            red "✗ Docker Engine not found. Installing..."
            curl -fsSL https://get.docker.com | sudo sh
            sudo usermod -aG docker "$USER" || true
            green "✓ Docker Engine installed."
        else
            blue "✓ Docker Engine found."
        fi

        # The docker group only applies to new logins, so fall back to sudo if needed
        local docker_cmd="docker"
        if ! docker info >/dev/null 2>&1; then
            docker_cmd="sudo docker"
        fi

        # Build and run app
        $docker_cmd compose -f docker-compose.yml up -d --build --remove-orphans
        green "✓ Containers started. Streamlit: http://localhost:${UI_PORT} | FastAPI: http://localhost:${API_PORT}/docs"

    else
        # Docker not supported. User can opt to build locally.
        local deploy_locally
        read -r -p "$(printf '\e[0;41mERROR! Distro is not supported by Docker engine.\e[0m\nNeeds to be one of: \e[0;32mUbuntu, Debian, Fedora, CentOS, RHEL.\e[0m\n\e[0;33mContinue without Docker? (Default \e[0;32myes\e[0;33m) [y/n]: \e[0m')" deploy_locally

        # Consent granted
        if [[ "$deploy_locally" == "y" || "$deploy_locally" == "Y" || -z "$deploy_locally" ]]; then
            bg_green "Consent granted. Building locally..."
            run_local
        else
            bg_red "Consent for local build denied. Stopping..."
            exit 1
        fi
    fi
}


####################################################################################################
# Case 3: Local (run start.sh directly on this machine)
####################################################################################################
run_local() {
    chmod +x start.sh
    # exec replaces this shell, so Ctrl+C reaches start.sh's own cleanup trap
    # shellcheck disable=SC2086
    exec bash start.sh ${START_ARGS}
}


# Dispatch
if [ "$IS_SLURM" = 1 ]; then
    run_slurm
elif [ "$IS_DOCKER" = 1 ]; then
    run_docker
elif [ "$IS_LOCAL" = 1 ]; then
    run_local
else
    fail "Invalid state."
fi

exit 0