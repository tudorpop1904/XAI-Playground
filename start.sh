#!/usr/bin/env bash
set -euo pipefail
# Ensure paths where uv installs by default
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# —————— Text formatting functions (printf) —————————
green()  { printf "\e[0;32m%s\e[0m\n" "$*"; }
red()    { printf "\e[0;31m%s\e[0m\n" "$*"; }
yellow() { printf "\e[0;33m%s\e[0m\n" "$*"; }
blue()   { printf "\e[0;34m%s\e[0m\n" "$*"; }

# If -q|--queues exists, enable RabbitMQ. Otherwise, don't
RABBITMQ_ENABLED=false
for arg in "$@"; do
    case "$arg" in
        -q|--queues) RABBITMQ_ENABLED=true ;;
        *) red "ERROR! Unknown flag '$arg'. Only -q|--queues is supported."; exit 1 ;;
    esac
done

export RABBITMQ_ENABLED

# —————— Port configuration ——————
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

# —————— Create storage directories (including logs for sbatch) ——————
mkdir -p storage/models
mkdir -p storage/datasets
mkdir -p storage/logs
mkdir -p storage/images

# —————— Node & CUDA Info ——————
NODE_NAME="${SLURMD_NODENAME:-$(hostname)}"
green "✓ Service active on node: ${NODE_NAME} (Slurm Job ID: ${SLURM_JOB_ID:-local})"

# —————— Install uv if it doesn't exist ——————
if command -v uv >/dev/null 2>&1; then
    green "✓ uv found."
else
    red "✗ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# —————— Sync venv (uv sync creates .venv if missing) ——————
if [ -d ".venv" ]; then
    blue "Virtual environment exists. Syncing dependencies..."
else
    yellow "Virtual environment doesn't exist. Initializing and syncing..."
fi
uv sync

# —————— Check CUDA acceleration ——————
blue "Checking CUDA acceleration..."
uv run python -c "import torch; print(f'  -> CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"

# —————— Cleanup trap (stops Uvicorn on Ctrl+C or exit) ——————
API_PID=""
cleanup() {
    local code=$?
    trap - INT TERM EXIT
    yellow "\nStopping Uvicorn and Streamlit..."
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup INT TERM EXIT

# —————— Display SSH tunnel command for user ——————
blue "─────────────────────────────────────────────────────────────────"
green "🔗 SSH TUNNEL COMMAND FOR LAPTOP (Run in PowerShell/Local Terminal):"
yellow "   ssh -L ${UI_PORT}:localhost:${UI_PORT} -L ${API_PORT}:localhost:${API_PORT} hpc.utcn"
echo ""
green "🌐 BROWSER ACCESS (after opening the tunnel):"
yellow "   Streamlit UI: http://localhost:${UI_PORT}"
yellow "   FastAPI Docs: http://localhost:${API_PORT}/docs"
blue "─────────────────────────────────────────────────────────────────"

# —————— Run API in background ——————
blue "Starting FastAPI on port ${API_PORT}..."
uv run uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

# Wait 2 seconds for API to initialize database and routes
sleep 2

# —————— Run Streamlit in foreground ——————
export API_BASE_URL="http://localhost:${API_PORT}"
blue "Starting Streamlit UI on port ${UI_PORT}..."
uv run streamlit run app.py --server.port "$UI_PORT" --server.address 0.0.0.0
