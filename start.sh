#!/usr/bin/env bash

# Asigură căile unde uv se instalează implicit
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Dezactivează RabbitMQ pe HPC (elimină mesajele repetate de reconectare)
export RABBITMQ_ENABLED=false

# —————— Text formatting functions (printf) —————————
green()  { printf "\e[0;32m%s\e[0m\n" "$*"; }
red()    { printf "\e[0;31m%s\e[0m\n" "$*"; }
yellow() { printf "\e[0;33m%s\e[0m\n" "$*"; }
blue()   { printf "\e[0;34m%s\e[0m\n" "$*"; }

# —————— Port configuration ——————
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

# —————— Slurm Self-Dispatch (Login Node -> Compute Node cu GPU) ——————
# Dacă rulăm pe login1 (fără SLURM_JOB_ID) și comanda srun există, redirecționăm pe nodul cu GPU
if [ -z "$SLURM_JOB_ID" ] && command -v srun >/dev/null 2>&1 && [ "$NO_SLURM" != "1" ]; then
    yellow "─────────────────────────────────────────────────────────────────"
    yellow "ℹ  Rulare detectată pe login node (fără acces direct la GPU)."
    yellow "ℹ  Se solicită alocare Slurm pe partiția GPU (compute1)..."
    yellow "─────────────────────────────────────────────────────────────────"

    SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
    SLURM_GRES="${SLURM_GRES:-gpu:nvidia_h200_2g.35gb:1}"
    SLURM_CPUS="${SLURM_CPUS:-8}"
    SLURM_MEM="${SLURM_MEM:-32G}"
    SLURM_TIME="${SLURM_TIME:-04:00:00}"

    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

    exec srun \
        --partition="$SLURM_PARTITION" \
        --gres="$SLURM_GRES" \
        --cpus-per-task="$SLURM_CPUS" \
        --mem="$SLURM_MEM" \
        --time="$SLURM_TIME" \
        --job-name="xai-playground" \
        bash "$SCRIPT_PATH" "$@"
fi

# —————— Informații Nod & CUDA ——————
NODE_NAME=$(hostname)
green "✓ Serviciu activ pe nodul: ${NODE_NAME} (Slurm Job ID: ${SLURM_JOB_ID:-local})"

# —————— Install uv if it doesn't exist ——————
if command -v uv >/dev/null 2>&1; then
    green "✓ uv găsit."
else
    red "✗ uv nu a fost găsit. Se instalează..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# —————— Sync venv (uv sync creează automat .venv dacă lipsește) ——————
if [ -d ".venv" ]; then
    blue "Virtual environment existent. Sincronizare dependențe..."
else
    yellow "Virtual environment inexistent. Inițializare și sincronizare..."
fi
uv sync

# —————— Verificare accelerare CUDA ——————
blue "Verificare accelerare GPU..."
uv run python -c "import torch; print(f'  -> CUDA Disponibil: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"

# —————— Cleanup trap (oprește Uvicorn la Ctrl+C sau exit) ——————
cleanup() {
    yellow "\nSe opresc serviciile Uvicorn și Streamlit..."
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup INT TERM EXIT

# —————— Afișare comandă tunel SSH pentru utilizator ——————
blue "─────────────────────────────────────────────────────────────────"
green "🔗 COMANDĂ TUNEL SSH PENTRU LAPTOP (Rulează în PowerShell/Terminal local):"
yellow "   ssh -L ${UI_PORT}:${NODE_NAME}:${UI_PORT} -L ${API_PORT}:${NODE_NAME}:${API_PORT} hpc.utcn"
echo ""
green "🌐 ACCES BROWSER (după deschiderea tunelului):"
yellow "   Streamlit UI: http://localhost:${UI_PORT}"
yellow "   FastAPI Docs: http://localhost:${API_PORT}/docs"
blue "─────────────────────────────────────────────────────────────────"

# —————— Run API in background ——————
blue "Pornire FastAPI pe portul ${API_PORT}..."
uv run uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

# Așteaptă 2 secunde ca API-ul să inițializeze baza de date și rutele
sleep 2

# —————— Run Streamlit in foreground ——————
export API_BASE_URL="http://localhost:${API_PORT}"
blue "Pornire Streamlit UI pe portul ${UI_PORT}..."
uv run streamlit run app.py --server.port "$UI_PORT" --server.address 0.0.0.0
