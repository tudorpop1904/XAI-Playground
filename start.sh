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

# —————— Install uv if it doesn't exist ——————
if command -v uv >/dev/null 2>&1; then
    green "✓ uv found."
else
    red "✗ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# —————— Sync venv (uv sync creează automat .venv dacă lipsește) ——————
if [ -d ".venv" ]; then
    blue "Virtual environment exists. Syncing..."
else
    yellow "Virtual environment not found. Initializing and syncing..."
fi
uv sync

# —————— Cleanup trap (oprește Uvicorn la Ctrl+C sau exit) ——————
cleanup() {
    yellow "\nShutting down services..."
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup INT TERM EXIT

# —————— Run API in background ——————
blue "Starting FastAPI on port ${API_PORT}..."
uv run uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

# Așteaptă 2 secunde ca API-ul să inițializeze baza de date și rutele
sleep 2

# —————— Run Streamlit in foreground ——————
export API_BASE_URL="http://localhost:${API_PORT}"
blue "Starting Streamlit UI on port ${UI_PORT} (API linked: ${API_BASE_URL})..."
uv run streamlit run app.py --server.port "$UI_PORT" --server.address 0.0.0.0
