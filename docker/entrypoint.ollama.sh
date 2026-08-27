#!/bin/bash
# ============================================================
# entrypoint.ollama.sh — Ollama Container Bootstrap
# ============================================================
# Starts the Ollama server, waits for it to be ready,
# then pulls the required LLM model if not already cached.
# ============================================================

set -e

echo "🧠 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to become responsive
echo "⏳ Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434 > /dev/null 2>&1; do
    sleep 2
done
echo "✅ Ollama server is ready."

# Pull the model if not already present
MODEL="llama3.1:8b-instruct-q4_K_M"
echo "📦 Ensuring model '$MODEL' is available..."
ollama pull "$MODEL"
echo "✅ Model '$MODEL' is ready."

# Keep the server running in the foreground
wait $OLLAMA_PID
