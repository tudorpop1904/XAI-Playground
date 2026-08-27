#!/bin/bash

echo -e "\033[1;36m🚀 Initializing Playground App Zero-Touch Automation...\033[0m"

# 1. Activate VENV
if [ ! -d ".venv" ]; then
    echo -e "\033[1;33mCreating virtual environment...\033[0m"
    python -m venv .venv
fi
source .venv/bin/activate

# 2. Download dependencies
echo -e "\033[1;33mInstalling dependencies from requirements.txt...\033[0m"
pip install --no-cache-dir -r requirements.txt > /dev/null

# 3. Start FastAPI Server in background
echo -e "\033[1;33mStarting FastAPI Backend on port 8000...\033[0m"
uvicorn api.main:app --port 8000 > /dev/null 2>&1 &

# 4. Pull Ollama models and run server
echo -e "\033[1;33mStarting Ollama Server...\033[0m"
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve > /dev/null 2>&1 &

# 5. Wait for Ollama and FastAPI to be ready
echo -e "\033[1;33mWaiting for backend services...\033[0m"
while ! nc -z localhost 11434 || ! nc -z localhost 8000; do
    sleep 2
done

echo -e "\033[1;32m✅ All systems GO! Launching Streamlit Dashboard...\033[0m"

# 6. Play Notification Sound (Terminal Bell)
echo -ne '\007'
sleep 0.2
echo -ne '\007'

# 7. Run the application
streamlit run app.py
