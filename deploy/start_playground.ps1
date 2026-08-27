<#
.SYNOPSIS
Zero-touch automation script for the Playground App on Windows.
#>

Write-Host "🚀 Initializing Playground App Zero-Touch Automation..." -ForegroundColor Cyan

# 1. Create and Activate Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}
Write-Host "Activating virtual environment..." -ForegroundColor Green
. .\.venv\Scripts\Activate.ps1

# 2. Install Dependencies
Write-Host "Installing dependencies from requirements-api.txt and requirements-ui.txt..." -ForegroundColor Yellow
pip install --no-cache-dir -r requirements-api.txt -r requirements-ui.txt | Out-Null

# 3. Start FastAPI Server in background
Write-Host "Starting FastAPI Backend on port 8000..." -ForegroundColor Yellow
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "api.main:app --port 8000"

# 4. Pull and Start Ollama Model in background
Write-Host "Starting Ollama Server..." -ForegroundColor Yellow
Start-Process -NoNewWindow -FilePath "ollama" -ArgumentList "serve"
Write-Host "Pulling LLM model..." -ForegroundColor Blue
# If ollama is running it might fail to serve again, but pull ensures we have it
Start-Process -NoNewWindow -FilePath "ollama" -ArgumentList "pull llama3.1:8b-instruct-q4_K_M" -Wait

$model_ready = $false
while (-not $model_ready) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434" -Method Head -ErrorAction Stop
        if ($response.StatusCode -eq 200) { $model_ready = $true }
    } catch {
        Start-Sleep -Seconds 2
    }
}
Start-Process -NoNewWindow -FilePath "ollama" -ArgumentList "serve"

# 5. Wait for Health Checks
Write-Host "Waiting for FastAPI to be reachable..." -ForegroundColor Yellow
$fastapi_ready = $false
while (-not $fastapi_ready) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/docs" -Method Head -ErrorAction Stop
        if ($response.StatusCode -eq 200) { $fastapi_ready = $true }
    } catch {
        Start-Sleep -Seconds 2
    }
}

Write-Host "Waiting for Ollama to be reachable..." -ForegroundColor Yellow
$ollama_ready = $false
while (-not $ollama_ready) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434" -Method Head -ErrorAction Stop
        if ($response.StatusCode -eq 200) { $ollama_ready = $true }
    } catch {
        Start-Sleep -Seconds 2
    }
}

Write-Host "✅ All systems GO! Launching Streamlit Dashboard..." -ForegroundColor Green


# 7. Play Notification Sound
[System.Console]::Beep(1000, 200)
[System.Console]::Beep(1200, 200)
[System.Console]::Beep(1500, 400)


# 8. Start Streamlit
# Streamlit will automatically open a browser window
streamlit run app.py
