[CmdletBinding()]
param (
    [Alias("q")]
    [switch]$Queues
)

# [CHANGE] Glyphs are built from code points. The `u{...} escape only exists in PowerShell 7+;
# in Windows PowerShell 5.1 it printed literally as "u{1F4C1}".
$Check  = [char]0x2713
$Cross  = [char]0x2717
$Folder = [char]::ConvertFromUtf32(0x1F4C1)
$Laptop = [char]::ConvertFromUtf32(0x1F4BB)

# Text Formatting Functions
function Yellow-Text() { Write-Host "$args" -ForegroundColor Yellow }
function Blue-Text() { Write-Host "$args" -ForegroundColor Blue }
function Green-Text() { Write-Host "$args" -ForegroundColor Green }
function Red-Text() { Write-Host "$args" -ForegroundColor Red }
function Orange-Text() { Write-Host "$args" -ForegroundColor DarkYellow }

# [CHANGE] Ports honor a preset API_PORT / UI_PORT, like start.sh's ${API_PORT:-8000}.
# Previously they were hard-overwritten to 8000/8501. Resolved before 'try' so Cleanup can use them.
$ApiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$UiPort  = if ($env:UI_PORT)  { $env:UI_PORT }  else { "8501" }
$env:API_PORT = $ApiPort
$env:UI_PORT  = $UiPort

$apiProc = $null

function Cleanup() {
    Orange-Text "`n[!] Shutting down. Performing cleanup..."

    # [CHANGE] Kill the API process tree first (uv spawns a child python/uvicorn)...
    if ($apiProc -and -not $apiProc.HasExited) {
        & taskkill /PID $apiProc.Id /T /F 2>&1 | Out-Null
    }

    # [CHANGE] ...then sweep the *configured* ports (was hard-coded 8501/8000), and only
    # listeners, so we never kill a client (e.g. a browser) that merely talks to those ports.
    foreach ($Port in @([int]$UiPort, [int]$ApiPort)) {
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique |
            ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    }
    Orange-Text "Streamlit and FastAPI/Uvicorn stopped. Cleanup complete."
}

# [CHANGE] Work from the script's own folder (was Get-Location, which broke when launched from
# elsewhere). Push/Pop so the caller's shell location is restored afterwards.
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Push-Location $ScriptDir

try {
    $env:PROJECT_ROOT = $ScriptDir

    # Ensure paths where uv installs by default (mirrors start.sh line 4)
    $env:Path = "$HOME\.local\bin;$env:Path"

    # Project root and node name
    Green-Text "$Folder Project root set: $env:PROJECT_ROOT"
    Blue-Text "$Laptop Service active on node: $env:COMPUTERNAME"

    # Check for uv. Fallback: install & refresh shell
    if (Get-Command "uv" -ErrorAction SilentlyContinue) {
        Green-Text "$Check uv found."
    }
    else {
        Red-Text "$Cross uv not found. Installing..."
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$HOME\.local\bin;" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

        # [CHANGE] Fail clearly if the install didn't work, instead of crashing later.
        if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
            Red-Text "ERROR! uv installation failed. Install it manually: https://docs.astral.sh/uv/"
            exit 1
        }
    }

    # Enable or disable RabbitMQ
    if ($Queues) {
        $env:RABBITMQ_ENABLED = "true"
        Yellow-Text "RabbitMQ message queue enabled."
    }
    else {
        $env:RABBITMQ_ENABLED = "false"
    }

    # Create storage directories
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\logs" | Out-Null
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\models" | Out-Null
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\datasets" | Out-Null
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\images" | Out-Null

    # Sync virtual environment
    # [CHANGE] 'uv' instead of 'python -m uv' (that only works if uv was pip-installed, which the
    # installer above does not do), and abort on failure like start.sh's 'set -e'.
    if (Test-Path ".venv") {
        Blue-Text "Virtual environment exists. Syncing dependencies..."
    }
    else {
        Yellow-Text "Virtual environment doesn't exist. Initializing and syncing..."
    }
    uv sync
    if ($LASTEXITCODE -ne 0) {
        Red-Text "ERROR! uv sync failed."
        exit 1
    }

    # Check CUDA acceleration
    # [CHANGE] The old one-liner used \" inside a PowerShell string, which isn't an escape there
    # (it would terminate the string). Rewritten with single quotes only, which survive both
    # PowerShell 5.1 and 7 when passed to a native command.
    Blue-Text "Checking CUDA hardware acceleration..."
    uv run python -c "import torch; c = torch.cuda.is_available(); print('  -> CUDA Available:', c, '| Device:', torch.cuda.get_device_name(0) if c else 'CPU Only')"
    if ($LASTEXITCODE -ne 0) {
        Red-Text "ERROR! CUDA check failed (is torch installed?)."
        exit 1
    }

    # Run API in background
    Blue-Text "Starting FastAPI on port $ApiPort..."
    $apiProc = Start-Process -FilePath "uv" -ArgumentList "run uvicorn api.main:app --host 0.0.0.0 --port $ApiPort" -PassThru -NoNewWindow

    # Wait 2 seconds for API to initialize database and routes
    Start-Sleep -Seconds 2

    # [CHANGE] Bail out if the API died during startup (e.g. port already in use) rather than
    # launching a UI that can't reach it.
    if ($apiProc.HasExited) {
        Red-Text "ERROR! FastAPI exited during startup (code $($apiProc.ExitCode)). Is port $ApiPort already in use?"
        exit 1
    }

    # Run Streamlit in foreground
    $env:API_BASE_URL = "http://localhost:$ApiPort"
    Blue-Text "Starting Streamlit UI on port $UiPort..."
    Yellow-Text "   Streamlit UI: http://localhost:$UiPort"
    Yellow-Text "   FastAPI Docs: http://localhost:$ApiPort/docs"
    uv run streamlit run app.py --server.port $UiPort --server.address 0.0.0.0
}
finally {
    Cleanup
    Pop-Location
}