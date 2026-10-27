<#
.SYNOPSIS
    Parametric script to run on local Windows machine for bootstrapping the XAI Playground app.

.DESCRIPTION
    Equivalent of app.sh for PowerShell environments.
    Supports Slurm (HPC cluster via SSH), Docker containers, and native local runs.

.PARAMETER Slurm
    Connect to HPC cluster via SSH and run on the cluster (login node gateway). Short form: -s or --slurm.

.PARAMETER Docker
    Build and run the application inside Docker containers. Short form: -d or --docker.

.PARAMETER Queues
    Enable RabbitMQ message queue. Short form: -q or --queues.

.PARAMETER User
    HPC username (required when -Slurm is specified). Short form: -u or --user=<str>.

.PARAMETER Key
    SSH key type to use. One of ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa. Default: ed25519. Short form: -k or --key=<str>.
#>

[CmdletBinding()]
param (
    [Alias("s")]
    [switch]$Slurm,

    [Alias("d")]
    [switch]$Docker,

    [Alias("q")]
    [switch]$Queues,

    [Alias("u")]
    [string]$User = "",

    [Alias("k")]
    [string]$Key = "ed25519",

    [Alias("h", "?")]
    [switch]$Help,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

# Text Formatting Helpers
function Red-Text()    { Write-Host "$args" -ForegroundColor Red }
function Green-Text()  { Write-Host "$args" -ForegroundColor Green }
function Yellow-Text() { Write-Host "$args" -ForegroundColor Yellow }
function Blue-Text()   { Write-Host "$args" -ForegroundColor Blue }
function BgRed-Text()  { Write-Host "$args" -BackgroundColor DarkRed -ForegroundColor White }
function BgGreen-Text(){ Write-Host "$args" -BackgroundColor DarkGreen -ForegroundColor White }

# Usage Manual
function Show-Usage() {
    Write-Host "========================================================================================"
    Write-Host "Usage for Slurm : .\app.ps1 -Slurm -User <HPC_username> [ -Key <SSH_key_type> ]" -ForegroundColor Green
    Write-Host "                  .\app.ps1 --slurm --user=<HPC_username> [ --key=<SSH_key_type> ]" -ForegroundColor Green
    Write-Host "========================================================================================"
    Write-Host "Usage for Docker: .\app.ps1 -Docker (or .\app.ps1 --docker / -d)" -ForegroundColor Cyan
    Write-Host "========================================================================================"
    Write-Host "Usage for Local : .\app.ps1 [ -Queues ] (or .\app.ps1 [ --queues / -q ])" -ForegroundColor Yellow
    Write-Host "========================================================================================`n"
    Write-Host "Note: Runs with both Docker and Slurm flags are not supported." -ForegroundColor Magenta
    Write-Host "Valid SSH key types: ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa (default: ed25519)."
}

function Fail-Exit([string]$Message) {
    Show-Usage
    Write-Host ""
    Red-Text "ERROR! $Message"
    exit 1
}

if ($Help) {
    Show-Usage
    exit 0
}

# Parse raw arguments if invoked with Unix-style GNU flags (e.g. --slurm, --user=tspop, -u tspop)
$allArgs = @()
if ($RemainingArgs) { $allArgs += $RemainingArgs }
if ($args) { $allArgs += $args }

$i = 0
while ($i -lt $allArgs.Count) {
    $arg = $allArgs[$i]
    if ($arg -in "-s", "--slurm") {
        $Slurm = $true
    }
    elseif ($arg -in "-d", "--docker") {
        $Docker = $true
    }
    elseif ($arg -in "-q", "--queues") {
        $Queues = $true
    }
    elseif ($arg -in "-h", "--help", "-?") {
        Show-Usage
        exit 0
    }
    elseif ($arg -like "--user=*") {
        $User = $arg.Substring(7)
    }
    elseif ($arg -eq "-u" -and ($i + 1) -lt $allArgs.Count) {
        $i++
        $User = $allArgs[$i]
    }
    elseif ($arg -like "--key=*") {
        $Key = $arg.Substring(6)
    }
    # [CHANGE] Also accept the space-separated GNU form (--user tspop / --key rsa), as app.cmd does.
    # Needed because CMD splits unquoted '=' and users may type either form.
    elseif ($arg -eq "--user" -and ($i + 1) -lt $allArgs.Count) {
        $i++
        $User = $allArgs[$i]
    }
    elseif ($arg -eq "--key" -and ($i + 1) -lt $allArgs.Count) {
        $i++
        $Key = $allArgs[$i]
    }
    elseif ($arg -eq "-k" -and ($i + 1) -lt $allArgs.Count) {
        $i++
        $Key = $allArgs[$i]
    }
    else {
        Fail-Exit "Unknown argument: '$arg'."
    }
    $i++
}

# Port configuration
$ApiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$UiPort  = if ($env:UI_PORT)  { $env:UI_PORT }  else { "8501" }

# Mutual exclusion check
if ($Slurm -and $Docker) {
    Fail-Exit "--slurm and --docker are mutually exclusive."
}

# Supported SSH key list
$ValidKeys = @("ecdsa", "ecdsa-sk", "ed25519", "ed25519-sk", "rsa")

# Forwarded start arguments
$StartArgs = if ($Queues) { "-q" } else { "" }

# Glyphs built from code points: the `u{...} escape only exists in PowerShell 7+
$Check = [char]0x2713

# Run relative to the script folder (start.ps1 / docker-compose.yml use relative paths)
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
# [CHANGE] Push-Location (popped in the 'finally' at the bottom) instead of Set-Location, which
# would leave the user's shell stranded in this folder after the script ends.
Push-Location $ScriptDir

####################################################################################################
# Case 1: Slurm (HPC Cluster via SSH)
####################################################################################################
function Run-Slurm() {
    if ([string]::IsNullOrWhiteSpace($User)) {
        Fail-Exit "Slurm flag set, but user parameter is missing. Specify -User <HPC_username>."
    }

    if ($User -notmatch "^[a-zA-Z_][a-zA-Z0-9_-]*$") {
        Fail-Exit "Username '$User' is invalid. Must start with a letter/underscore and contain only alphanumeric, '_', or '-'."
    }

    if ($Key -notin $ValidKeys) {
        Fail-Exit "Invalid SSH key type '$Key'. Must be one of: $($ValidKeys -join ', ')."
    }

    # Ensure ~/.ssh folder exists
    $SshDir = Join-Path $HOME ".ssh"
    $SshConfig = Join-Path $SshDir "config"

    if (-not (Test-Path $SshDir)) {
        New-Item -ItemType Directory -Path $SshDir -Force | Out-Null
    }
    if (-not (Test-Path $SshConfig)) {
        New-Item -ItemType File -Path $SshConfig -Force | Out-Null
    }

    # Check if Host hpc.utcn is already configured
    $ConfigContent = Get-Content $SshConfig -Raw -ErrorAction SilentlyContinue
    if ($ConfigContent -match "(?m)^Host\s+hpc\.utcn\b") {
        Yellow-Text "Host 'hpc.utcn' already exists in ~/.ssh/config. Keeping existing config (--user/--key ignored)."
    }
    else {
        $ConfigBlock = @"

Host hpc.utcn
    Hostname 193.226.17.41
    Port 51225
    ServerAliveInterval 60
    User $User
    IdentityFile ~/.ssh/id_$Key
    LocalForward ${UiPort} localhost:${UiPort}
    LocalForward ${ApiPort} localhost:${ApiPort}
"@
        Add-Content -Path $SshConfig -Value $ConfigBlock
        Green-Text "$Check Added 'hpc.utcn' entry to ~/.ssh/config."
    }

    if (-not (Get-Command "ssh" -ErrorAction SilentlyContinue)) {
        Fail-Exit "ssh not found. Enable the Windows OpenSSH Client feature."
    }

    # Step 1: Update repository on the HPC login node
    Blue-Text "Updating repository on the HPC cluster..."
    # [CHANGE] Sent as a single-line command ARGUMENT, not piped via stdin. Piping appends a CRLF
    # that bash reads as part of the last line ('chmod +x start.sh\r' -> file not found), which
    # made this step fail every time. No double quotes on purpose: PowerShell 5.1 strips them when
    # passing arguments to native executables; '~' is expanded by the remote shell instead.
    $RemoteSetup = 'set -e; [ -d ~/XAI-Playground ] || git clone https://github.com/tudorpop1904/XAI-Playground.git ~/XAI-Playground; cd ~/XAI-Playground; git pull origin main; chmod +x start.sh'
    ssh hpc.utcn $RemoteSetup
    if ($LASTEXITCODE -ne 0) {
        Red-Text "ERROR! Failed to connect to or update the repo on the HPC cluster."
        exit 1
    }

    # Step 2: Launch attached tmux session with start.sh
    Green-Text "Once started, open in your browser:"
    Yellow-Text "   Streamlit UI: http://localhost:${UiPort}"
    Yellow-Text "   FastAPI Docs: http://localhost:${ApiPort}/docs"
    Write-Host ""

    ssh -t hpc.utcn "cd ~/XAI-Playground && tmux new-session -A -s xai 'bash start.sh $StartArgs'"
}

####################################################################################################
# Case 2: Docker
####################################################################################################
function Run-Docker() {
    Yellow-Text "Checking Docker availability..."
    $DockerCmd = Get-Command "docker" -ErrorAction SilentlyContinue

    $DockerRunning = $false
    if ($DockerCmd) {
        docker info 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $DockerRunning = $true
        }
    }

    if ($DockerRunning) {
        Green-Text "$Check Docker Engine is running."
        Blue-Text "Building and launching containers..."
        docker compose -f docker-compose.yml up -d --build --remove-orphans
        if ($LASTEXITCODE -ne 0) {
            Red-Text "ERROR! docker compose failed."
            exit 1
        }

        Green-Text "$Check Containers started."
        Yellow-Text "Streamlit UI: http://localhost:${UiPort}"
        Yellow-Text "FastAPI Docs: http://localhost:${ApiPort}/docs"
    }
    else {
        BgRed-Text "ERROR! Docker is not running or not installed."
        $Choice = Read-Host "Continue by launching locally without Docker? (Default: yes) [y/n]"
        if ([string]::IsNullOrWhiteSpace($Choice) -or $Choice -eq "y" -or $Choice -eq "Y") {
            BgGreen-Text "Consent granted. Launching locally..."
            Run-Local
        }
        else {
            BgRed-Text "Consent denied. Exiting."
            exit 1
        }
    }
}

####################################################################################################
# Case 3: Local Execution
####################################################################################################
function Run-Local() {
    $ScriptPath = Join-Path $ScriptDir "start.ps1"
    if (-not (Test-Path $ScriptPath)) {
        Fail-Exit "start.ps1 not found at $ScriptPath"
    }

    if ($Queues) {
        & $ScriptPath -Queues
    }
    else {
        & $ScriptPath
    }
}

# Dispatch
try {
    if ($Slurm) {
        Run-Slurm
    }
    elseif ($Docker) {
        Run-Docker
    }
    else {
        Run-Local
    }
}
finally {
    # [CHANGE] Restore the caller's working directory (see Push-Location above)
    Pop-Location
}
