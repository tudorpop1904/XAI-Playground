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

.PARAMETER Help
    Print out the usage manual. Short form: -h or --help.
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

$esc = [char]27

# Text Formatting Helpers
function Write-Formatted() {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $false)]
        [string]$Message,
        
        [Parameter(Mandatory = $false)]
        [string]$Color = "Default",
        
        [Parameter(Mandatory = $false)]
        [string]$Bg = "Default",
        
        [Parameter(Mandatory = $false)]
        [switch]$Bold,
        
        [Parameter(Mandatory = $false)]
        [switch]$Dim,
        
        [Parameter(Mandatory = $false)]
        [switch]$Italic,
        
        [Parameter(Mandatory = $false)]
        [switch]$Underline,

        [Parameter(Mandatory = $false)]
        [switch]$Blink,

        [Parameter(Mandatory = $false)]
        [switch]$Invert,

        [Parameter(Mandatory = $false)]
        [switch]$Hidden,

        [Parameter(Mandatory = $false)]
        [switch]$Strike
    )

    $Format = "$esc["
    
    if ($Bold) { $Format = "${Format};1" }
    if ($Dim) { $Format = "${Format};2" }
    if ($Italic) { $Format = "${Format};3" }
    if ($Underline) { $Format = "${Format};4" }

    if ($Color -eq "Black") { $Format = "${Format};30" }
    elseif ($Color -eq "Red") { $Format = "${Format};31" }
    elseif ($Color -eq "Green") { $Format = "${Format};32" }
    elseif ($Color -eq "Yellow") { $Format = "${Format};33" }
    elseif ($Color -eq "Blue") { $Format = "${Format};34" }
    elseif ($Color -eq "Magenta") { $Format = "${Format};35" }
    elseif ($Color -eq "Cyan") { $Format = "${Format};36" }
    elseif ($Color -eq "White") { $Format = "${Format};37" }
    elseif ($Color -eq "Gray") { $Format = "${Format};90" }

    if ($Bg -eq "Black") { $Format = "${Format};40" }
    elseif ($Bg -eq "Red") { $Format = "${Format};41" }
    elseif ($Bg -eq "Green") { $Format = "${Format};42" }
    elseif ($Bg -eq "Yellow") { $Format = "${Format};43" }
    elseif ($Bg -eq "Blue") { $Format = "${Format};44" }
    elseif ($Bg -eq "Magenta") { $Format = "${Format};45" }
    elseif ($Bg -eq "Cyan") { $Format = "${Format};46" }
    elseif ($Bg -eq "White") { $Format = "${Format};47" }
    elseif ($Bg -eq "Gray") { $Format = "${Format};100" }

    if ($Blink) { $Format = "${Format};5" }
    if ($Invert) { $Format = "${Format};7" }
    if ($Hidden) { $Format = "${Format};8" }
    if ($Strike) { $Format = "${Format};9" }

    $Format = "${Format}m"
    Write-Host "$Format$Message$esc[0m"
}

# Usage Manual
function Show-Usage() {
    Write-Formatted ""
    Write-Formatted "╔══════════════════════════════════════════════════════════════════════════════════════════════╗" -Bold -Color Black -Bg White
    Write-Formatted "║                                                                                              ║" -Bold -Color Black -Bg White
    Write-Formatted "║                    Windows PowerShell bootstrap script for XAI Playground                    ║" -Bold -Color Black -Bg White
    Write-Formatted "║                                         Usage Manual                                         ║" -Bold -Color Black -Bg White
    Write-Formatted "║                                                                                              ║" -Bold -Color Black -Bg White
    Write-Formatted "╚══════════════════════════════════════════════════════════════════════════════════════════════╝" -Bold -Color Black -Bg White
    Read-Host "Press <Enter> to continue..."
    Write-Formatted "┌────────────────────────────────────────────────────────────────────────────┐" -Bold
    Write-Formatted "│                                  General                                   │" -Bold
    Write-Formatted "$esc[1m│$esc[22m                          (Available for all three)                         $esc[1m│$esc[22m"
    Write-Formatted "│ Options:                                                                   │" -Bold
    Write-Formatted "$esc[1m│$esc[22m   -Queues,   --queues,   -q         - Enable RabbitMQ                      $esc[1m│$esc[22m"
    Write-Formatted "$esc[1m│$esc[22m   -Help,     --help,     -h         - Show this help message               $esc[1m│$esc[22m"
    Write-Formatted "$esc[1m│$esc[22m   -?                                - Throw error, show this help message  $esc[1m│$esc[22m"
    Write-Formatted "│                                                                            │" -Bold
    Write-Formatted "$esc[1m│$esc[22m   Examples:                                                                $esc[1m│$esc[22m"
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 [ -Queues ] [ -Help ]                                          $esc[1m│$esc[22m"
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 [ --queues ] [ --help ]                                        $esc[1m│$esc[22m"
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 [ -q ] [ -h ]                                                  $esc[1m│$esc[22m"
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 -? (where ? is not one of the other options -- See below)      $esc[1m│$esc[22m"
    Write-Formatted "└────────────────────────────────────────────────────────────────────────────┘" -Bold
    Read-Host "Press <Enter> to continue..."
    Write-Formatted "┌───────────────────────────────────────────────────────────────────────────────────┐" -Color Blue -Bold
    Write-Formatted "│                                       Slurm                                       │" -Color Blue -Bold
    Write-Formatted "│ Options:                                                                          │" -Color Blue -Bold
    Write-Formatted "$esc[1m│$esc[22m   -Slurm,  --slurm,  -s                 - Run on HPC cluster                      $esc[1m│$esc[22m" -Color Blue
    Write-Formatted "$esc[1m│$esc[22m   -User,   --user,   -u <HPC_username>  - HPC username                            $esc[1m│$esc[22m" -Color Blue
    Write-Formatted "$esc[1m│$esc[22m   -Key,    --key,    -k <SSH_key_type>  - SSH key type (ed25519 or rsa)           $esc[1m│$esc[22m" -Color Blue
    Write-Formatted "│                                                                                   │" -Color Blue -Bold
    Write-Formatted "│ Examples:                                                                         │" -Color Blue -Bold
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 -Slurm -User <HPC_username> [ -Key <SSH_key_type> ] [ -Queues ]       $esc[1m│$esc[22m" -Color Blue
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 --slurm --user=<HPC_username> [ --key=<SSH_key_type> ] [ --queues ]   $esc[1m│$esc[22m" -Color Blue
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 -s -u <HPC_username> [ -k <SSH_key_type> ] [ -q ]                     $esc[1m│$esc[22m" -Color Blue
    Write-Formatted "│                                                                                   │" -Color Blue -Bold
    Write-Formatted "│ Runs the app via an SSH session in the university's HPC cluster using Slurm.      │" -Color Blue -Bold
    Write-Formatted "└───────────────────────────────────────────────────────────────────────────────────┘" -Color Blue -Bold
    Read-Host "Press <Enter> to continue..."
    Write-Formatted "┌────────────────────────────────────────────────────────────────────┐" -Color Cyan -Bold
    Write-Formatted "│                               Docker                               │" -Color Cyan -Bold
    Write-Formatted "│ Options:                                                           │" -Color Cyan -Bold
    Write-Formatted "$esc[1m│$esc[22m   -Docker, --docker, -d                                            $esc[1m│$esc[22m" -Color Cyan
    Write-Formatted "│                                                                    │" -Color Cyan -Bold
    Write-Formatted "│ Examples:                                                          │" -Color Cyan -Bold
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 -Docker [ -Queues ]                                    $esc[1m│$esc[22m" -Color Cyan
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 --docker [ --queues ]                                  $esc[1m│$esc[22m" -Color Cyan
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 -d [ -q ]                                              $esc[1m│$esc[22m" -Color Cyan
    Write-Formatted "│                                                                    │" -Color Cyan -Bold
    Write-Formatted "│ Builds the Docker image locally and runs the app in a container.   │" -Color Cyan -Bold
    Write-Formatted "└────────────────────────────────────────────────────────────────────┘" -Color Cyan -Bold
    Read-Host "Press <Enter> to continue..."
    Write-Formatted "┌─────────────────────────────────────────────────────────────────────────┐" -Color Yellow -Bold
    Write-Formatted "│                                  Local                                  │" -Color Yellow -Bold
    Write-Formatted "│ Options:                                                                │" -Color Yellow -Bold
    Write-Formatted "$esc[1m│$esc[22m   ONLY -Queues, --queues, or -q                                         $esc[1m│$esc[22m" -Color Yellow
    Write-Formatted "$esc[1m│$esc[22m   NO -Slurm, --slurm, -s, -Docker, --docker, or -d                      $esc[1m│$esc[22m" -Color Yellow
    Write-Formatted "│                                                                         │" -Color Yellow -Bold
    Write-Formatted "│ Examples:                                                               │" -Color Yellow -Bold
    Write-Formatted "$esc[1m│$esc[22m   .\app.ps1 [ -Queues ] (or .\app.ps1 [ --queues / -q ])                $esc[1m│$esc[22m" -Color Yellow
    Write-Formatted "│       (Default mode, runs locally without Docker or Slurm)              │" -Color Yellow -Bold
    Write-Formatted "└─────────────────────────────────────────────────────────────────────────┘" -Color Yellow -Bold
    Read-Host "Press <Enter> to continue..."
    Write-Formatted "Note: Runs with both Docker and Slurm flags are not supported." -Color Magenta
    Write-Formatted ""
    Write-Formatted "Valid SSH key types: ecdsa, ecdsa-sk, ed25519, ed25519-sk, rsa (default: ed25519)." -Color Cyan
    Write-Formatted ""
    Write-Formatted "Set XAI_MUTE=1 to silence the audio cues." -Dim
    Write-Formatted ""
}

# Glyphs built from code points: the `u{...} escape only exists in PowerShell 7+
$Check = [char]0x2713
$Cross = [char]0x2716
$Folder = [char]::ConvertFromUtf32(0x1F4C1)
$Laptop = [char]::ConvertFromUtf32(0x1F4BB)

# This script lives in <root>\scripts\bootstrap; everything else (docker-compose.yml, storage\, app.py, .venv)
# is relative to the project root.
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))

# Audio cue (never throws, never blocks; see audio\play.ps1). -Detach right before the process exits.
function Cue([string]$Name, [switch]$Detach) {
    try { & (Join-Path $ScriptDir "audio\play.ps1") $Name -Detach:$Detach } catch { }
}

# Fatal error. -Hint adds the "run -Help" pointer (use it for argument errors).
function Exit-Fail([string]$Message, [switch]$Hint) {
    Cue "error" -Detach
    Write-Formatted ""
    Write-Formatted "$Cross ERROR! $Message" -Color Red -Bold
    if ($Hint) {
        Write-Formatted "Refer to the user manual by running: $esc[1;33m.\app.ps1 -Help$esc[0m" -Color Red
    }
    Write-Formatted ""
    exit 1
}

# Ask for consent. Without a terminal the default answer is used, so unattended runs never hang.
function Confirm-Action([string]$Prompt, [bool]$DefaultYes = $true) {
    $hint = if ($DefaultYes) { "Y/n" } else { "y/N" }
    if ([Console]::IsInputRedirected) {
        Write-Formatted "$Prompt [$hint] (no terminal: using default)" -Color Yellow
        return $DefaultYes
    }
    $reply = Read-Host "$esc[33m$Prompt $esc[2m[$hint]$esc[0m"
    if ([string]::IsNullOrWhiteSpace($reply)) { return $DefaultYes }
    return ($reply.Trim() -match '^(y|yes)$')
}

# Service card (61 columns wide; the globe emoji is 2 columns wide, so only the title is padded)
function Show-Banner([string]$Title) {
    Write-Formatted "┌─────────────────────────────────────────────────────────────┐" -Color Blue -Bold
    Write-Formatted ("│  🌐  " + $Title.PadRight(55) + "│") -Color Green -Bold
    Write-Formatted "├─────────────────────────────────────────────────────────────┤" -Color Blue -Bold
    Write-Formatted ("│  • Streamlit UI :  " + "http://localhost:$UiPort".PadRight(41) + "│") -Color Green
    Write-Formatted ("│  • FastAPI Docs :  " + "http://localhost:$ApiPort/docs".PadRight(41) + "│") -Color Green
    Write-Formatted "└─────────────────────────────────────────────────────────────┘" -Color Blue -Bold
    Write-Formatted ""
}

if ($Help) {
    Cue "info"
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
    $hasNext = ($i + 1) -lt $allArgs.Count
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
        Cue "info"
        Show-Usage
        exit 0
    }
    elseif ($arg -like "--user=*") {
        $User = $arg.Substring(7)
    }
    elseif ($arg -like "--key=*") {
        $Key = $arg.Substring(6)
    }
    # Also accept the space-separated GNU form (--user tspop / --key rsa), as app.cmd and app.sh do.
    elseif ($arg -in "-u", "--user", "-k", "--key") {
        if (-not $hasNext) { Exit-Fail "Missing value for '$arg'." -Hint }
        $i++
        if ($arg -in "-u", "--user") { $User = $allArgs[$i] } else { $Key = $allArgs[$i] }
    }
    else {
        Exit-Fail "Unknown argument: '$arg'." -Hint
    }
    $i++
}

# Port configuration
$ApiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$UiPort = if ($env:UI_PORT) { $env:UI_PORT }  else { "8501" }
$env:API_PORT = $ApiPort
$env:UI_PORT = $UiPort

# Mutual exclusion check
if ($Slurm -and $Docker) {
    Exit-Fail "--slurm and --docker are mutually exclusive." -Hint
}

# Supported SSH key list
$ValidKeys = @("ecdsa", "ecdsa-sk", "ed25519", "ed25519-sk", "rsa")

# Forwarded start arguments
$StartArgs = if ($Queues) { "-q" } else { "" }

$apiProc = $null
$localStarted = $false

function Cleanup() {
    if (-not $localStarted) { return }
    Write-Formatted "`n[!] Shutting down. Performing cleanup..." -Color Yellow

    # Kill the API process tree first (uv spawns child python/uvicorn)
    if ($apiProc -and -not $apiProc.HasExited) {
        & taskkill /PID $apiProc.Id /T /F 2>&1 | Out-Null
    }

    # Sweep the configured ports. Safe: Confirm-PortFree guaranteed they were free (or freed with
    # consent) at startup, so any listener now is ours.
    foreach ($Port in @([int]$UiPort, [int]$ApiPort)) {
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    }
    Write-Formatted "Streamlit and FastAPI/Uvicorn stopped. Cleanup complete." -Color Yellow
    Cue "shutdown" -Detach
}

# —————— Ports ——————
function Get-PortOwners([int]$Port) {
    try {
        @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique)
    }
    catch { @() }
}

# Never starts on top of something else, and never kills a process without asking
function Confirm-PortFree([int]$Port, [string]$EnvName) {
    $owners = @(Get-PortOwners $Port)
    if ($owners.Count -eq 0) { return }
    Cue "warning"
    Write-Formatted "⚠ WARNING! Port $Port ($EnvName) is already in use by:" -Color Yellow -Bold
    foreach ($ownerPid in $owners) {
        $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
        $procName = if ($proc) { $proc.ProcessName } else { "unknown" }
        Write-Formatted "    PID $ownerPid ($procName)" -Color Yellow
    }
    if (-not (Confirm-Action "Stop it so the app can use port ${Port}?" $false)) {
        Exit-Fail "Port $Port is busy. Free it, or set the $EnvName environment variable to another port."
    }
    foreach ($ownerPid in $owners) { Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
    if (@(Get-PortOwners $Port).Count -gt 0) {
        Exit-Fail "Could not free port $Port (try an elevated terminal)."
    }
    Write-Formatted "$Check Port $Port freed." -Color Green
}

# —————— uv ——————
function Confirm-Uv() {
    if (Get-Command "uv" -ErrorAction SilentlyContinue) {
        Write-Formatted "$Check uv found." -Color Green
        return
    }
    Cue "warning"
    Write-Formatted "⚠ WARNING! uv (Python package manager) was not found." -Color Yellow -Bold
    if (-not (Confirm-Action "Install it from astral.sh now? (user-level, no administrator rights)" $true)) {
        Exit-Fail "uv is required to run locally. Install it manually: https://docs.astral.sh/uv/"
    }
    Write-Formatted "Installing uv..." -Color Blue -Italic
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$HOME\.local\bin;" + [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
        Exit-Fail "uv installation failed. Install it manually: https://docs.astral.sh/uv/"
    }
    Write-Formatted "$Check uv installed." -Color Green
}

####################################################################################################
# Case 1: Slurm (HPC Cluster via SSH)
####################################################################################################
function Start-Slurm() {
    if ([string]::IsNullOrWhiteSpace($User)) {
        Exit-Fail "Slurm flag set, but user argument not set. Specify -User <HPC_username>." -Hint
    }

    if ($User -notmatch "^[a-zA-Z_][a-zA-Z0-9_-]*$") {
        Exit-Fail "Username '$User' is invalid. Must start with a letter/underscore and contain only alphanumeric, '_', or '-'." -Hint
    }

    if ($Key -notin $ValidKeys) {
        Exit-Fail "Invalid SSH key type '$Key'. Must be one of: $($ValidKeys -join ', ')." -Hint
    }

    if (-not (Get-Command "ssh" -ErrorAction SilentlyContinue)) {
        Exit-Fail "Command 'ssh' not found. Enable the Windows OpenSSH Client (Settings > System > Optional features)."
    }

    $SshDir = Join-Path $HOME ".ssh"
    $SshConfig = Join-Path $SshDir "config"

    # Add the HPC host entry as per the HPC tutorial, unless already present
    $ConfigContent = if (Test-Path $SshConfig) { Get-Content $SshConfig -Raw -ErrorAction SilentlyContinue } else { "" }
    if ($ConfigContent -match "(?m)^\s*Host\s+hpc\.utcn\b") {
        Write-Formatted "Host 'hpc.utcn' already exists in ~/.ssh/config. Keeping existing config (--user/--key ignored)." -Color Yellow
    }
    else {
        Write-Formatted "The 'hpc.utcn' SSH host is not configured yet." -Color Yellow
        if (-not (Confirm-Action "Add it to ~/.ssh/config (User $User, key ~/.ssh/id_$Key, forwards $UiPort/$ApiPort)?" $true)) {
            Exit-Fail "Consent denied. The 'hpc.utcn' entry is required to connect."
        }
        if (-not (Test-Path (Join-Path $SshDir "id_$Key"))) {
            Write-Formatted "⚠ SSH key ~/.ssh/id_$Key not found. Create one with 'ssh-keygen -t $Key' and register the public key on the cluster." -Color Yellow
        }
        if (-not (Test-Path $SshDir)) { New-Item -ItemType Directory -Path $SshDir -Force | Out-Null }
        if (-not (Test-Path $SshConfig)) { New-Item -ItemType File -Path $SshConfig -Force | Out-Null }
        $ConfigBlock = @"

Host hpc.utcn
    Hostname 193.226.17.41
    Port 51225
    ServerAliveInterval 60
    User $User
    IdentityFile ~/.ssh/id_$Key
    LocalForward ${UiPort} 127.0.0.1:${UiPort}
    LocalForward ${ApiPort} 127.0.0.1:${ApiPort}
"@
        Add-Content -Path $SshConfig -Value $ConfigBlock
        Write-Formatted "$Check Added 'hpc.utcn' entry to ~/.ssh/config." -Color Green
    }

    # Launch attached tmux session with app.sh
    Write-Formatted "⚡ Instantiating HPC cluster session..." -Color Blue -Bold
    Show-Banner "XAI Playground Remote Forwarding Configured"

    $RemoteSetup = @"
set -euo pipefail
[ -d ~/XAI-Playground ] || git clone https://github.com/tudorpop1904/XAI-Playground.git ~/XAI-Playground
cd ~/XAI-Playground
git merge --abort 2>/dev/null || true
git fetch origin main 2>/dev/null || true
git reset --hard origin/main 2>/dev/null || true
chmod +x scripts/bootstrap/app.sh
tmux new-session -A -s xai 'bash scripts/bootstrap/app.sh $StartArgs; exec bash'
"@
    $RemoteSetup = $RemoteSetup -replace "`r", ""

    ssh -t hpc.utcn $RemoteSetup
    if ($LASTEXITCODE -ne 0) {
        Exit-Fail "Remote cluster session exited with code $LASTEXITCODE."
    }
    Write-Formatted "Remote session closed." -Color Yellow
    Cue "shutdown" -Detach
}

####################################################################################################
# Case 2: Docker
####################################################################################################
function Test-DockerReady() {
    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) { return $false }
    docker info 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Wait-Docker([int]$TimeoutSec = 180) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $spin = '|', '/', '-', '\'
    $n = 0
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if (Test-DockerReady) { Write-Host -NoNewline "`r$esc[K"; return $true }
        Write-Host -NoNewline "`r$esc[34m$($spin[$n % 4]) Waiting for the Docker daemon... $([int]$sw.Elapsed.TotalSeconds)s$esc[0m"
        $n++
        Start-Sleep -Milliseconds 500
    }
    Write-Host -NoNewline "`r$esc[K"
    return $false
}

function Start-DockerDaemon() {
    $desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $desktop) {
        Start-Process -FilePath $desktop
    }
    else {
        try { Start-Service -Name docker -ErrorAction Stop } catch { return $false }
    }
    return (Wait-Docker)
}

# Equivalent of `curl -fsSL https://get.docker.com | sh` for Windows: download the official installer
function Install-DockerDesktop() {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "amd64" }
    $url = "https://desktop.docker.com/win/main/$arch/Docker%20Desktop%20Installer.exe"
    $tmp = Join-Path $env:TEMP "DockerDesktopInstaller.exe"

    Write-Formatted "Downloading Docker Desktop (~600 MB, no progress bar)..." -Color Blue -Italic
    $oldProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"   # the progress bar makes Invoke-WebRequest ~10x slower
    try { Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing }
    catch { return $false }
    finally { $ProgressPreference = $oldProgress }

    Write-Formatted "Running the installer (approve the administrator prompt)..." -Color Blue -Italic
    try {
        $installer = Start-Process -FilePath $tmp -ArgumentList "install", "--quiet", "--accept-license" -Verb RunAs -Wait -PassThru
    }
    catch { return $false }
    finally { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }

    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    return ($installer.ExitCode -eq 0 -and [bool](Get-Command "docker" -ErrorAction SilentlyContinue))
}

function Start-Docker() {
    Write-Formatted "Checking Docker availability..." -Color Yellow -Bold
    $ready = $false

    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        Cue "warning"
        Write-Formatted "⚠ WARNING! Docker Engine not found." -Color Yellow -Bold
        if (Confirm-Action "Download and install Docker Desktop now? (administrator approval needed; a restart may follow)" $false) {
            if (Install-DockerDesktop) {
                Write-Formatted "$Check Docker Desktop installed." -Color Green
            }
            else {
                Write-Formatted "⚠ The Docker installation did not complete (a restart may be required)." -Color Yellow
            }
        }
        else {
            Write-Formatted "Consent denied. Skipping the Docker installation." -Color Yellow
        }
    }

    if (Get-Command "docker" -ErrorAction SilentlyContinue) {
        if (Test-DockerReady) {
            $ready = $true
        }
        else {
            Cue "warning"
            Write-Formatted "⚠ WARNING! Docker daemon is not running." -Color Yellow -Bold
            if (Confirm-Action "Start it now?" $true) {
                Write-Formatted "Consent granted. Starting Docker..." -Color Green -Bold
                $ready = Start-DockerDaemon
                if (-not $ready) { Write-Formatted "⚠ Docker did not become ready in time." -Color Yellow }
            }
            else {
                Write-Formatted "Consent denied. Not starting the daemon." -Color Yellow
            }
        }
    }

    if ($ready) {
        $useV2 = $true
        docker compose version 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            if (Get-Command "docker-compose" -ErrorAction SilentlyContinue) { $useV2 = $false }
            else { Exit-Fail "Docker Compose not found. Install it: https://docs.docker.com/compose/install/" }
        }
        Write-Formatted "$Check Docker Engine is running." -Color Green -Bold
        Write-Formatted "Building and launching containers..." -Color Blue -Bold -Italic
        if ($useV2) { docker compose -f docker-compose.yml up -d --build --remove-orphans }
        else { docker-compose -f docker-compose.yml up -d --build --remove-orphans }
        if ($LASTEXITCODE -ne 0) {
            Exit-Fail "Docker Compose failed."
        }
        Cue "startup"
        Write-Formatted "$Check Containers started." -Color Green -Bold
        Show-Banner "XAI Playground Containers Ready"
    }
    else {
        Cue "warning"
        Write-Formatted "⚠ WARNING! Docker is not available." -Color Yellow -Bold
        if (Confirm-Action "Continue by launching locally without Docker?" $true) {
            Write-Formatted "Consent granted. Launching locally..." -Color Green -Bold
            Start-Local
        }
        else {
            Exit-Fail "Consent denied. Exiting."
        }
    }
}

####################################################################################################
# Case 3: Local Execution (formerly start.ps1)
####################################################################################################
function Start-Local() {
    $script:localStarted = $true
    $env:PROJECT_ROOT = $ProjectRoot

    # Ensure paths where uv installs by default
    $env:Path = "$HOME\.local\bin;$env:Path"

    # Enable or disable RabbitMQ
    if ($Queues) {
        $env:RABBITMQ_ENABLED = "true"
        Write-Formatted "RabbitMQ message queue enabled." -Color Yellow
    }
    else {
        $env:RABBITMQ_ENABLED = "false"
    }

    # Create storage directories
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\logs" | Out-Null
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\models" | Out-Null
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\datasets" | Out-Null
    New-Item -ItemType Directory -Force -Path "$env:PROJECT_ROOT\storage\images" | Out-Null

    # Project root and node name
    Write-Formatted "$Folder Project root set: $env:PROJECT_ROOT" -Color Green
    Write-Formatted "$Laptop Service active on node: $env:COMPUTERNAME" -Color Blue

    Confirm-Uv
    Confirm-PortFree ([int]$ApiPort) "API_PORT"
    Confirm-PortFree ([int]$UiPort) "UI_PORT"

    # Sync virtual environment
    if (Test-Path ".venv") {
        Write-Formatted "Virtual environment exists. Syncing dependencies..." -Color Blue
    }
    else {
        Write-Formatted "Virtual environment doesn't exist. Initializing and syncing..." -Color Yellow
    }
    uv sync
    if ($LASTEXITCODE -ne 0) {
        Exit-Fail "uv sync failed."
    }

    # Check CUDA acceleration
    Write-Formatted "Checking CUDA hardware acceleration..." -Color Blue
    uv run python -c "import torch; c = torch.cuda.is_available(); print('  -> CUDA Available:', c, '| Device:', torch.cuda.get_device_name(0) if c else 'CPU Only')"
    if ($LASTEXITCODE -ne 0) {
        Exit-Fail "CUDA check failed (is torch installed?)."
    }

    # Run API in background
    Write-Formatted "⚡ Starting FastAPI on port $ApiPort..." -Color Blue
    $script:apiProc = Start-Process -FilePath "uv" -ArgumentList "run uvicorn api.main:app --host 0.0.0.0 --port $ApiPort" -PassThru -NoNewWindow
    $null = $script:apiProc.Handle   # caches the handle so ExitCode is readable later

    # Wait (up to 30 s) until FastAPI is actually listening; bail out if it died
    $spin = '|', '/', '-', '\'
    $tries = 0
    while ($tries -lt 60) {
        if ($script:apiProc.HasExited) {
            Write-Host ""
            Exit-Fail "FastAPI exited during startup (code $($script:apiProc.ExitCode)). Check the output above."
        }
        if (@(Get-PortOwners ([int]$ApiPort)).Count -gt 0) { break }
        Write-Host -NoNewline "`r$esc[34m$($spin[$tries % 4]) Waiting for FastAPI...$esc[0m"
        Start-Sleep -Milliseconds 500
        $tries++
    }
    Write-Host -NoNewline "`r$esc[K"
    if (@(Get-PortOwners ([int]$ApiPort)).Count -eq 0) {
        Exit-Fail "FastAPI did not start listening on port $ApiPort within 30 s."
    }

    Cue "startup"
    Show-Banner "XAI Playground Web Services Ready"

    # Run Streamlit in foreground
    $env:API_BASE_URL = "http://localhost:$ApiPort"
    Write-Formatted "🚀 Launching Streamlit UI on port $UiPort..." -Color Blue
    uv run streamlit run app.py --server.port $UiPort --server.address 0.0.0.0
}

# Dispatch
Push-Location $ProjectRoot
try {
    if ($Slurm) {
        Start-Slurm
    }
    elseif ($Docker) {
        Start-Docker
    }
    else {
        Start-Local
    }
}
finally {
    Cleanup
    # Restore the caller's working directory
    Pop-Location
}
