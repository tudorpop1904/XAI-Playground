<#
.SYNOPSIS
    Best-effort audio cue for the XAI Playground bootstrap scripts (pure PowerShell, no other shell).

.DESCRIPTION
    Usage: .\play.ps1 <startup|shutdown|warning|error|info> [-Detach]

    Never throws and never blocks. Use -Detach right before the process exits (fatal error,
    shutdown): the sound then plays from a small helper PowerShell process that outlives this
    one, otherwise it would be cut off.

      Windows (PS 5.1 / 7)   -> System.Media.SoundPlayer
      macOS pwsh             -> afplay
      Linux pwsh             -> paplay | pw-play | aplay | ffplay | mpv | play (if an audio server/card exists)
      SSH session / nothing usable -> terminal bell

    Set XAI_MUTE=1 to silence everything.
#>
param(
    [Parameter(Position = 0)][string]$Name,
    [switch]$Detach
)

$esc = [char]27

function Ring([int]$Count) {
    for ($i = 0; $i -lt $Count; $i++) {
        [Console]::Write([char]7)
        if ($i -lt $Count - 1) { Start-Sleep -Milliseconds 150 }
    }
}

try {
    if (-not $Name -or $env:XAI_MUTE) { return }

    $bells = 1
    if ($Name -eq "error") { $bells = 3 } elseif ($Name -eq "warning") { $bells = 2 }

    $wav = Join-Path (Join-Path $PSScriptRoot "wav") "$Name.wav"
    if (-not (Test-Path -LiteralPath $wav)) {
        Write-Host "$esc[2mAudio file '$Name.wav' not found.$esc[0m"
        return
    }

    # Remote shell: the speakers are on the client, not here -> ring the terminal instead
    if ($env:SSH_CONNECTION -or $env:SSH_CLIENT -or $env:SSH_TTY) { Ring $bells; return }

    $onWindows = ($PSVersionTable.PSVersion.Major -lt 6) -or [bool]$IsWindows
    if ($onWindows) {
        if ($Detach) {
            # Helper process that outlives us. EncodedCommand = no quoting issues with odd paths.
            $cmd = "(New-Object System.Media.SoundPlayer '" + ($wav -replace "'", "''") + "').PlaySync()"
            $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
            Start-Process -FilePath (Get-Process -Id $PID).Path -NoNewWindow `
                -ArgumentList "-NoProfile", "-NonInteractive", "-EncodedCommand", $enc
        }
        else {
            $player = New-Object System.Media.SoundPlayer -ArgumentList $wav
            $global:XaiSoundPlayer = $player   # keep a reference alive while it plays
            $player.Play()
        }
        return
    }

    # macOS / Linux PowerShell: call the native players directly
    if ($IsMacOS) {
        $players = @(, @("afplay"))
    }
    else {
        $uid = (& id -u 2>$null)
        $rt = if ($env:XDG_RUNTIME_DIR) { $env:XDG_RUNTIME_DIR } else { "/run/user/$uid" }
        $hasAudio = $env:PULSE_SERVER -or (Test-Path "$rt/pulse/native") -or (Test-Path "$rt/pipewire-0") -or (Test-Path "/dev/snd")
        if (-not $hasAudio) { Ring $bells; return }   # headless server
        $players = @(
            @("paplay"), @("pw-play"), @("aplay", "-q"),
            @("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
            @("mpv", "--no-video", "--really-quiet"), @("play", "-q")
        )
    }
    foreach ($p in $players) {
        if (Get-Command $p[0] -ErrorAction SilentlyContinue) {
            $args2 = @($p | Select-Object -Skip 1) + $wav
            Start-Process -FilePath $p[0] -ArgumentList $args2 `
                -RedirectStandardOutput "/dev/null" -RedirectStandardError "/dev/null"
            return
        }
    }
    Ring $bells
}
catch {
    [Console]::Write([char]7)
}