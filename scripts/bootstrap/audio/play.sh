#!/usr/bin/env bash
# Best-effort audio cue for the XAI Playground bootstrap scripts (pure Bash, no other shell).
#
#   Run:     bash audio/play.sh <startup|shutdown|warning|error|info>
#   Source:  . audio/play.sh; play_sound <name>
#
# Never fails and never blocks. Resolution order:
#   SSH session (e.g. the HPC tmux)  -> terminal bell (the speakers are on the client, not here)
#   macOS                            -> afplay
#   Linux / WSL with audio           -> paplay | pw-play | aplay | ffplay | mpv | sox play
#   Git Bash / MSYS / Cygwin         -> ffplay | mpv | sox play, if one happens to be installed
#   anything else (headless server)  -> terminal bell (silent when there is no terminal either)
# Set XAI_MUTE=1 to silence everything.

_xai_audio_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Ring the terminal bell N times. Writes to the controlling terminal, so it also works over SSH/tmux.
_xai_bell() {
    local n="${1:-1}" i
    for ((i = 0; i < n; i++)); do
        { printf '\a' >/dev/tty; } 2>/dev/null || return 0
        if [ "$i" -lt $((n - 1)) ]; then sleep 0.15 2>/dev/null || true; fi
    done
    return 0
}

play_sound() {
    local name="${1:-}" wav bells=1 rt has_pulse=0 has_alsa=0
    [ -n "$name" ] || return 0
    [ -z "${XAI_MUTE:-}" ] || return 0
    case "$name" in
        error)   bells=3 ;;
        warning) bells=2 ;;
    esac

    wav="$_xai_audio_dir/wav/$name.wav"
    if [ ! -f "$wav" ]; then
        printf '\e[2mAudio file %s.wav not found.\e[0m\n' "$name"
        return 0
    fi

    # Remote shell: playing here would be pointless -> ring the client's terminal instead
    if [ -n "${SSH_CONNECTION:-}${SSH_CLIENT:-}${SSH_TTY:-}" ]; then
        ( _xai_bell "$bells" ) >/dev/null 2>&1 &
        return 0
    fi

    set --
    case "$(uname -s 2>/dev/null)" in
        Darwin)
            if command -v afplay >/dev/null 2>&1; then set -- afplay "$wav"; fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            has_pulse=1   # nothing to probe here; only ffplay / mpv / play can possibly exist
            ;;
        *)
            # Is there anything to play *to*? (a headless cloud server has neither)
            rt="${XDG_RUNTIME_DIR:-/run/user/$(id -u 2>/dev/null || echo 0)}"
            if [ -S "$rt/pulse/native" ] || [ -S "$rt/pipewire-0" ] || [ -n "${PULSE_SERVER:-}" ]; then has_pulse=1; fi
            if [ -d /dev/snd ]; then has_alsa=1; fi
            ;;
    esac

    if [ $# -eq 0 ] && [ "$has_pulse$has_alsa" != "00" ]; then
        if [ "$has_pulse" = 1 ] && command -v paplay >/dev/null 2>&1; then
            set -- paplay "$wav"
        elif [ "$has_pulse" = 1 ] && command -v pw-play >/dev/null 2>&1; then
            set -- pw-play "$wav"
        elif [ "$has_alsa" = 1 ] && command -v aplay >/dev/null 2>&1; then
            set -- aplay -q "$wav"
        elif command -v ffplay >/dev/null 2>&1; then
            set -- ffplay -nodisp -autoexit -loglevel quiet "$wav"
        elif command -v mpv >/dev/null 2>&1; then
            set -- mpv --no-video --really-quiet "$wav"
        elif command -v play >/dev/null 2>&1; then
            set -- play -q "$wav"
        fi
    fi

    if [ $# -gt 0 ]; then
        ( trap '' HUP; "$@" >/dev/null 2>&1 & )
        return 0
    fi

    # No usable player: fall back to the terminal bell
    ( _xai_bell "$bells" ) >/dev/null 2>&1 &
    return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    play_sound "$@"
fi
