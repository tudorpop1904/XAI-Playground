@echo off
rem Best-effort audio cue for the XAI Playground bootstrap scripts (pure CMD, no other shell).
rem
rem   Usage: call play.cmd NAME [wait]      NAME = startup, shutdown, warning, error or info
rem
rem Never fails. CMD has no built-in silent audio, so it tries, in order:
rem   1. ffplay (if installed)                    - silent, no window
rem   2. Windows Media Player (minimized)         - closes itself when the sound ends
rem   3. terminal bell                            - always works, even on headless/remote consoles
rem "wait" blocks until the sound ends (use right before the script exits).
rem Set XAI_MUTE=1 to silence everything.
setlocal
if defined XAI_MUTE exit /b 0
if "%~1"=="" exit /b 0

if not defined ESC for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%b"

set "WAV=%~dp0wav\%~1.wav"
if not exist "%WAV%" (
    echo %ESC%[2mAudio file '%~1.wav' not found.%ESC%[0m
    exit /b 0
)

where ffplay >nul 2>nul
if not errorlevel 1 goto use_ffplay
set "WMP=%ProgramFiles%\Windows Media Player\wmplayer.exe"
if exist "%WMP%" goto use_wmp
goto bell

:use_ffplay
if /i "%~2"=="wait" (
    ffplay -nodisp -autoexit -loglevel quiet "%WAV%" >nul 2>nul
) else (
    start "" /B ffplay -nodisp -autoexit -loglevel quiet "%WAV%" >nul 2>nul
)
exit /b 0

:use_wmp
if /i "%~2"=="wait" (
    start "" /min /wait "%WMP%" /Play /Close "%WAV%"
) else (
    start "" /min "%WMP%" /Play /Close "%WAV%"
)
exit /b 0

:bell
rem BEL (0x07) has no keyboard form in batch; forfiles can emit it
for /f %%B in ('forfiles /p "%SystemRoot%\System32" /m cmd.exe /c "cmd /c echo 0x07" 2^>nul') do <nul set /p "=%%B"
exit /b 0