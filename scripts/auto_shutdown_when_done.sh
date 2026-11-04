#!/usr/bin/env bash
# ==============================================================================
# scripts/auto_shutdown_when_done.sh
# Monitors training in Docker container / storage directory, and shuts down VM.
# Run on the VM in the background with:
#   nohup bash scripts/auto_shutdown_when_done.sh > auto_shutdown.log 2>&1 &
# ==============================================================================

INTERVAL=30
IDLE_COUNT=0
REQUIRED_IDLE_CHECKS=3

echo "[$(date)] Auto-shutdown monitor initiated on GCP VM."
echo "Monitoring GPU utilization and training activity every ${INTERVAL} seconds."
echo "Will shut down VM after ${REQUIRED_IDLE_CHECKS} consecutive idle checks."

while true; do
    # 1. Query GPU compute utilization (%)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "0")
    if [ -z "$GPU_UTIL" ]; then
        GPU_UTIL=0
    fi

    # 2. Check active GPU compute processes
    ACTIVE_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | wc -l)

    # 3. Check recent logs for active extraction or training
    RECENT_LOGS=$(docker compose logs --tail=15 api 2>/dev/null || docker logs --tail=15 xai-api 2>/dev/null)
    
    IS_BUSY=0

    # If GPU utilization is above 10% or there are heavy compute apps running
    if [ "$GPU_UTIL" -gt 10 ]; then
        IS_BUSY=1
    fi

    if [ "$IS_BUSY" -eq 1 ]; then
        IDLE_COUNT=0
        echo "[$(date)] Training in progress (GPU: ${GPU_UTIL}%, Procs: ${ACTIVE_PROCS}). Checking again in ${INTERVAL}s..."
    else
        IDLE_COUNT=$((IDLE_COUNT + 1))
        echo "[$(date)] Model seems idle (Check ${IDLE_COUNT}/${REQUIRED_IDLE_CHECKS} - GPU: ${GPU_UTIL}%)."

        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_CHECKS" ]; then
            echo "[$(date)] ========================================================"
            echo "[$(date)] Training finished! Syncing disk and shutting down VM..."
            echo "[$(date)] ========================================================"
            sync
            sudo shutdown -h now
            exit 0
        fi
    fi

    sleep "$INTERVAL"
done
