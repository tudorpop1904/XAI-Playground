"""
scripts/watch_and_shutdown.py
=============================
Local watcher script for Windows/Linux to monitor the GCP VM training and stop the instance.
Run with:
    python scripts/watch_and_shutdown.py
"""

import subprocess
import time
import sys

VM_NAME = "xai-vm"
ZONE = "europe-west2-a"
CHECK_INTERVAL_SECONDS = 45
IDLE_THRESHOLD_CHECKS = 3

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip()

def check_vm_gpu_utilization() -> int:
    # SSH into VM and query GPU utilization via nvidia-smi
    ssh_cmd = [
        "gcloud", "compute", "ssh", VM_NAME,
        f"--zone={ZONE}",
        "--command=nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0"
    ]
    code, out = run_cmd(ssh_cmd)
    if code != 0 or not out:
        return 0
    try:
        # Extract last line in case of motd/banner
        last_line = out.strip().split("\n")[-1].strip()
        return int(last_line)
    except Exception:
        return 0

def stop_vm_instance():
    print(f"\n[!] Stopping GCP VM instance '{VM_NAME}'...")
    stop_cmd = [
        "gcloud", "compute", "instances", "stop", VM_NAME,
        f"--zone={ZONE}"
    ]
    code, out = run_cmd(stop_cmd)
    if code == 0:
        print("[+] VM stopped successfully! Good night!")
    else:
        print(f"[-] Failed to stop VM: {out}")

def main():
    print("=" * 60)
    print(f"[*] Starting Auto-Shutdown Monitor for '{VM_NAME}' in zone '{ZONE}'")
    print(f"[*] Polling every {CHECK_INTERVAL_SECONDS} seconds...")
    print("=" * 60)

    idle_count = 0

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        gpu_util = check_vm_gpu_utilization()

        if gpu_util > 10:
            idle_count = 0
            print(f"[{timestamp}] Model training active (GPU: {gpu_util}%). Continuing...")
        else:
            idle_count += 1
            print(f"[{timestamp}] GPU idle / training finished (Check {idle_count}/{IDLE_THRESHOLD_CHECKS}, GPU: {gpu_util}%)")

            if idle_count >= IDLE_THRESHOLD_CHECKS:
                print(f"[{timestamp}] Training finished! Triggering VM shutdown.")
                stop_vm_instance()
                sys.exit(0)

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
