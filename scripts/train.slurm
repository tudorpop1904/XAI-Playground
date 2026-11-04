#!/usr/bin/env bash

#SBATCH --job-name=xai-train
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=storage/logs/train_%j.log

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

uv run --frozen python -m api.workers.train_worker "$@" 