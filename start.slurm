#!/usr/bin/env bash
#SBATCH --job-name=xai-playground
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=storage/logs/slurm_%j.log

mkdir -p storage/logs
./start.sh
