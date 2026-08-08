#!/bin/bash
#SBATCH --job-name=sbi-dgpo
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "[nersc] host=$(hostname) job=${SLURM_JOB_ID} started=$(date --iso-8601=seconds)"
nvidia-smi
srun .venv/bin/sbi-toy --config config/nersc.yaml --stage all
