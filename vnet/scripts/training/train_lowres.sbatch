#!/bin/bash
#SBATCH --job-name=vnet_lowres
#SBATCH --output=logs/vnet_lowres_%j.out
#SBATCH --error=logs/vnet_lowres_%j.err
#SBATCH --partition=ondemand-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

# Create log directory if it does not exist
mkdir -p logs

# Go to project root (directory where you run sbatch from)
cd "${SLURM_SUBMIT_DIR}"

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"

# Load modules if needed on your cluster
# module load apptainer/1.2.5
# module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

# Activate your environment if needed
# source .venv/bin/activate

module purge
module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

pip install --user h5py setproctitle matplotlib

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -m src.training.train \
  --data /bsuscratch/jamesstringham/data \
  --save work/vnet.lowres.oneclass \
  --nEpochs 300 \
  --batchSz 1 \
  --ngpu 1