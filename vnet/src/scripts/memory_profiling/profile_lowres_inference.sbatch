#!/bin/bash
#SBATCH --job-name=vnet_lowres_resume
#SBATCH --output=logs/vnet_lowres_resume_%j.out
#SBATCH --error=logs/vnet_lowres_resume_%j.err
#SBATCH --partition=ondemand-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"

module purge
module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY


python -m src.inference.infer_knee \
  --ckpt work/vnet.lowres.oneclass/vnet_model_best.pth.tar \
  --im /bsuscratch/jamesstringham/data/valid/valid_001_V00.im \
  --out work/pred_lowres.h5 \
  --profile-memory