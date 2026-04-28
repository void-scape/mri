#!/bin/bash
#SBATCH --job-name=vnet_7t
#SBATCH --output=logs/vnet_7t_%j.out
#SBATCH --error=logs/vnet_7t_%j.err
#SBATCH --partition=ondemand-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install --user h5py setproctitle matplotlib

python -m src.training.train_7t \
  --data 7T_Data/Neal_7T_Cartilages_20200504.hdf5 \
  --pretrained work/vnet.lowres.oneclass/vnet_model_best.pth.tar \
  --save work/vnet.7t_oneclass_transfer \
  --nEpochs 500 \
  --batchSz 1 \
  --ngpu 1