#!/bin/bash
#SBATCH --job-name=lowres_infer_dice
#SBATCH --partition=ondemand-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/highres_infer_dice_%j.out
#SBATCH --error=logs/highres_infer_dice_%j.err

set -e

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs predictions

module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

python -m src.inference.infer_7t \
  --ckpt "work/vnet.7t_oneclass_transfer/vnet_model_best.pth.tar" \
  --im "7T_Data/extracted_volumes/volume_10.im" \
  --seg "7T_Data/extracted_volumes/volume_10.seg" \
  --out "predictions/highres_volume_10_pred.h5" \
  --device cuda \
  --dice \
  --profile-memory