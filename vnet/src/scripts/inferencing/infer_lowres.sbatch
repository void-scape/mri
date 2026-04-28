#!/bin/bash
#SBATCH --job-name=lowres_infer_dice
#SBATCH --partition=ondemand-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lowres_infer_dice_%j.out
#SBATCH --error=logs/lowres_infer_dice_%j.err

set -e

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs predictions

module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

CKPT="work/vnet.lowres.oneclass/vnet_model_best.pth.tar"
IM_PATH="/bsuscratch/jamesstringham/data/valid/valid_001_V00.im"
SEG_PATH="/bsuscratch/jamesstringham/data/valid/valid_001_V00.seg"
OUT_PATH="predictions/valid_001_V00_pred.npy"

echo "Checkpoint: $CKPT"
echo "Image:      $IM_PATH"
echo "Seg:        $SEG_PATH"
echo "Output:     $OUT_PATH"

# Run inference exactly through current standalone inference script
python -m src.inference.infer_knee \
  --ckpt "$CKPT" \
  --im "$IM_PATH" \
  --out "$OUT_PATH" \
  --seg "$SEG_PATH" \
  --device cuda \
  --profile-memory