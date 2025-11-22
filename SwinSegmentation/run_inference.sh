#!/bin/bash
#SBATCH --job-name=chexmask_swin_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# This is a SLURM batch script for inference on ASU SOL cluster

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load required modules
module load cuda/11.8
module load anaconda3

# Activate your conda environment if you have one
# conda activate chexmask_env

# Create logs directory
mkdir -p logs

# Set the checkpoint path (modify this to your best model)
CHECKPOINT="checkpoints/swin_seg_VinDr-CXR/best_model.pth"

# Run inference on test set
echo "Running inference on test set..."
python inference.py \
    --checkpoint $CHECKPOINT \
    --dataset test \
    --save_masks \
    --save_vis

echo "=========================================="
echo "Inference complete!"
echo "=========================================="
