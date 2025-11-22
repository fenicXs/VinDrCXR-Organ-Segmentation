#!/bin/bash
#SBATCH --job-name=chexmask_swin_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# This is a SLURM batch script for training on ASU SOL cluster
# Adjust the partition, GPU type, and resource requirements as needed

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load required modules (adjust based on your cluster setup)
module load cuda/11.8
module load anaconda3

# Activate your conda environment if you have one
# conda activate chexmask_env

# Print Python and PyTorch versions
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo ""

# Create logs directory
mkdir -p logs

# Run training
echo "Starting training..."
python train.py

echo "=========================================="
echo "Training complete!"
echo "=========================================="

# Run testing/inference after training completes
echo ""
echo "=========================================="
echo "Starting inference on test set..."
echo "=========================================="
python inference.py --checkpoint checkpoints/swin_seg_VinDr-CXR/best_model.pth --split test

echo "=========================================="
echo "Inference complete!"
echo "Results saved to outputs directory"
echo "=========================================="

# Analyze and visualize results
echo ""
echo "=========================================="
echo "Generating result analysis..."
echo "=========================================="
python analyze_results.py

echo ""
echo "=========================================="
echo "All tasks complete!"
echo "=========================================="
