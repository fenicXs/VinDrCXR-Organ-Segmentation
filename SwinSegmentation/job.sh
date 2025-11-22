#!/bin/bash
#SBATCH -N 1                     # number of nodes
#SBATCH -c 8                     # number of CPU cores
#SBATCH -t 0-16:00:00            # walltime 4 hours (h:mm:ss or d-hh:mm:ss)
#SBATCH -p htc                   # partition: htc is often preferred for throughput
#SBATCH -q public                # QoS (keep or change if your project requires)
#SBATCH --gres=gpu:a100:1        # request 1 A100 GPU specifically (or --gres=gpu:1 for any GPU)
#SBATCH --mem=32G                # memory
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pkrish52@asu.edu
#SBATCH --export=NONE

# Load modules and activate environment
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0   # match the CUDA module available on SOL
source activate /home/mmohan21/myenv            # or `mamba activate medmnist`

# Move to repo directory
cd /home/pkrish52/IPA/Segmentation/VinDrCXR_Segmentation/SwinSegmentation

echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Starting training..."
# Run training
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