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
cd /home/pkrish52/IPA/Segmentation/VinDrCXR_Segmentation/SwinSegmentation/

echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Run training
python inference.py --checkpoint /home/pkrish52/IPA/Segmentation/VinDrCXR_Segmentation/SwinSegmentation/checkpoints/swin_seg_VinDr-CXR/checkpoint_epoch_29.pth --dataset test --save_vis --save_masks