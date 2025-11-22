# Swin-UNet Segmentation for CheXmask Dataset

This directory contains a complete implementation of Swin Transformer-based U-Net for chest X-ray segmentation using the CheXmask dataset.

## 📁 Project Structure

```
SwinSegmentation/
├── config.py              # Configuration file (modify this!)
├── train.py              # Training script
├── inference.py          # Inference script
├── requirements.txt      # Python dependencies
├── models/
│   ├── swin_transformer.py  # Swin Transformer implementation
│   └── swin_unet.py         # Swin-UNet model
├── utils/
│   ├── dataset.py        # Dataset loader
│   ├── losses.py         # Loss functions and metrics
│   └── visualization.py  # Visualization utilities
├── checkpoints/          # Saved model checkpoints
└── outputs/              # Training outputs and predictions
```

## 🚀 Quick Start

### 1. Install Dependencies

First, make sure you have PyTorch installed with CUDA support (for GPU training):

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Configure Paths

**IMPORTANT**: Edit `config.py` and set the following:

- `IMAGE_BASE_PATH`: Path to your chest X-ray images
- `DATASET_NAME`: Which dataset you're using (e.g., 'VinDr-CXR', 'MIMIC-CXR-JPG', etc.)
- Other hyperparameters as needed

### 3. Train the Model

```bash
python train.py
```

To resume from a checkpoint:
```bash
python train.py --resume checkpoints/swin_seg_VinDr-CXR/best_model.pth
```

### 4. Run Inference

On test set:
```bash
python inference.py --checkpoint checkpoints/swin_seg_VinDr-CXR/best_model.pth --dataset test --save_masks --save_vis
```

On a single image:
```bash
python inference.py --checkpoint checkpoints/swin_seg_VinDr-CXR/best_model.pth --image_path path/to/image.png --output_dir results/
```

## 📊 Model Architecture

The Swin-UNet model combines:
- **Encoder**: Swin Transformer with hierarchical attention
- **Decoder**: Symmetric U-Net style decoder with skip connections
- **Output**: Multi-class segmentation (4 classes: Background, Right Lung, Left Lung, Heart)

Default configuration:
- Input size: 1024×1024 (grayscale)
- Embed dim: 96
- Depths: [2, 2, 6, 2]
- Num heads: [3, 6, 12, 24]
- Window size: 8

## 🎯 Training Details

### Loss Functions
- Cross Entropy Loss
- Dice Loss
- Combined Loss (default)

### Data Augmentation
- Random rotation (±10°)
- Random brightness/contrast
- Random gamma correction
- Gaussian noise
- Horizontal flip

### Optimization
- Optimizer: AdamW
- Learning rate: 1e-4
- Scheduler: Cosine annealing
- Mixed precision training (AMP)
- Early stopping

## 📈 Monitoring

During training, the following are saved:
- Model checkpoints every N epochs
- Best model based on validation Dice score
- Training curves (loss, Dice, IoU)
- Sample predictions every 5 epochs

## 🔧 Configuration Options

Key parameters in `config.py`:

```python
# Dataset
DATASET_NAME = 'VinDr-CXR'
IMAGE_BASE_PATH = 'path/to/images'
QUALITY_THRESHOLD = 0.7  # Filter low-quality annotations

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
IMAGE_SIZE = 1024

# Model
SWIN_CONFIG = {
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 8,
}
```

## 📝 Notes

1. **GPU Memory**: 1024×1024 images require significant GPU memory. Reduce `BATCH_SIZE` or `IMAGE_SIZE` if you encounter OOM errors.

2. **Image Paths**: The dataset loader needs to be customized based on your image directory structure. Modify the `_load_image` method in `utils/dataset.py`.

3. **Quality Filtering**: Only samples with `Dice RCA (Mean) >= 0.7` are used by default.

4. **Multi-GPU**: Set `USE_MULTI_GPU = True` in config to use DataParallel.

## 📊 Expected Performance

With proper training, you should achieve:
- Mean Dice Score: 0.90+
- Mean IoU Score: 0.85+
- Per-organ metrics vary by dataset and quality

## 🐛 Troubleshooting

**Issue**: "Image not found" errors
- **Solution**: Check `IMAGE_BASE_PATH` in config.py and customize `_load_image()` in dataset.py

**Issue**: Out of memory errors
- **Solution**: Reduce `BATCH_SIZE` (try 2 or 1) or `IMAGE_SIZE` (try 512)

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📚 References

- Swin Transformer: https://arxiv.org/abs/2103.14030
- CheXmask Dataset: https://physionet.org/content/chexmask-cxr-segmentation-data/
- HybridGNet: https://github.com/ngaggion/HybridGNet


## Notes on Data Usage

- The CheXmask-derived masks remain governed by the CheXmask-Database license. Ensure compliance with the usage restrictions of each underlying dataset (MIMIC-CXR, CheXpert, VinDr-CXR).
- No raw images are distributed in this repository. Only official metadata tables and curated splits are included.

## Citation

If you use the CheXmask dataset assets included or referenced by this repository, please cite the original release:

```
@misc{gaggion2023chexmaskPhysioNet,
   author = {Gaggion, N. and Mosquera, C. and Aineseder, M. and Mansilla, L. and Milone, D. and Ferrante, E.},
   title = {{CheXmask Database: a large-scale dataset of anatomical segmentation masks for chest x-ray images (version 0.1)}},
   year = {2023},
   howpublished = {PhysioNet},
   note = {https://doi.org/10.13026/dx54-8351}
}
```

For experiments or derivatives relying on the Swin-Unet implementation, please cite the corresponding publications:

```
@InProceedings{swinunet,
   author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
   title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
   booktitle = {Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
   year = {2022}
}

@misc{cao2021swinunet,
   title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
   author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
   year = {2021},
   eprint = {2105.05537},
   archivePrefix = {arXiv},
   primaryClass = {eess.IV}
}
```

## Acknowledgements

This work stands on the shoulders of the CheXmask-Database and Swin-Unet teams. Please cite their publications when publishing results obtained from this workflow.
