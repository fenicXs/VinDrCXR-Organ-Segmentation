"""
Configuration file for CheXmask Segmentation with Swin Transformer
"""

import os

CODE_DIR = r"/home/pkrish52/IPA/Segmentation/VinDrCXR_Segmentation/SwinSegmentation"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

BASE_PATH = r"/scratch/pkrish52"
DATASET_DIR = os.path.join(BASE_PATH, "CheXmask Dataset")
DATABASE_DIR = os.path.join(BASE_PATH, "CheXmask-Database")

DATASET_NAME = 'VinDr-CXR'

CSV_PATH = os.path.join(DATASET_DIR, "Preprocessed", f"{DATASET_NAME}.csv")

IMAGE_BASE_PATH = r"/scratch/tprasad6/my_mimic_project/vinDR/vinbigdata"

USE_VINDR_OFFICIAL_SPLIT = True
VINDR_SPLIT_DIR = os.path.join(PROJECT_ROOT, "VinDrCXR Official split")
VINDR_TRAIN_SPLIT_PATH = os.path.join(VINDR_SPLIT_DIR, "vindrcxr_organSegmentation_train.txt")
VINDR_TEST_SPLIT_PATH = os.path.join(VINDR_SPLIT_DIR, "vindrcxr_organSegmentation_test.txt")

QUALITY_THRESHOLD = 0.75  # Higher quality masks

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # Used for random splits or as val fraction of official VinDr train set
TEST_RATIO = 0.15

SEED = 42

NUM_CLASSES = 4

IMAGE_SIZE = 1024  # Must be divisible by patch_size * window_size (4 * 7 = 28); 896/4=224, 224/7=32

SWIN_CONFIG = {
    'img_size': IMAGE_SIZE,
    'patch_size': 4,
    'in_chans': 1,  # Grayscale images
    'num_classes': NUM_CLASSES,
    'embed_dim': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32],
    'window_size': 8,
    'mlp_ratio': 4.,
    'qkv_bias': True,
    'drop_rate': 0.0,
    'drop_path_rate': 0.2,
    'ape': False,
    'patch_norm': True,
}

USE_PRETRAINED = True
PRETRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "swin_base_patch4_window7_224.pth")

FREEZE_ENCODER_EPOCHS = 3  # Freeze encoder for first 3 epochs in test run
PRETRAINED_NORMALIZE_INPUT = True  # Use ImageNet normalization

BATCH_SIZE = 8  # Small batch for testing
NUM_EPOCHS = 40  # Short validation run (3-5 epochs as requested)

LEARNING_RATE = 2e-4  # Reduced for batch 64 (was 3e-4 for batch 32)
WEIGHT_DECAY = 0.01  # Good for regularization

OPTIMIZER = 'adamw'

USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'  # 'cosine', 'step', 'plateau'
SCHEDULER_PATIENCE = 15  # Increased for 50 epochs
SCHEDULER_FACTOR = 0.5
WARMUP_EPOCHS = 8  # CRITICAL: Increased for batch 64 (was 5 for batch 32)

GRADIENT_CLIP_NORM = 0.5  # Reduced from 1.0 for more aggressive clipping

GRADIENT_ACCUMULATION_STEPS = 1

CLASS_WEIGHTS = [0.2, 1.0, 1.0, 1.5]  # More balanced, emphasize heart (smallest organ)

LOSS_TYPE = 'combined'
DICE_WEIGHT = 0.7  # Prioritize Dice for segmentation
CE_WEIGHT = 0.3

USE_EARLY_STOPPING = False  # Disabled to let training complete (enable after convergence verified)
EARLY_STOPPING_PATIENCE = 20  # When enabled, use 20 for 50-epoch training

USE_AUGMENTATION = True
AUG_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation_range': 10,  # Reduced from 15 to match CheXmask's approach (std deviation ~3.33 degrees)
    'brightness': 0.2,  # Reduced from 0.3 for more conservative augmentation
    'contrast': 0.2,  # Reduced from 0.3
    'gamma_range': (85, 115),  # Slightly reduced range for stability (0.85-1.15)
    'noise_std': 0.01,  # Reduced from 0.02 to match CheXmask more closely (1/128 ≈ 0.0078)
}

CHECKPOINT_DIR = os.path.join(CODE_DIR, "checkpoints")

OUTPUT_DIR = os.path.join(CODE_DIR, "outputs")

SAVE_EVERY = 10  # Save every 10 epochs (reduces disk usage)

KEEP_BEST_N = 3

EXPERIMENT_NAME = f"swin_seg_{DATASET_NAME}"

LOG_FREQ = 10  # Log every 10 iterations for better monitoring

VAL_FREQ = 1

DEVICE = 'cuda'  # 'cuda' or 'cpu'

NUM_WORKERS = 8  # Disable multiprocessing for Windows compatibility

USE_AMP = True

USE_MULTI_GPU = False

PRED_THRESHOLD = 0.5

USE_CRF = False  # Conditional Random Field post-processing
USE_MORPHOLOGY = True  # Morphological operations (opening, closing)

SAVE_VISUALIZATIONS = True
VIS_OVERLAY_ALPHA = 0.4
VIS_FREQ = 5  # Save prediction visualizations every N epochs
VIS_ON_IMPROVEMENT = True  # Save visualization when validation improves

METRICS = ['dice', 'iou', 'precision', 'recall', 'hausdorff']
COMPUTE_HAUSDORFF = False  # Computationally expensive

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, EXPERIMENT_NAME), exist_ok=True)
