"""
Test script to verify the model and pipeline
"""

import sys
import torch
import numpy as np

print("=" * 60)
print("CheXmask Swin-UNet Setup Verification")
print("=" * 60)

print("\n1. Testing config import...")
try:
    import config
    print("   ✓ Config imported successfully")
    print(f"   - Dataset: {config.DATASET_NAME}")
    print(f"   - Image size: {config.IMAGE_SIZE}")
    print(f"   - Batch size: {config.BATCH_SIZE}")
except Exception as e:
    print(f"   ✗ Error importing config: {e}")
    sys.exit(1)

print("\n2. Testing model imports...")
try:
    from models.swin_unet import build_swin_unet
    print("   ✓ Model imports successful")
except Exception as e:
    print(f"   ✗ Error importing models: {e}")
    sys.exit(1)

print("\n3. Testing model construction...")
try:
    model = build_swin_unet(config.SWIN_CONFIG)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   ✓ Model built successfully")
    print(f"   - Parameters: {num_params:.2f}M")
except Exception as e:
    print(f"   ✗ Error building model: {e}")
    sys.exit(1)

print("\n4. Testing forward pass...")
try:
    dummy_input = torch.randn(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Expected: (1, {config.NUM_CLASSES}, {config.IMAGE_SIZE}, {config.IMAGE_SIZE})")
    
    assert output.shape == (1, config.NUM_CLASSES, config.IMAGE_SIZE, config.IMAGE_SIZE), \
        "Output shape mismatch!"
    print("   ✓ Output shape correct")
except Exception as e:
    print(f"   ✗ Error in forward pass: {e}")
    sys.exit(1)

print("\n5. Testing loss functions...")
try:
    from utils.losses import get_loss_function
    
    criterion = get_loss_function(config)
    
    dummy_target = torch.randint(0, config.NUM_CLASSES, (1, config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    loss = criterion(output, dummy_target)
    
    print(f"   ✓ Loss function works")
    print(f"   - Loss value: {loss.item():.4f}")
except Exception as e:
    print(f"   ✗ Error testing loss: {e}")
    sys.exit(1)

print("\n6. Testing metrics...")
try:
    from utils.losses import SegmentationMetrics
    
    metrics = SegmentationMetrics(config.NUM_CLASSES, device='cpu')
    metrics.update(output, dummy_target)
    result = metrics.get_mean_metrics()
    
    print(f"   ✓ Metrics computation works")
    print(f"   - Mean Dice: {result['mean_dice']:.4f}")
    print(f"   - Mean IoU: {result['mean_iou']:.4f}")
except Exception as e:
    print(f"   ✗ Error testing metrics: {e}")
    sys.exit(1)

print("\n7. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available")
    print(f"   - Device: {torch.cuda.get_device_name(0)}")
    print(f"   - CUDA version: {torch.version.cuda}")
    print(f"   - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        model_gpu = model.to('cuda')
        dummy_input_gpu = dummy_input.to('cuda')
        with torch.no_grad():
            output_gpu = model_gpu(dummy_input_gpu)
        print(f"   ✓ GPU forward pass successful")
        
        del model_gpu, dummy_input_gpu, output_gpu
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ⚠ Warning: GPU forward pass failed: {e}")
else:
    print(f"   ⚠ Warning: CUDA not available, will use CPU")
    print(f"   - Training will be very slow on CPU")

print("\n8. Checking dataset paths...")
import os
if os.path.exists(config.CSV_PATH):
    print(f"   ✓ CSV file found: {config.CSV_PATH}")
    
    try:
        import pandas as pd
        df = pd.read_csv(config.CSV_PATH, nrows=1)
        print(f"   ✓ CSV file is readable")
        print(f"   - Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not read CSV: {e}")
else:
    print(f"   ⚠ Warning: CSV file not found: {config.CSV_PATH}")

if config.IMAGE_BASE_PATH != "SET_YOUR_IMAGE_PATH_HERE":
    if os.path.exists(config.IMAGE_BASE_PATH):
        print(f"   ✓ Image base path exists: {config.IMAGE_BASE_PATH}")
    else:
        print(f"   ⚠ Warning: Image path not found: {config.IMAGE_BASE_PATH}")
else:
    print(f"   ⚠ Warning: IMAGE_BASE_PATH not set in config.py")
    print(f"   - Please update config.py with your image directory path")

print("\n9. Checking required packages...")
required_packages = [
    'torch', 'torchvision', 'numpy', 'pandas', 'cv2', 
    'sklearn', 'albumentations', 'matplotlib', 'tqdm'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'cv2':
            import cv2
        elif package == 'sklearn':
            import sklearn
        else:
            __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT FOUND")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠ Missing packages: {', '.join(missing_packages)}")
    print(f"   Install with: pip install -r requirements.txt")

print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print("✓ Model architecture: OK")
print("✓ Forward pass: OK")
print("✓ Loss functions: OK")
print("✓ Metrics: OK")

if torch.cuda.is_available():
    print("✓ GPU support: OK")
else:
    print("⚠ GPU support: NOT AVAILABLE (will be slow)")

if config.IMAGE_BASE_PATH != "SET_YOUR_IMAGE_PATH_HERE":
    print("✓ Configuration: OK")
else:
    print("⚠ Configuration: NEEDS SETUP")
    print("\n  NEXT STEPS:")
    print("  1. Edit config.py and set IMAGE_BASE_PATH")
    print("  2. Verify your image files are accessible")
    print("  3. Run: python train.py")

if missing_packages:
    print(f"⚠ Dependencies: MISSING {len(missing_packages)} packages")
else:
    print("✓ Dependencies: OK")

print("\n" + "=" * 60)
print("Ready to train! Run: python train.py")
print("=" * 60)
