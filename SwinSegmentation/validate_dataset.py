"""
Quick validation script to check dataset loading with real CheXmask data
Run this before full training to ensure everything works
"""

import pandas as pd
import os
import sys
from config import *

def validate_csv_data():
    """Validate the CSV file can be loaded and has expected structure"""
    print(f"🔍 Validating CSV file: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: CSV file not found at {CSV_PATH}")
        return False
        
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV loaded successfully. Shape: {df.shape}")
        
        required_cols = ['image_id', 'Left Lung', 'Right Lung', 'Heart', 'Height', 'Width']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ ERROR: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return False
        else:
            print(f"✅ All required columns present")
            
        print(f"\n📊 Sample data:")
        print(f"First image_id: {df.iloc[0]['image_id']}")
        print(f"Image dimensions: {df.iloc[0]['Height']}x{df.iloc[0]['Width']}")
        print(f"Has segmentation masks: {len(df.iloc[0]['Left Lung']) > 10}")  # RLE should be long string
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        return False

def validate_image_path():
    """Check if image path is configured (user needs to set this)"""
    print(f"\n📁 Checking image path: {IMAGE_BASE_PATH}")
    
    if "path\\to\\your" in IMAGE_BASE_PATH:
        print(f"⚠️  WARNING: Image path not configured!")
        print(f"   Current path: {IMAGE_BASE_PATH}")
        print(f"   You need to:")
        print(f"   1. Download VinDr-CXR dataset from https://vindr.ai/datasets/cxr")
        print(f"   2. Update IMAGE_BASE_PATH in config.py to point to your images")
        return False
    elif os.path.exists(IMAGE_BASE_PATH):
        print(f"✅ Image path exists")
        files = os.listdir(IMAGE_BASE_PATH)[:5]
        print(f"   Sample files: {files}")
        return True
    else:
        print(f"❌ ERROR: Image path does not exist: {IMAGE_BASE_PATH}")
        return False

def validate_dependencies():
    """Check if all required modules can be imported"""
    print(f"\n📦 Checking dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
        
        from utils.dataset import CheXmaskDataset
        print(f"✅ Custom dataset module loaded")
        
        from models.swin_unet_pretrained import PretrainedSwinUNet
        print(f"✅ Pretrained SwinUNet model loaded")
        
        return True
        
    except ImportError as e:
        print(f"❌ ERROR: Missing dependency: {e}")
        return False

def main():
    """Run all validations"""
    print("🚀 CheXmask Dataset Validation")
    print("=" * 50)
    
    csv_ok = validate_csv_data()
    deps_ok = validate_dependencies()
    images_ok = validate_image_path()
    
    print("\n" + "=" * 50)
    if csv_ok and deps_ok:
        if images_ok:
            print("🎉 ALL VALIDATIONS PASSED - Ready for training!")
        else:
            print("⚠️  PARTIAL SUCCESS - CSV and code ready, but image path needs configuration")
            print("   Update IMAGE_BASE_PATH in config.py and you'll be ready to train")
    else:
        print("❌ VALIDATION FAILED - Please fix the issues above")
    
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Epochs: {NUM_EPOCHS} (short validation run)")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

if __name__ == "__main__":
    main()