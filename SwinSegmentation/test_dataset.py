#!/usr/bin/env python3
"""
Test dataset loading with the mock data.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.dataset import get_dataloaders


def test_dataset():
    """Test dataset loading."""
    print("Testing Dataset Loading")
    print("=" * 50)
    
    print(f"CSV path: {config.CSV_PATH}")
    print(f"Image base path: {config.IMAGE_BASE_PATH}")
    print(f"CSV exists: {os.path.exists(config.CSV_PATH)}")
    print(f"Image dir exists: {os.path.exists(config.IMAGE_BASE_PATH)}")
    
    try:
        print("\nLoading dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(config)
        
        print(f"✓ Train loader: {len(train_loader)} batches, {len(train_loader.dataset)} samples")
        print(f"✓ Val loader: {len(val_loader)} batches, {len(val_loader.dataset)} samples")  
        print(f"✓ Test loader: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
        
        print("\nTesting train batch...")
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image']
            masks = batch['mask']
            
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Masks shape: {masks.shape}")
            print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Mask classes: {torch.unique(masks).tolist()}")
            
            if batch_idx >= 2:  # Test first 3 batches
                break
        
        print("\n✓ Dataset loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\nDataset is ready for training!")
    else:
        print("\nDataset test failed.")
        sys.exit(1)