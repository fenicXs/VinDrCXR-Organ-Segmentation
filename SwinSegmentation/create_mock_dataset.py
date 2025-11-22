#!/usr/bin/env python3
"""
Create mock dataset for testing the pretrained Swin-UNet training pipeline.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config


def mask_to_rle(mask):
    """Convert mask to RLE encoding (simplified)."""
    if np.sum(mask) == 0:
        return ""
    
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_mock_dataset():
    """Create a small mock dataset for testing."""
    print("Creating mock dataset for testing...")
    
    mock_base = Path(config.BASE_PATH) / "mock_data"
    mock_images_dir = mock_base / "images"
    mock_annotations_dir = mock_base / "annotations"
    mock_dataset_dir = Path(config.DATASET_DIR)
    mock_preprocessed_dir = mock_dataset_dir / "Preprocessed"
    
    mock_images_dir.mkdir(parents=True, exist_ok=True)
    mock_annotations_dir.mkdir(parents=True, exist_ok=True)
    mock_preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Mock images directory: {mock_images_dir}")
    print(f"Mock dataset directory: {mock_dataset_dir}")
    
    num_samples = 20  # Small dataset for testing
    img_size = config.IMAGE_SIZE
    
    mock_data = []
    
    for i in range(num_samples):
        np.random.seed(i)  # For reproducibility
        
        img = np.random.normal(0.3, 0.2, (img_size, img_size))
        img = np.clip(img, 0, 1)
        
        y, x = np.ogrid[:img_size, :img_size]
        center_y, center_x = img_size // 2, img_size // 2
        
        right_lung_mask = ((x - center_x * 0.7) ** 2 / (center_x * 0.4) ** 2 + 
                          (y - center_y) ** 2 / (center_y * 0.6) ** 2) < 1
        
        left_lung_mask = ((x - center_x * 1.3) ** 2 / (center_x * 0.4) ** 2 + 
                         (y - center_y) ** 2 / (center_y * 0.6) ** 2) < 1
        
        heart_mask = ((x - center_x * 0.8) ** 2 / (center_x * 0.15) ** 2 + 
                     (y - center_y * 1.2) ** 2 / (center_y * 0.25) ** 2) < 1
        
        img[right_lung_mask] *= 1.5
        img[left_lung_mask] *= 1.5
        img[heart_mask] *= 1.3
        img = np.clip(img, 0, 1)
        
        img_8bit = (img * 255).astype(np.uint8)
        
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        mask[right_lung_mask] = 1  # Right lung
        mask[left_lung_mask] = 2   # Left lung  
        mask[heart_mask] = 3       # Heart (overwrites lung if overlapping)
        
        img_filename = f"mock_image_{i:03d}.png"
        mask_filename = f"mock_mask_{i:03d}.png"
        
        img_path = mock_images_dir / img_filename
        mask_path = mock_annotations_dir / mask_filename
        
        Image.fromarray(img_8bit, mode='L').save(img_path)
        Image.fromarray(mask, mode='L').save(mask_path)
        
        right_lung_rle = mask_to_rle(right_lung_mask.astype(np.uint8))
        left_lung_rle = mask_to_rle(left_lung_mask.astype(np.uint8))
        heart_rle = mask_to_rle(heart_mask.astype(np.uint8))
        
        mock_data.append({
            'image_path': img_filename,  # Just filename, not full path
            'mask_path': str(mask_path),
            'quality_score': np.random.uniform(0.8, 1.0),  # High quality scores
            'dataset': 'VinDr-CXR',
            'image_id': f'mock_{i:03d}',
            'study_id': f'study_{i // 4:03d}',  # Group images into studies
            'has_right_lung': 1,
            'has_left_lung': 1,
            'has_heart': 1,
            'Height': img_size,
            'Width': img_size,
            'Right Lung': right_lung_rle,
            'Left Lung': left_lung_rle,
            'Heart': heart_rle,
        })
    
    df = pd.DataFrame(mock_data)
    csv_path = mock_preprocessed_dir / f"{config.DATASET_NAME}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Created {num_samples} mock samples")
    print(f"✓ Saved CSV to: {csv_path}")
    
    global IMAGE_BASE_PATH
    config.IMAGE_BASE_PATH = str(mock_images_dir)
    
    print(f"✓ Updated IMAGE_BASE_PATH to: {config.IMAGE_BASE_PATH}")
    
    return str(csv_path), str(mock_images_dir)


if __name__ == "__main__":
    csv_path, img_dir = create_mock_dataset()
    print(f"\nMock dataset created successfully!")
    print(f"CSV file: {csv_path}")
    print(f"Images directory: {img_dir}")