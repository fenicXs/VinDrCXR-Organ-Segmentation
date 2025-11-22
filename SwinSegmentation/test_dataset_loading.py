"""
Test script to verify dataset loading with actual VinDr images
"""
import os
import torch
from config import *
from utils.dataset import CheXmaskDataset
import pandas as pd

def test_dataset_loading():
    """Test if we can load actual data samples"""
    print("🧪 Testing dataset loading...")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Total samples in CSV: {len(df)}")
    
    found_images = []
    for i, row in df.head(10).iterrows():
        image_id = row['image_id']
        
        train_path = os.path.join(IMAGE_BASE_PATH, 'train', f"{image_id}.dicom")
        test_path = os.path.join(IMAGE_BASE_PATH, 'test', f"{image_id}.dicom")
        
        if os.path.exists(train_path):
            found_images.append((image_id, 'train'))
        elif os.path.exists(test_path):
            found_images.append((image_id, 'test'))
    
    print(f"Found {len(found_images)} images from first 10 CSV entries:")
    for img_id, split in found_images[:5]:
        print(f"  - {img_id} (in {split}/)")
    
    if not found_images:
        print("❌ No matching images found! CSV image_ids don't match VinDr filenames")
        return False
    
    try:
        found_ids = [img_id for img_id, _ in found_images]
        df_filtered = df[df['image_id'].isin(found_ids)]
        
        dataset = CheXmaskDataset(
            csv_path=None,  # We'll pass dataframe directly
            image_base_path=IMAGE_BASE_PATH,
            dataset_name=DATASET_NAME,
            dataframe=df_filtered  # Pass filtered dataframe
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        sample = dataset[0]
        image, mask = sample['image'], sample['mask']
        
        print(f"✅ Sample loaded successfully:")
        print(f"   Image shape: {image.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Image dtype: {image.dtype}")
        print(f"   Mask dtype: {mask.dtype}")
        print(f"   Unique mask values: {torch.unique(mask)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\n🎉 Dataset loading test PASSED - Ready for training!")
    else:
        print("\n❌ Dataset loading test FAILED - Check the issues above")