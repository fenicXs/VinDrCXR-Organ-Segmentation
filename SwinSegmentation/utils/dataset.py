"""
Dataset loader for CheXmask database
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split


def _find_image_id_column(df):
    """Return the column name that stores the image identifier."""
    possible_cols = ['image_id', 'Image Index', 'dicom_id', 'Path', 'ImageID']
    for col in possible_cols:
        if col in df.columns:
            return col
    return df.columns[0]


def _load_split_file(split_path):
    """Load a text file containing one image id per line."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def get_mask_from_RLE(rle, height, width):
    """
    Decode Run-Length Encoding (RLE) to binary mask
    
    Args:
        rle: RLE string (space-separated integers)
        height: Image height
        width: Image width
    
    Returns:
        mask: Binary mask array (height, width) with values 0 or 255
    """
    runs = np.array([int(x) for x in rle.split()])
    starts = runs[::2]
    lengths = runs[1::2]

    mask = np.zeros((height * width), dtype=np.uint8)

    for start, length in zip(starts, lengths):
        start -= 1  
        end = start + length
        mask[start:end] = 255

    mask = mask.reshape((height, width))
    
    return mask


class CheXmaskDataset(Dataset):
    """
    PyTorch Dataset for CheXmask segmentation data
    
    Args:
        csv_path: Path to the CSV file containing annotations
        image_base_path: Base path where the actual X-ray images are stored
        dataset_name: Name of the dataset (e.g., 'VinDr-CXR', 'MIMIC-CXR-JPG')
        quality_threshold: Minimum Dice RCA (Mean) score to include samples
        transform: Optional transforms to apply
        mode: 'train', 'val', or 'test'
    """
    
    def __init__(self, csv_path, image_base_path, dataset_name, 
                 quality_threshold=0.7, transform=None, mode='train', dataframe=None):
        super().__init__()
        
        self.csv_path = csv_path
        self.image_base_path = image_base_path
        self.dataset_name = dataset_name
        self.quality_threshold = quality_threshold
        self.transform = transform
        self.mode = mode
        
        if dataframe is not None:
            print(f"Using provided dataframe with {len(dataframe)} samples")
            self.df = dataframe
        else:
            print(f"Loading {dataset_name} dataset from {csv_path}")
            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df)} samples")
        
        if 'Dice RCA (Mean)' in self.df.columns:
            self.df = self.df[self.df['Dice RCA (Mean)'] >= quality_threshold]
            print(f"After quality filtering (>= {quality_threshold}): {len(self.df)} samples")
        
        self.df = self.df.reset_index(drop=True)
        
        self.image_id_col = _find_image_id_column(self.df)
    
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, image_id):
        """
        Load chest X-ray image
        
        Note: This function needs to be customized based on how images are organized
        in your dataset. Different datasets have different directory structures.
        """
        if self.dataset_name == 'VinDr-CXR':
            img_path = os.path.join(self.image_base_path, 'train', f"{image_id}.dicom")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_base_path, 'test', f"{image_id}.dicom")
            
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_base_path, 'train', image_id)
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_base_path, 'test', image_id)
        
        elif self.dataset_name == 'MIMIC-CXR-JPG':
            img_path = os.path.join(self.image_base_path, image_id)
        
        elif self.dataset_name == 'Padchest':
            img_path = os.path.join(self.image_base_path, image_id)
        
        elif self.dataset_name == 'CheXpert':
            img_path = os.path.join(self.image_base_path, image_id)
        
        else:
            img_path = os.path.join(self.image_base_path, image_id)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        if img_path.lower().endswith('.dicom') or img_path.lower().endswith('.dcm'):
            try:
                import pydicom
                from pydicom.pixel_data_handlers.util import apply_voi_lut
                
                dcm = pydicom.dcmread(img_path)
                
                image = apply_voi_lut(dcm.pixel_array, dcm)
                
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    image = np.max(image) - image
                
                if image.dtype != np.uint8:
                    image = image.astype(np.float64)
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    image = (image * 255).astype(np.uint8)
                
            except Exception as e:
                raise ValueError(f"Could not load DICOM image {img_path}: {e}")
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
        
        return image
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        
        row = self.df.iloc[idx]
        
        image_id = row[self.image_id_col]
        
        try:
            image = self._load_image(image_id)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            image = np.zeros((1024, 1024), dtype=np.float32)
        
        height = int(row['Height'])
        width = int(row['Width'])
        
        right_lung_mask = get_mask_from_RLE(row['Right Lung'], height, width)
        left_lung_mask = get_mask_from_RLE(row['Left Lung'], height, width)
        heart_mask = get_mask_from_RLE(row['Heart'], height, width)
        
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height))
        
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[right_lung_mask > 0] = 1
        mask[left_lung_mask > 0] = 2
        mask[heart_mask > 0] = 3
        
        image = image.astype(np.float32) / 255.0
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        sample = {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'height': height,
            'width': width
        }

        return sample


def create_data_splits(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                      quality_threshold=0.7, seed=42, dataset_name=None,
                      train_split_file=None, test_split_file=None, use_official_split=False):
    """
    Split dataset into train, validation, and test sets.
    
    When use_official_split is True (VinDr-CXR), train_split_file and
    test_split_file should point to the official split lists. val_ratio is
    applied to the official train set to carve out a validation subset.
    
    Returns:
        train_indices, val_indices, test_indices
    """
    df = pd.read_csv(csv_path)
    
    if 'Dice RCA (Mean)' in df.columns:
        df = df[df['Dice RCA (Mean)'] >= quality_threshold]
    
    df = df.reset_index(drop=True)
    
    image_id_col = _find_image_id_column(df)

    if use_official_split and train_split_file and test_split_file:
        train_ids = _load_split_file(train_split_file)
        test_ids = _load_split_file(test_split_file)

        df_image_ids = df[image_id_col].astype(str)
        train_mask = df_image_ids.isin(train_ids)
        test_mask = df_image_ids.isin(test_ids)

        train_indices_all = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        missing_train = len(train_ids) - len(train_indices_all)
        missing_test = len(test_ids) - len(test_indices)
        if missing_train > 0:
            print(f"Warning: {missing_train} official train ids not found after filtering/CSV load")
        if missing_test > 0:
            print(f"Warning: {missing_test} official test ids not found after filtering/CSV load")

        if len(train_indices_all) == 0:
            raise ValueError("No training samples matched the official VinDr-CXR split list.")
        if len(test_indices) == 0:
            raise ValueError("No test samples matched the official VinDr-CXR split list.")

        train_indices, val_indices = train_test_split(
            train_indices_all, test_size=val_ratio, random_state=seed, shuffle=True
        )
        return train_indices, val_indices, test_indices

    indices = np.arange(len(df))

    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed, shuffle=True
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_size), random_state=seed, shuffle=True
    )

    return train_indices, val_indices, test_indices


def get_dataloaders(config):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        config: Configuration module
        
    Returns:
        train_loader, val_loader, test_loader
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    train_transform = None
    val_transform = None
    
    if config.USE_AUGMENTATION:
        train_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5) if config.AUG_CONFIG['horizontal_flip'] else A.NoOp(),
            A.Rotate(
                limit=config.AUG_CONFIG['rotation_range'],  # ±10 degrees
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_LINEAR
            ),
            A.RandomBrightnessContrast(
                brightness_limit=config.AUG_CONFIG['brightness'],  # 0.2 = ±20%
                contrast_limit=config.AUG_CONFIG['contrast'],  # 0.2 = ±20%
                p=0.5
            ),
            A.RandomGamma(
                gamma_limit=(85, 115),  # Albumentations expects (min%, max%)
                p=0.5
            ),
            A.GaussNoise(p=0.3),  # Uses default noise parameters
        ])
    else:
        train_transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        ])
    
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    ])
    
    use_official_split = (
        getattr(config, "USE_VINDR_OFFICIAL_SPLIT", False)
        and str(config.DATASET_NAME).lower() == "vindr-cxr"
    )
    split_val_ratio = getattr(config, "VAL_RATIO", 0.15)

    train_indices, val_indices, test_indices = create_data_splits(
        config.CSV_PATH,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=split_val_ratio,
        test_ratio=config.TEST_RATIO,
        quality_threshold=config.QUALITY_THRESHOLD,
        seed=config.SEED,
        dataset_name=config.DATASET_NAME,
        train_split_file=getattr(config, "VINDR_TRAIN_SPLIT_PATH", None) if use_official_split else None,
        test_split_file=getattr(config, "VINDR_TEST_SPLIT_PATH", None) if use_official_split else None,
        use_official_split=use_official_split
    )
    
    if use_official_split:
        print("Using official VinDr-CXR train/test split files")

    print(f"Data splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    train_dataset_full = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=train_transform,  # Set transform directly
        mode='train'
    )
    
    val_dataset_full = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=val_transform,  # Separate transform for validation
        mode='val'
    )
    
    test_dataset_full = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=val_transform,  # No augmentation for test
        mode='test'
    )
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    import config
    
    dataset = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=None,
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Unique mask values: {torch.unique(sample['mask'])}")
