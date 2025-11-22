#!/usr/bin/env python3
"""
Download pretrained Swin-UNet weights from the official repository.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, filename: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(filename)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False


def main():
    """Download the pretrained Swin-Tiny weights."""
    weights_urls = {
        "swin_tiny_patch4_window7_224.pth": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
    }
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    print("Downloading pretrained Swin Transformer weights...")
    
    for filename, url in weights_urls.items():
        filepath = weights_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists")
            continue
        
        print(f"Downloading {filename}...")
        success = download_file(url, str(filepath))
        
        if success:
            print(f"✓ Successfully downloaded {filename}")
        else:
            print(f"✗ Failed to download {filename}")
            return False
    
    print("\nAll pretrained weights downloaded successfully!")
    print(f"Weights saved to: {weights_dir.absolute()}")
    return True


if __name__ == "__main__":
    main()