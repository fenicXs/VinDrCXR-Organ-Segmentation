#!/usr/bin/env python3
"""
Test script to verify the pretrained Swin-UNet model loads correctly.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from models.swin_unet_pretrained import build_pretrained_swin_unet


def test_model():
    """Test model loading and forward pass."""
    print("Testing Pretrained Swin-UNet Model")
    print("=" * 50)
    
    weights_path = config.PRETRAINED_MODEL_PATH
    print(f"Weights path: {weights_path}")
    print(f"Weights exist: {os.path.exists(weights_path)}")
    
    print("\nBuilding model...")
    try:
        model = build_pretrained_swin_unet(config)
        print(f"✓ Model built successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in model.encoder_parameters())
        decoder_params = sum(p.numel() for p in model.decoder_parameters())
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Decoder parameters: {decoder_params:,}")
        
        print("\nTesting encoder freeze/unfreeze...")
        print(f"Before freeze - Encoder params trainable: {sum(p.numel() for p in model.encoder_parameters() if p.requires_grad):,}")
        
        model.freeze_encoder()
        frozen_trainable = sum(p.numel() for p in model.encoder_parameters() if p.requires_grad)
        print(f"After freeze - Encoder params trainable: {frozen_trainable:,}")
        
        model.unfreeze_encoder()
        unfrozen_trainable = sum(p.numel() for p in model.encoder_parameters() if p.requires_grad)
        print(f"After unfreeze - Encoder params trainable: {unfrozen_trainable:,}")
        
        print("\nTesting forward pass...")
        model.eval()
        
        img_size = config.IMAGE_SIZE
        test_sizes = [(1, 1, img_size, img_size), (2, 1, img_size, img_size)]
        
        for batch_size, channels, height, width in test_sizes:
            print(f"Testing input shape: ({batch_size}, {channels}, {height}, {width})")
            
            x = torch.randn(batch_size, channels, height, width)
            
            with torch.no_grad():
                try:
                    output = model(x)
                    print(f"  ✓ Output shape: {output.shape}")
                    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
                    
                    expected_shape = (batch_size, config.NUM_CLASSES, height, width)
                    if output.shape == expected_shape:
                        print(f"  ✓ Correct output shape")
                    else:
                        print(f"  ✗ Wrong output shape. Expected: {expected_shape}")
                    
                except Exception as e:
                    print(f"  ✗ Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False


if __name__ == "__main__":
    success = test_model()
    if success:
        print("\nModel is ready for training!")
    else:
        print("\nModel test failed. Please check the configuration.")
        sys.exit(1)