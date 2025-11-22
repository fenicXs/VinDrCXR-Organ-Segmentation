#!/usr/bin/env python3
"""
Quick test run of the training pipeline with pretrained Swin-UNet.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from models.swin_unet_pretrained import build_pretrained_swin_unet
from utils.dataset import get_dataloaders
from utils.losses import get_loss_function, SegmentationMetrics


def quick_test():
    """Run a quick test of the training pipeline."""
    print("Quick Training Pipeline Test")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nBuilding pretrained model...")
    model = build_pretrained_swin_unet(config)
    model = model.to(device)
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    criterion = get_loss_function(config)
    criterion = criterion.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print("\nTesting training step...")
    model.train()
    
    try:
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        print(f"Input shape: {images.shape}")
        print(f"Target shape: {masks.shape}")
        
        outputs = model(images)
        print(f"Output shape: {outputs.shape}")
        
        loss = criterion(outputs, masks)
        print(f"Loss: {loss.item():.4f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✓ Training step successful!")
        
        print("\nTesting validation step...")
        model.eval()
        
        with torch.no_grad():
            batch = next(iter(val_loader))
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            val_loss = criterion(outputs, masks)
            
            metrics = SegmentationMetrics(config.NUM_CLASSES, device)
            metrics.update(outputs, masks)
            val_metrics = metrics.get_mean_metrics()
            
            print(f"Val loss: {val_loss.item():.4f}")
            print(f"Val Dice: {val_metrics['mean_dice']:.4f}")
            print(f"Val IoU: {val_metrics['mean_iou']:.4f}")
        
        print("✓ Validation step successful!")
        
        print("\nTesting encoder freeze/unfreeze...")
        
        encoder_params_trainable = sum(p.numel() for p in model.encoder_parameters() if p.requires_grad)
        print(f"Encoder trainable params: {encoder_params_trainable:,}")
        
        if encoder_params_trainable == 0:
            print("✓ Encoder is correctly frozen")
        else:
            print("✓ Encoder is not frozen (as expected)")
        
        model.unfreeze_encoder()
        encoder_params_trainable = sum(p.numel() for p in model.encoder_parameters() if p.requires_grad)
        print(f"After unfreeze - Encoder trainable params: {encoder_params_trainable:,}")
        
        print("\n✓ All tests passed! Ready for full training.")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\nPipeline test successful! You can now run the full training.")
    else:
        print("\nPipeline test failed. Please check the configuration.")
        sys.exit(1)