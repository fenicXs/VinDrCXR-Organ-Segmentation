"""
Debug the full forward pass
"""
import torch
import sys
sys.path.append('.')

from models.swin_unet import SwinUNet

config = {
    'img_size': 1024,
    'patch_size': 4,
    'in_chans': 1,
    'num_classes': 4,
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 8,
    'mlp_ratio': 4.,
    'qkv_bias': True,
    'drop_rate': 0.0,
    'drop_path_rate': 0.2,
    'ape': False,
    'patch_norm': True,
}

model = SwinUNet(config)

import torch.nn as nn

orig_forward = model.forward

def debug_forward(x):
    print(f"Input shape: {x.shape}")
    bottleneck, encoder_features = model.encoder(x)
    print(f"Bottleneck shape: {bottleneck.shape}")
    
    B, L, C = bottleneck.shape
    H = W = int(L ** 0.5)
    bottleneck_spatial = bottleneck.transpose(1, 2).contiguous().view(B, C, H, W)
    print(f"Bottleneck spatial shape: {bottleneck_spatial.shape}")
    
    x = bottleneck_spatial
    
    for i, decoder_block in enumerate(model.decoder_blocks):
        print(f"\nDecoder block {i}:")
        print(f"  Input shape: {x.shape}")
        x = decoder_block[0](x)  
        print(f"  After ConvTranspose: {x.shape}")
        x = decoder_block[1](x)  
        x = decoder_block[2](x) 
        
        skip_idx = len(encoder_features) - 2 - i
        print(f"  Skip index: {skip_idx}")
        if skip_idx >= 0:
            skip_feat, skip_H, skip_W = encoder_features[skip_idx]
            print(f"  Skip feat original: {skip_feat.shape}")
            print(f"  Skip H, W from encoder: {skip_H}, {skip_W}")
            
            B_skip, L_skip, C_skip = skip_feat.shape
            H_correct = W_correct = int(L_skip ** 0.5)
            print(f"  Calculated H, W from L={L_skip}: {H_correct}, {W_correct}")
            
            skip_feat = skip_feat.transpose(1, 2).contiguous().view(B, C_skip, H_correct, W_correct)
            print(f"  Skip feat spatial: {skip_feat.shape}")
            
            if x.shape[2:] != skip_feat.shape[2:]:
                skip_feat = nn.functional.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                print(f"  After interpolate: {skip_feat.shape}")
            
            concat = torch.cat([x, skip_feat], dim=1)
            print(f"  Concatenated shape: {concat.shape}")
            print(f"  DecoderBlock expects: in={decoder_block[3].conv1.in_channels}, out={decoder_block[3].conv1.out_channels}")
        else:
            print(f"  No skip connection")
        
        break  # Stop after first to see the problem
    
    return None

try:
    x = torch.randn(1, 1, 1024, 1024)
    debug_forward(x)
except Exception as e:
    print(f"\nError: {e}")
