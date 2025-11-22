"""
Debug script to check encoder output dimensions
"""
import torch
import sys
sys.path.append('.')

from models.swin_transformer import SwinTransformerEncoder

config = {
    'img_size': 1024,
    'patch_size': 4,
    'in_chans': 1,
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

encoder = SwinTransformerEncoder(**config)

x = torch.randn(1, 1, 1024, 1024)
bottleneck, features = encoder(x)

print("Bottleneck shape:", bottleneck.shape)
print("\nEncoder features:")
for i, (feat, H, W) in enumerate(features):
    print(f"  Level {i}: Feature shape={feat.shape}, H={H}, W={W}, Channels={feat.shape[-1]}")
