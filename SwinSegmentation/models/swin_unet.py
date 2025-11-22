"""
Swin-UNet: Swin Transformer based U-Net for Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_transformer import SwinTransformerEncoder


class PatchExpanding(nn.Module):
    """Patch Expanding Layer - opposite of Patch Merging"""
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        x = x.view(B, H, W, C * 2)

        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)

        return x


class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.has_skip = skip_channels > 0

    def forward(self, x, skip=None):
        if skip is not None and self.has_skip:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SwinUNet(nn.Module):
    """
    Swin-UNet: Swin Transformer based segmentation network
    
    Args:
        config: Configuration dictionary containing model parameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_classes = config.get('num_classes', 4)
        img_size = config.get('img_size', 224)
        patch_size = config.get('patch_size', 4)
        in_chans = config.get('in_chans', 1)
        embed_dim = config.get('embed_dim', 96)
        depths = config.get('depths', [2, 2, 6, 2])
        num_heads = config.get('num_heads', [3, 6, 12, 24])
        window_size = config.get('window_size', 7)
        mlp_ratio = config.get('mlp_ratio', 4.)
        qkv_bias = config.get('qkv_bias', True)
        drop_rate = config.get('drop_rate', 0.)
        drop_path_rate = config.get('drop_path_rate', 0.2)
        ape = config.get('ape', False)
        patch_norm = config.get('patch_norm', True)
        
        self.encoder = SwinTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=ape,
            patch_norm=patch_norm
        )
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(self.num_layers - 1):
            in_channels = int(embed_dim * 2 ** (self.num_layers - 1 - i))
            skip_channels = int(embed_dim * 2 ** (self.num_layers - 2 - i))
            out_channels = skip_channels
            
            self.decoder_blocks.append(
                nn.ModuleList([
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    DecoderBlock(out_channels, skip_channels, out_channels)
                ])
            )
        
        final_channels = embed_dim
        
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(final_channels, final_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(final_channels // 2, final_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_channels // 4),
            nn.ReLU(inplace=True),
        )
        
        self.seg_head = nn.Conv2d(final_channels // 4, self.num_classes, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bottleneck, encoder_features = self.encoder(x)
        
        B, L, C = bottleneck.shape
        H = W = int(L ** 0.5)
        bottleneck = bottleneck.transpose(1, 2).contiguous().view(B, C, H, W)
        
        x = bottleneck
        
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block[0](x)  # Upsample (bilinear)
            x = decoder_block[1](x)  # Conv2d
            x = decoder_block[2](x)  # BatchNorm
            x = decoder_block[3](x)  # ReLU
            
            skip_idx = len(encoder_features) - 2 - i
            if skip_idx >= 0:
                skip_feat, skip_H, skip_W = encoder_features[skip_idx]
                B_skip, L_skip, C_skip = skip_feat.shape
                H_skip = W_skip = int(L_skip ** 0.5)
                skip_feat = skip_feat.transpose(1, 2).contiguous().view(B_skip, C_skip, H_skip, W_skip)
                
                if x.shape[2:] != skip_feat.shape[2:]:
                    skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                
                x = decoder_block[4](x, skip_feat)  # DecoderBlock is now at index 4
            else:
                x = decoder_block[4](x, None)
        
        x = self.final_upsample(x)
        
        x = self.seg_head(x)
        
        return x


def build_swin_unet(config):
    """
    Build Swin-UNet model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: SwinUNet model
    """
    model = SwinUNet(config)
    return model


if __name__ == "__main__":
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
    
    model = build_swin_unet(config)
    
    x = torch.randn(1, 1, 1024, 1024)
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
