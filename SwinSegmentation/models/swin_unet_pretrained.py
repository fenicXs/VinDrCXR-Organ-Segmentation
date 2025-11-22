"""Pretrained Swin-UNet integration using ImageNet weights.

This module adapts the official Swin-UNet implementation so that we can
initialise from the public `swin_tiny_patch4_window7_224.pth` checkpoint
and fine-tune on CheXmask.  The code is a carefully trimmed copy of the
reference implementation with small changes to:

* avoid external dependencies (`einops`, `timm`)
* support single-channel inputs by internally repeating them to 3 channels
* expose helper methods to freeze/unfreeze the encoder during warm-up
* skip loading unmatched decoder head weights when transferring the encoder
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn.init import trunc_normal_


def to_2tuple(value):
    """Convert an int or iterable to a tuple of length 2."""
    if isinstance(value, tuple):
        return value
    return (value, value)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):  # type: ignore[override]
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _rearrange_expand(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Utility that mimics einops.rearrange used in the original repo."""
    B, H, W, C = x.shape
    new_c = C // (scale * scale)
    x = x.view(B, H, W, scale, scale, new_c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H * scale, W * scale, new_c)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # type: ignore[override]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):  # type: ignore[override]
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(self.window_size[0] * self.window_size[1],
                                 self.window_size[0] * self.window_size[1], -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):  # type: ignore[override]
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
        self.scale = dim_scale

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        x = _rearrange_expand(x, self.scale)
        x = x.view(B, -1, C // (self.scale * self.scale))
        x = self.norm(x)
        return x


class FinalPatchExpandX4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        x = _rearrange_expand(x, self.dim_scale)
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=4,
                 embed_dim=96, depths=(2, 2, 6, 2), depths_decoder=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, use_checkpoint=False, final_upsample="expand_first"):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer)
            else:
                idx = self.num_layers - 1 - i_layer
                layer_up = BasicLayerUp(
                    dim=int(embed_dim * 2 ** idx),
                    input_resolution=(patches_resolution[0] // (2 ** idx),
                                      patches_resolution[1] // (2 ** idx)),
                    depth=depths_decoder[idx],
                    num_heads=num_heads[idx],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:idx]):sum(depths[:idx + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpandX4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=4,
                dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        for idx, layer_up in enumerate(self.layers_up):
            if idx == 0:
                x = layer_up(x)
            else:
                skip = x_downsample[self.num_layers - 1 - idx]
                x = torch.cat([x, skip], dim=-1)
                x = self.concat_back_dim[idx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        return x

    def upsample_output(self, x):
        H, W = self.patches_resolution
        B, L, _ = x.shape
        assert L == H * W
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, H * 4, W * 4, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.output(x)
        return x

    def forward(self, x):  # type: ignore[override]
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.upsample_output(x)
        return x


class PretrainedSwinUNet(nn.Module):
    """Wrapper that handles grayscale inputs and pretrained weight loading."""

    def __init__(self, img_size=512, num_classes=4, in_chans=1, pretrained_path: str | None = None,
                 mlp_ratio=4.0, window_size=7, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), drop_path_rate=0.2, use_checkpoint=False,
                 normalize_input=True):
        super().__init__()
        self.in_chans = in_chans
        self.normalize_input = normalize_input
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=4,
            in_chans=3,  # keep RGB to remain compatible with pretrained weights
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=use_checkpoint,
            final_upsample="expand_first",
        )

        if pretrained_path and os.path.isfile(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

    def forward(self, x):  # type: ignore[override]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.normalize_input:
            x = (x - self.mean) / self.std
        return self.swin_unet(x)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        for name, param in self.swin_unet.named_parameters():
            if name.startswith("patch_embed") or name.startswith("layers") or name.startswith("absolute_pos_embed"):
                yield param

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        for name, param in self.swin_unet.named_parameters():
            if name.startswith("layers_up") or name.startswith("norm_up") or name.startswith("up") or name.startswith("output"):
                yield param

    def freeze_encoder(self):
        for param in self.encoder_parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder_parameters():
            param.requires_grad = True

    def _load_pretrained_weights(self, pretrained_path: str):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if "model" not in checkpoint:
            pretrained_dict = {k[17:]: v for k, v in checkpoint.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "output" not in k}
            self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            return

        pretrained_dict = checkpoint["model"]
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict[current_k] = v

        model_dict = self.swin_unet.state_dict()
        drop_keys = []
        for k, v in full_dict.items():
            if k in model_dict and full_dict[k].shape != model_dict[k].shape:
                drop_keys.append(k)
        for k in drop_keys:
            full_dict.pop(k, None)

        self.swin_unet.load_state_dict(full_dict, strict=False)


def build_pretrained_swin_unet(config_module) -> PretrainedSwinUNet:
    img_size = config_module.IMAGE_SIZE
    num_classes = config_module.NUM_CLASSES
    in_chans = config_module.SWIN_CONFIG.get('in_chans', 1)
    pretrained_path = getattr(config_module, 'PRETRAINED_MODEL_PATH', None)
    depths = tuple(config_module.SWIN_CONFIG.get('depths', [2, 2, 6, 2]))
    num_heads = tuple(config_module.SWIN_CONFIG.get('num_heads', [3, 6, 12, 24]))
    embed_dim = config_module.SWIN_CONFIG.get('embed_dim', 96)
    window_size = config_module.SWIN_CONFIG.get('window_size', 7)
    mlp_ratio = config_module.SWIN_CONFIG.get('mlp_ratio', 4.0)
    drop_path_rate = config_module.SWIN_CONFIG.get('drop_path_rate', 0.2)
    use_checkpoint = getattr(config_module, 'USE_GRADIENT_CHECKPOINTING', False)

    model = PretrainedSwinUNet(
        img_size=img_size,
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained_path=pretrained_path,
        mlp_ratio=mlp_ratio,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        normalize_input=getattr(config_module, 'PRETRAINED_NORMALIZE_INPUT', True),
    )
    if getattr(config_module, 'FREEZE_ENCODER_EPOCHS', 0) > 0:
        model.freeze_encoder()
    return model
