"""
Swin-UNet: timm Swin-Tiny encoder + UNet-style decoder with skip connections.
2-channel input adapter: average pretrained 3-channel patch embed across input dim.
Note: timm Swin features_only returns (B, H, W, C) — we permute to (B, C, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(out_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    def __init__(self, backbone: str = "swin_tiny_patch4_window7_224",
                 in_channels: int = 2, num_classes: int = 1, img_size: int = 256):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            img_size=img_size,
            out_indices=(0, 1, 2, 3),
        )

        # Adapt first patch embed from 3-channel to in_channels
        patch_embed = self.encoder.patch_embed
        old_proj = patch_embed.proj
        new_weight = old_proj.weight.data.mean(dim=1, keepdim=True).expand(-1, in_channels, -1, -1)
        new_proj = nn.Conv2d(
            in_channels, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        new_proj.weight = nn.Parameter(new_weight.contiguous())
        if old_proj.bias is not None:
            new_proj.bias = nn.Parameter(old_proj.bias.data.clone())
        patch_embed.proj = new_proj

        # Probe a dummy forward to learn the feature layout and channel sizes
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            feats = self.encoder(dummy)

        # Determine layout: prefer timm's explicit metadata (timm>=0.9);
        # fall back to an empirical channels-last probe for older timm
        # (NHWC means the last dim — channels — is larger than dim 1, a
        # spatial dim).
        if hasattr(self.encoder, "output_fmt"):
            self._needs_permute = "NHWC" in str(self.encoder.output_fmt)
        else:
            self._needs_permute = feats[0].shape[-1] > feats[0].shape[1]

        ch = [f.shape[3] if self._needs_permute else f.shape[1] for f in feats]
        print(f"Swin feature channels: {ch}")

        # Decoder: up from deepest to shallowest
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])

        # Swin stage 0 has stride=4 relative to input; ConvTranspose2d(stride=4) to reach full res
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[0] // 2, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, num_classes, kernel_size=1),
        )
        self.img_size = img_size

    def _to_bchw(self, feat: torch.Tensor) -> torch.Tensor:
        if self._needs_permute:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_feats = self.encoder(x)
        feats = [self._to_bchw(f) for f in raw_feats]
        f0, f1, f2, f3 = feats

        d = self.up3(f3, f2)
        d = self.up2(d, f1)
        d = self.up1(d, f0)
        out = self.final_up(d)
        # Ensure exact output size
        if out.shape[-2:] != (self.img_size, self.img_size):
            out = F.interpolate(out, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
        return out  # (B, 1, H, W)
