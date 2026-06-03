"""
TransUNet: ResNet50-V2 CNN stem + ViT bottleneck + cascaded upsampling decoder.
2-channel input; 1-channel logit output.
Reference: Chen et al., TransUNet (2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─── CNN stem (ResNet50 first 3 stages, pretrained) ─────────────────────────

class CNNStem(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        resnet = timm.create_model("resnet50", pretrained=True, features_only=True,
                                   out_indices=(0, 1, 2))
        # Adapt layer0 conv1 to in_channels
        old_conv = resnet.conv1
        new_weight = old_conv.weight.data.mean(dim=1, keepdim=True).expand(-1, in_channels, -1, -1)
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        new_conv.weight = nn.Parameter(new_weight.contiguous())
        resnet.conv1 = new_conv
        self.encoder = resnet
        # channel sizes for stages 0,1,2: 64, 256, 512
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            feats = self.encoder(dummy)
        self.skip_channels = [f.shape[1] for f in feats]  # e.g. [64, 256, 512]
        self.out_channels = self.skip_channels[-1]

    def forward(self, x):
        feats = self.encoder(x)
        return feats   # list: [s0, s1, s2]


# ─── ViT bottleneck ──────────────────────────────────────────────────────────

class ViTBottleneck(nn.Module):
    def __init__(self, in_channels: int, grid_size: int = 16,
                 num_layers: int = 12, num_heads: int = 12, hidden_dim: int = 768,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.grid = grid_size
        self.hidden = hidden_dim
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        seq_len = grid_size * grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_back = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H/16, W/16) — we interpolate to grid×grid if needed
        B = x.shape[0]
        x = F.interpolate(x, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        x = tokens.transpose(1, 2).reshape(B, self.hidden, self.grid, self.grid)
        return self.proj_back(x)   # (B, C, grid, grid)


# ─── Decoder ─────────────────────────────────────────────────────────────────

class DecodeBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─── No-skip decoder block ────────────────────────────────────────────────────

class NoSkipDecodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


# ─── Full TransUNet ───────────────────────────────────────────────────────────

class TransUNet(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 1, img_size: int = 256,
                 vit_grid: int = 16, vit_layers: int = 12, vit_heads: int = 12,
                 vit_hidden: int = 768):
        super().__init__()
        self.stem = CNNStem(in_channels)
        bottleneck_ch = self.stem.out_channels
        skip_chs = self.stem.skip_channels

        self.vit = ViTBottleneck(bottleneck_ch, grid_size=vit_grid,
                                 num_layers=vit_layers, num_heads=vit_heads,
                                 hidden_dim=vit_hidden)

        self.dec2 = DecodeBlock(bottleneck_ch, skip_chs[1], 256)
        self.dec1 = DecodeBlock(256, skip_chs[0], 128)
        self.dec0 = NoSkipDecodeBlock(128, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0, s1, s2 = self.stem(x)
        bottleneck = self.vit(s2)
        d = self.dec2(bottleneck, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d)
        out = self.final(d)
        # Ensure output matches input resolution
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
