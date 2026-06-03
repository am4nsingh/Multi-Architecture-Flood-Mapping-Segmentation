"""
SegFormer with 2-channel SAR input.
Backbone: nvidia/mit-b2 (HuggingFace transformers).
The 3-channel patch-embed conv is replaced by averaging weights across the
channel dim and duplicating to 2 channels.
"""

import logging
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def _adapt_patch_embed(conv: nn.Conv2d, in_channels: int = 2) -> nn.Conv2d:
    """Replace a pretrained Conv2d(C_in_old, ...) with Conv2d(in_channels, ...)."""
    old_weight = conv.weight.data  # (C_out, C_in_old, kH, kW)
    new_weight = old_weight.mean(dim=1, keepdim=True).expand(-1, in_channels, -1, -1)
    new_conv = nn.Conv2d(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    new_conv.weight = nn.Parameter(new_weight.contiguous())
    if conv.bias is not None:
        new_conv.bias = nn.Parameter(conv.bias.data.clone())
    return new_conv


class SegFormerFlood(nn.Module):
    def __init__(self, backbone: str = "nvidia/mit-b2", in_channels: int = 2):
        super().__init__()
        config = SegformerConfig.from_pretrained(backbone, num_labels=1, ignore_mismatched_sizes=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone, config=config, ignore_mismatched_sizes=True
        )
        # Replace first patch embed conv (3 → in_channels)
        first_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = _adapt_patch_embed(first_conv, in_channels)

        # Replace the decode head classifier to output 1 channel
        decode_head = self.model.decode_head
        old_cls = decode_head.classifier
        decode_head.classifier = nn.Conv2d(old_cls.in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, H, W)
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, 1, H/4, W/4)
        # Upsample to input resolution
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits  # (B, 1, H, W)
