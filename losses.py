"""BCE + Dice combined loss for binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)  # (B, 1, H, W)
        B = probs.shape[0]
        probs = probs.view(B, -1)      # (B, H*W)
        targets = targets.view(B, -1)  # (B, H*W)
        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1, H, W), targets: (B, H, W)
        targets_4d = targets.unsqueeze(1)
        bce_loss = self.bce(logits, targets_4d)
        dice_loss = self.dice(logits, targets_4d)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
