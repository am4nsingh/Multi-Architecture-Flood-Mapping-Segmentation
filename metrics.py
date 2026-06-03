"""
Binary segmentation metrics: mIoU, mDICE, per-class IoU/F1/Precision/Recall.
All functions accept torch tensors or numpy arrays; work batch-wise.
"""

import torch
import numpy as np


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def iou_per_class(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
    p = pred == cls
    t = target == cls
    inter = (p & t).sum()
    union = (p | t).sum()
    return float(inter) / float(union + 1e-6)


def mean_iou(pred: np.ndarray, target: np.ndarray) -> float:
    return (iou_per_class(pred, target, 0) + iou_per_class(pred, target, 1)) / 2.0


def dice_per_class(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
    p = pred == cls
    t = target == cls
    inter = (p & t).sum()
    return float(2 * inter) / float(p.sum() + t.sum() + 1e-6)


def mean_dice(pred: np.ndarray, target: np.ndarray) -> float:
    return (dice_per_class(pred, target, 0) + dice_per_class(pred, target, 1)) / 2.0


def per_class_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    results = {}
    for cls, name in [(0, "background"), (1, "flood")]:
        tp = ((pred == cls) & (target == cls)).sum()
        fp = ((pred == cls) & (target != cls)).sum()
        fn = ((pred != cls) & (target == cls)).sum()
        if tp + fp + fn == 0:
            # class absent in both prediction and target — undefined, skip it
            results[name] = None
            continue
        precision = float(tp) / float(tp + fp + 1e-6)
        recall = float(tp) / float(tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = float(tp) / float(tp + fp + fn + 1e-6)
        results[name] = {
            "IoU": iou, "F1": f1,
            "Precision": precision, "Recall": recall,
        }
    return results


class MetricAccumulator:
    """Accumulate per-batch metrics and compute dataset-level averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._iou = []
        self._dice = []
        self._flood_iou = []
        self._flood_f1 = []
        self._flood_precision = []
        self._flood_recall = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)
        preds = (probs > threshold).long()
        targets = targets.long()

        for b in range(preds.shape[0]):
            p = _to_numpy(preds[b])
            t = _to_numpy(targets[b])
            self._iou.append(mean_iou(p, t))
            self._dice.append(mean_dice(p, t))
            flood = per_class_metrics(p, t)["flood"]
            # Skip tiles where the flood class is absent (would otherwise
            # contribute a spurious IoU=0 and deflate the average)
            if flood is not None:
                self._flood_iou.append(flood["IoU"])
                self._flood_f1.append(flood["F1"])
                self._flood_precision.append(flood["Precision"])
                self._flood_recall.append(flood["Recall"])

    @staticmethod
    def _mean(xs) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def compute(self) -> dict:
        return {
            "mIoU": self._mean(self._iou),
            "mDICE": self._mean(self._dice),
            "flood_IoU": self._mean(self._flood_iou),
            "flood_F1": self._mean(self._flood_f1),
            "flood_Precision": self._mean(self._flood_precision),
            "flood_Recall": self._mean(self._flood_recall),
        }
