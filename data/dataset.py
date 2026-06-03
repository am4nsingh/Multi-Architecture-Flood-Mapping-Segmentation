"""
FloodSARDataset: loads (before, after, mask) triplets from splits.csv.
Returns (image[2,H,W], label[H,W]) float32 tensors.
"""

import json
import csv
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from pathlib import Path


def _remap_mask(mask: np.ndarray) -> np.ndarray:
    u = set(np.unique(mask).tolist())
    if u <= {0, 1, 2}:
        # tri-class: 0,1 → non-flood, 2 → flood
        return np.where(mask == 2, 1, 0).astype(np.uint8)
    else:
        return (mask > 0).astype(np.uint8)


def _load_sar(path: str) -> np.ndarray:
    img = tifffile.imread(path).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)
    return img


class FloodSARDataset(Dataset):
    def __init__(self, splits_csv: str, split: str, stats: dict | None = None,
                 transform=None, drop_nodata: bool = True):
        self.split = split
        self.transform = transform

        rows = []
        with open(splits_csv) as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    rows.append(r)

        if drop_nodata and split == "train":
            nodata = self._nodata_index(splits_csv, rows)
            before_count = len(rows)
            rows = [r for r in rows if (r["city"], r["x"], r["y"]) not in nodata]
            dropped = before_count - len(rows)
            if dropped:
                print(f"  Dropped {dropped} nodata tiles from {split}")

        self.rows = rows
        self.stats = stats

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _nodata_index(splits_csv: str, rows: list) -> set:
        """Set of (city, x, y) keys for pure-nodata tiles.

        Built once by scanning the tiles, then cached in a sidecar JSON next to
        splits.csv so subsequent runs skip the (expensive) repeated full scan.
        The sidecar is removed by make_splits.py whenever the split is
        regenerated, forcing a rebuild against the new partition.
        """
        index_path = Path(splits_csv).with_name("nodata_index.json")
        if index_path.exists():
            with open(index_path) as f:
                return {tuple(k) for k in json.load(f)}

        keys = []
        for r in rows:
            b = tifffile.imread(r["before"])
            if b.max() == 0 and tifffile.imread(r["after"]).max() == 0:
                keys.append([r["city"], r["x"], r["y"]])
        with open(index_path, "w") as f:
            json.dump(keys, f)
        print(f"  Built nodata index: {len(keys)} nodata tiles → {index_path.name}")
        return {tuple(k) for k in keys}

    def __getitem__(self, idx):
        r = self.rows[idx]
        before = _load_sar(r["before"])
        after = _load_sar(r["after"])
        mask = tifffile.imread(r["mask"])

        label = _remap_mask(mask).astype(np.float32)

        image = np.stack([before, after], axis=0)  # (2, H, W) RAW amplitudes

        # Augment BEFORE normalization so speckle noise operates in amplitude domain
        if self.transform is not None:
            image, label = self.transform(image, label)

        # Normalize AFTER augmentation: clip → log1p → z-score
        image[0] = self._normalize(image[0], "before")
        image[1] = self._normalize(image[1], "after")

        return (
            torch.from_numpy(image),
            torch.from_numpy(label),
        )

    def _normalize(self, x: np.ndarray, channel: str) -> np.ndarray:
        if self.stats and channel in self.stats:
            q99 = self.stats[channel]["q99"]
            mean = self.stats[channel]["mean"]
            std = self.stats[channel]["std"]
        else:
            q99 = float(np.percentile(x, 99)) if x.max() > 0 else 1.0
            mean, std = 0.0, 1.0

        x = np.log1p(np.clip(x, 0, q99))
        if std > 0:
            x = (x - mean) / std
        return x


def compute_train_stats(splits_csv: str, out_json: str, n_sample: int = 2000):
    """Compute per-channel q99/mean/std on a random sample of train tiles."""
    import random

    rows = []
    with open(splits_csv) as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                rows.append(r)

    rng = random.Random(1337)
    sample = rng.sample(rows, min(n_sample, len(rows)))

    channels = {"before": [], "after": []}
    for r in sample:
        for ch in ("before", "after"):
            img = _load_sar(r[ch])
            channels[ch].append(img.ravel())

    stats = {}
    for ch, arrays in channels.items():
        flat = np.concatenate(arrays)
        q99 = float(np.percentile(flat, 99))
        logged = np.log1p(np.clip(flat, 0, q99))
        stats[ch] = {
            "q99": q99,
            "mean": float(logged.mean()),
            "std": float(logged.std()),
        }
        print(f"  {ch}: q99={q99:.4f}  mean={stats[ch]['mean']:.4f}  std={stats[ch]['std']:.4f}")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to {out_json}")
    return stats


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    splits_csv = str(ROOT / "flood_mapping" / "splits.csv")
    out_json = str(ROOT / "flood_mapping" / "data" / "train_stats.json")
    compute_train_stats(splits_csv, out_json)
