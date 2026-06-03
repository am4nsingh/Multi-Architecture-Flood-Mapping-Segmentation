"""
Inference on a folder of (before, after) tile pairs.
Usage: python predict.py --config config/segformer.yaml --checkpoint checkpoints/segformer_best.pt
                         --before <dir> --after <dir> --out <dir>
Tile pairs are matched by filename stem (excluding the role token).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tifffile

from utils import load_config, build_model

ROOT = Path(__file__).resolve().parent.parent


def normalize(x: np.ndarray, stats: dict, channel: str) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0, posinf=0, neginf=0)
    x = np.clip(x, 0, None)
    q99 = stats[channel]["q99"]
    mean = stats[channel]["mean"]
    std = stats[channel]["std"]
    x = np.log1p(np.clip(x, 0, q99))
    if std > 0:
        x = (x - mean) / std
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stats_json = str(ROOT / cfg["data"]["stats_json"])
    with open(stats_json) as f:
        stats = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    before_dir = Path(args.before)
    after_dir = Path(args.after)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    before_files = {p.stem: p for p in before_dir.glob("*.tif")}
    after_files = {p.stem: p for p in after_dir.glob("*.tif")}
    common = sorted(set(before_files) & set(after_files))
    print(f"Found {len(common)} matched pairs")

    with torch.no_grad():
        for stem in common:
            b = normalize(tifffile.imread(str(before_files[stem])), stats, "before")
            a = normalize(tifffile.imread(str(after_files[stem])), stats, "after")
            image = torch.from_numpy(np.stack([b, a], axis=0)).unsqueeze(0).to(device)
            logits = model(image)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob > args.threshold).astype(np.uint8) * 255
            tifffile.imwrite(str(out_dir / f"{stem}_pred.tif"), pred)

    print(f"Predictions written to {out_dir}/")


if __name__ == "__main__":
    main()
