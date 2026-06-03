"""
Evaluate a trained model on the frozen test split.
Usage: python evaluate.py --config config/segformer.yaml --checkpoint checkpoints/segformer_best.pt
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader

from utils import load_config, build_model

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_csv = str(ROOT / cfg["data"]["splits_csv"])
    stats_json = str(ROOT / cfg["data"]["stats_json"])
    with open(stats_json) as f:
        stats = json.load(f)

    from data.dataset import FloodSARDataset
    from metrics import MetricAccumulator

    test_ds = FloodSARDataset(splits_csv, "test", stats=stats, drop_nodata=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pred_dir = ROOT / cfg["eval"]["pred_dir"] / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [0.3, 0.4, 0.5, 0.6]
    accs = {t: MetricAccumulator() for t in thresholds}

    tile_idx = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)

            for t in thresholds:
                accs[t].update(logits, labels, threshold=t)

            # Save predictions at default threshold
            preds = (probs > args.threshold).cpu().numpy().astype(np.uint8) * 255

            for b in range(preds.shape[0]):
                tifffile.imwrite(str(pred_dir / f"pred_{tile_idx:05d}.tif"), preds[b])
                tile_idx += 1

    # Report metrics
    report_dir = ROOT / cfg["eval"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for t in thresholds:
        results[str(t)] = accs[t].compute()

    out_json = report_dir / f"{model_name}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Thresh':>6}  {'mIoU':>8}  {'mDICE':>8}  {'flood_IoU':>10}  {'flood_F1':>9}")
    for t in thresholds:
        m = results[str(t)]
        mark = " ←" if t == args.threshold else ""
        print(f"  {t:.1f}   {m['mIoU']:.4f}   {m['mDICE']:.4f}   {m['flood_IoU']:.4f}      {m['flood_F1']:.4f}{mark}")

    print(f"\nMetrics saved to {out_json}")
    print(f"Predictions saved to {pred_dir}/")


if __name__ == "__main__":
    main()
