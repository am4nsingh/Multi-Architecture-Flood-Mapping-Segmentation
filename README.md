# Multi-Architecture Flood Mapping Segmentation

Binary semantic segmentation of flooded regions from Sentinel-1 SAR before/after image pairs, comparing three transformer-based architectures.

## Overview

This project implements a complete deep learning pipeline for satellite-based flood extent mapping. Given pre-event and post-event Synthetic Aperture Radar (SAR) imagery, the models produce pixel-level binary flood masks.

## Architectures

| Model | Backbone | Parameters | Flood IoU | Flood F1 |
|---|---|---|---|---|
| **Swin-UNet** | `swin_tiny_patch4_window7_224` + UNet decoder | ~30M | **0.892** | **0.943** |
| **TransUNet** | ResNet-50 + 12-layer ViT + CNN decoder | ~90M | 0.892 | 0.942 |
| SegFormer | `nvidia/mit-b2` + MLP decode head | ~28M | 0.530 | 0.692 |

All models accept **2-channel** input (before/after SAR) and output a single binary logit map. Metrics reported at threshold 0.5 on the held-out test set.

## Project Structure

```
.
├── config/                 # Training configurations (YAML)
│   ├── base.yaml           # Shared defaults
│   ├── segformer.yaml
│   ├── swin_unet.yaml
│   └── transunet.yaml
├── data/                   # Data pipeline
│   ├── build_index.py      # Build triplet index (before + after + mask)
│   ├── dataset.py          # PyTorch dataset with SAR normalization
│   ├── make_splits.py      # Spatial train/val/test splits
│   └── transforms.py       # Augmentation transforms
├── models/                 # Architecture definitions
│   ├── segformer.py
│   ├── swin_unet.py
│   └── transunet.py
├── reports/                # Per-model evaluation metrics (JSON)
├── runs/                   # Experiment logs (configs, metrics per epoch)
├── train.py                # Training loop
├── evaluate.py             # Test-set evaluation with threshold sweep
├── predict.py              # Inference on new tiles
├── losses.py               # BCE + Dice compound loss
├── metrics.py              # mIoU, mDICE, SSIM
├── utils.py                # Logging and checkpoint utilities
├── requirements.txt
├── splits.csv              # Frozen train/val/test split assignments
└── triplets.csv            # Before-after-mask file triplets
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Pipeline

Run once before training:

```bash
# 1. Build triplet index (before + after + mask matched by city, x, y)
python data/build_index.py

# 2. Create spatial train/val/test splits
python data/make_splits.py

# 3. Compute per-channel SAR statistics (cached automatically)
python data/dataset.py
```

## Training

```bash
python train.py --config config/segformer.yaml
python train.py --config config/swin_unet.yaml
python train.py --config config/transunet.yaml
```

Checkpoints are saved to `checkpoints/` and TensorBoard logs to `runs/`.

## Evaluation

```bash
python evaluate.py --config config/swin_unet.yaml --checkpoint checkpoints/swin_unet_best.pt
```

Outputs per-tile metrics and a threshold sweep to `reports/`.

## Inference

```bash
python predict.py \
  --config config/swin_unet.yaml \
  --checkpoint checkpoints/swin_unet_best.pt \
  --before /path/to/before_tiles/ \
  --after  /path/to/after_tiles/ \
  --out    /path/to/output_masks/
```

## Key Design Decisions

- **Spatial splits** &mdash; Tile origins split by y-coordinate strips with a 256 px buffer to prevent train/test overlap leakage.
- **2-channel adapter** &mdash; Pretrained 3-channel patch-embed weights are averaged across the channel dimension and transferred to a new 2-channel convolution.
- **SAR normalization** &mdash; `clip(0, q99) -> log1p -> z-score`, statistics computed on the train set only.
- **Compound loss** &mdash; `0.5 * BCE(pos_weight=neg/pos) + 0.5 * Dice` to handle severe class imbalance.

## Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection-over-Union across flood and non-flood classes |
| **mDICE** | Mean Dice coefficient across both classes |
| **Flood IoU** | IoU for the flood class specifically |
| **Flood F1** | F1 score for the flood class |
| **SSIM** | Structural similarity between predicted and ground-truth masks |
