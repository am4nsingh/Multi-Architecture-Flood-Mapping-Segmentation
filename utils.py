"""Shared helpers used by train.py, evaluate.py, and predict.py."""

from pathlib import Path

import yaml
import torch.nn as nn


def load_config(path: str) -> dict:
    """Load a YAML config, deep-merging config/base.yaml when 'defaults' is set."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        base_path = Path(path).parent / "base.yaml"
        with open(base_path) as f:
            base = yaml.safe_load(f)

        def merge(a, b):
            for k, v in b.items():
                if k in a and isinstance(a[k], dict) and isinstance(v, dict):
                    merge(a[k], v)
                elif k not in a:
                    a[k] = v
            return a

        cfg = merge(cfg, base)
        del cfg["defaults"]
    return cfg


def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model named in cfg['model']['name']."""
    name = cfg["model"]["name"]
    if name == "segformer":
        from models.segformer import SegFormerFlood
        return SegFormerFlood(
            backbone=cfg["model"]["backbone"],
            in_channels=cfg["model"]["in_channels"],
        )
    elif name == "swin_unet":
        from models.swin_unet import SwinUNet
        return SwinUNet(
            backbone=cfg["model"]["backbone"],
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
            img_size=cfg["model"]["img_size"],
        )
    elif name == "transunet":
        from models.transunet import TransUNet
        return TransUNet(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
            img_size=cfg["model"]["img_size"],
            vit_grid=cfg["model"]["vit_patches"],
            vit_layers=cfg["model"]["vit_layers"],
            vit_heads=cfg["model"]["vit_heads"],
            vit_hidden=cfg["model"]["vit_hidden"],
        )
    else:
        raise ValueError(f"Unknown model: {name}")
