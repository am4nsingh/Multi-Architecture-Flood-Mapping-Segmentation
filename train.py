"""
Entry point: python train.py --config config/segformer.yaml
Trains the specified model; saves best-val and last checkpoints.
"""

import argparse
import csv
import datetime
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import yaml

from utils import load_config, build_model

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic so runs with the same seed are reproducible
    # (otherwise the experiment tracking / config hashing is cosmetic).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(splits_csv: str) -> float:
    import tifffile
    pos = neg = 0
    with open(splits_csv) as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                continue
            mask = tifffile.imread(r["mask"])
            u = set(np.unique(mask).tolist())
            if u <= {0, 1, 2}:
                label = np.where(mask == 2, 1, 0)
            else:
                label = (mask > 0).astype(np.uint8)
            pos += label.sum()
            neg += (label == 0).sum()
    w = neg / max(pos, 1)
    print(f"pos_weight: {w:.2f}  (neg={neg}, pos={pos})")
    return float(w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to resume from. If omitted, the run "
                             "auto-resumes from <model>_last.pt when it exists.")
    parser.add_argument("--fresh", action="store_true",
                        help="Start from scratch even if a checkpoint exists "
                             "(existing checkpoints are archived first, never overwritten).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]

    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {model_name}")

    # ── Data ──────────────────────────────────────────────────────────────────
    splits_csv = str(ROOT / cfg["data"]["splits_csv"])
    stats_json = str(ROOT / cfg["data"]["stats_json"])

    if not Path(stats_json).exists():
        from data.dataset import compute_train_stats
        stats = compute_train_stats(splits_csv, stats_json)
    else:
        with open(stats_json) as f:
            stats = json.load(f)

    from data.dataset import FloodSARDataset
    from data.transforms import train_transform

    train_ds = FloodSARDataset(splits_csv, "train", stats=stats, transform=train_transform())
    val_ds = FloodSARDataset(splits_csv, "val", stats=stats)

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    from losses import CombinedLoss
    if "pos_weight" in stats:
        pw = stats["pos_weight"]
    else:
        pw = compute_pos_weight(splits_csv)
        stats["pos_weight"] = pw
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2)
    pos_weight = torch.tensor([pw], device=device)
    criterion = CombinedLoss(
        bce_weight=cfg["loss"]["bce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        pos_weight=pos_weight,
    )

    # ── Optimiser & schedule ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])

    epochs = cfg["training"]["epochs"]
    steps = len(train_loader) * epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg["training"]["lr"],
        total_steps=steps,
        pct_start=cfg["training"]["warmup_frac"],
        anneal_strategy="cos",
    )

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp = cfg["training"]["amp"] and device.type == "cuda"
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Logging ───────────────────────────────────────────────────────────────
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    log_dir = ROOT / cfg["training"]["log_dir"] / model_name / f"{run_id}_{cfg_hash}"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    with open(log_dir / "environment.json", "w") as f:
        json.dump({"torch": torch.__version__, "cuda": str(torch.version.cuda),
                    "seed": cfg["training"]["seed"]}, f, indent=2)
    writer = SummaryWriter(log_dir)
    csv_log = open(log_dir / "metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_log)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_mDICE",
                         "val_flood_IoU", "val_flood_F1", "val_flood_Precision", "val_flood_Recall"])

    ckpt_dir = ROOT / cfg["training"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou = 0.0
    patience_count = 0

    last_ckpt = ckpt_dir / f"{model_name}_last.pt"

    # Decide the checkpoint to resume from:
    #  - explicit --resume PATH always wins
    #  - otherwise auto-resume from <model>_last.pt unless --fresh was passed
    resume_path = args.resume
    if resume_path is None and not args.fresh and last_ckpt.exists():
        resume_path = str(last_ckpt)
        print(f"Found {last_ckpt.name} — auto-resuming (pass --fresh to start over).")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}  (best_miou={best_miou:.4f})")
    elif args.fresh and any(ckpt_dir.glob(f"{model_name}_*.pt")):
        # Never silently destroy prior weights: archive them before overwriting
        import shutil
        backup = ckpt_dir / f"_backup_{model_name}_{run_id}"
        backup.mkdir(parents=True, exist_ok=True)
        for p in ckpt_dir.glob(f"{model_name}_*.pt"):
            shutil.move(str(p), str(backup / p.name))
        print(f"--fresh: archived existing {model_name} checkpoints to {backup.name}/")

    # ── Grad accumulation for TransUNet ───────────────────────────────────────
    accum_steps = cfg["training"].get("grad_accum", 1)

    from metrics import MetricAccumulator

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                logits = model(images)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * accum_steps

        train_loss /= len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        acc = MetricAccumulator()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                    logits = model(images)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                acc.update(logits, labels)

        val_loss /= len(val_loader)
        m = acc.compute()

        print(f"Epoch {epoch+1:03d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"mIoU={m['mIoU']:.4f}  mDICE={m['mDICE']:.4f}  "
              f"flood_IoU={m['flood_IoU']:.4f}  flood_F1={m['flood_F1']:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for k, v in m.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

        csv_writer.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{m['mIoU']:.6f}", f"{m['mDICE']:.6f}",
                             f"{m['flood_IoU']:.6f}", f"{m['flood_F1']:.6f}",
                             f"{m['flood_Precision']:.6f}", f"{m['flood_Recall']:.6f}"])
        csv_log.flush()

        # ── Checkpoints ───────────────────────────────────────────────────────
        # Update best_miou FIRST so state dict contains the correct value
        is_best = False
        if m["mIoU"] > best_miou:
            best_miou = m["mIoU"]
            patience_count = 0
            is_best = True
        else:
            patience_count += 1

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": cfg,
        }
        torch.save(state, ckpt_dir / f"{model_name}_last.pt")
        if (epoch + 1) % 2 == 0:
            torch.save(state, ckpt_dir / f"{model_name}_epoch{epoch+1:02d}.pt")

        if is_best:
            torch.save(state, ckpt_dir / f"{model_name}_best.pt")
            print(f"  → New best mIoU: {best_miou:.4f}")
        elif patience_count >= cfg["training"]["early_stop_patience"]:
            print(f"Early stop at epoch {epoch+1}")
            break

    csv_log.close()
    writer.close()
    print(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
