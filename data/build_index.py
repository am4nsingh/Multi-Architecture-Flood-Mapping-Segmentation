"""
Build triplet manifest: match before/after images with masks by (city, x, y).
Outputs: flood_mapping/triplets.csv
"""

import os
import re
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEFORE_DIR = ROOT / "images" / "before"
AFTER_DIR = ROOT / "images" / "after"
MASKS_DIR = ROOT / "masks"
OUT_CSV = ROOT / "flood_mapping" / "triplets.csv"

# Filename pattern: <city>_<date>_<role>_<x>_<y>.tif
PATTERN = re.compile(r"^(.+?)_(\d{4}-\d{2}-\d{2})_(before|after|mask)_(\d+)_(\d+)\.tif$")


def parse_files(directory: Path, role: str) -> dict:
    index = {}
    for p in directory.glob("*.tif"):
        m = PATTERN.match(p.name)
        if not m:
            continue
        city, date, r, x, y = m.groups()
        if r != role:
            continue
        if p.stat().st_size == 0:
            continue
        index[(city, x, y)] = str(p)
    return index


def main():
    print("Scanning directories...")
    before = parse_files(BEFORE_DIR, "before")
    after = parse_files(AFTER_DIR, "after")
    masks = parse_files(MASKS_DIR, "mask")

    print(f"  before: {len(before)}  after: {len(after)}  masks: {len(masks)}")

    keys = set(before) & set(after) & set(masks)
    print(f"  matched triplets: {len(keys)}")

    rows = sorted(
        [{"city": k[0], "x": k[1], "y": k[2],
          "before": before[k], "after": after[k], "mask": masks[k]}
         for k in keys],
        key=lambda r: (r["city"], int(r["x"]), int(r["y"]))
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["city", "x", "y", "before", "after", "mask"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written: {OUT_CSV}  ({len(rows)} triplets)")


if __name__ == "__main__":
    main()
