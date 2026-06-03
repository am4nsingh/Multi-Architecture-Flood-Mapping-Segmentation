"""
City-aware train/val/test split.
Splits entire cities into partitions to prevent spatial data leakage.
Falls back to per-city spatial strips when fewer than 3 cities exist.
Outputs: flood_mapping/splits.csv
"""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS_CSV = ROOT / "flood_mapping" / "triplets.csv"
SPLITS_CSV = ROOT / "flood_mapping" / "splits.csv"

VAL_FRAC = 0.15
TEST_FRAC = 0.15
BUFFER = 256  # px — one tile gap between strips (used in per-city fallback)
SEED = 1337


def _per_city_spatial_split(city_rows):
    """Fallback: spatial strip split within a single city.

    Buffer zones between strips are EXCLUDED from all splits so that no
    training tile is spatially adjacent to a val/test tile (SAR backscatter
    is spatially autocorrelated within a city).
    """
    ys = sorted({int(r["y"]) for r in city_rows})
    y_min, y_max = ys[0], ys[-1]
    y_range = y_max - y_min

    test_hi = y_min + int(TEST_FRAC * y_range)
    val_lo = test_hi + BUFFER
    val_hi = val_lo + int(VAL_FRAC * y_range)
    train_lo = val_hi + BUFFER  # second buffer between val and train

    assignments = {}
    for r in city_rows:
        y = int(r["y"])
        if y < test_hi:
            assignments[id(r)] = "test"
        elif val_lo <= y < val_hi:
            assignments[id(r)] = "val"
        elif y >= train_lo:
            assignments[id(r)] = "train"
        # else: buffer zone — tile excluded from all splits
    return assignments


def main():
    with open(TRIPLETS_CSV) as f:
        rows = list(csv.DictReader(f))

    cities = sorted({r["city"] for r in rows})
    n_cities = len(cities)
    print(f"Found {n_cities} cities: {cities}")

    if n_cities >= 3:
        rng = random.Random(SEED)
        rng.shuffle(cities)
        n_test = max(1, int(TEST_FRAC * n_cities))
        n_val = max(1, int(VAL_FRAC * n_cities))

        test_cities = set(cities[:n_test])
        val_cities = set(cities[n_test:n_test + n_val])
        train_cities = set(cities[n_test + n_val:])

        print(f"test  cities: {sorted(test_cities)}")
        print(f"val   cities: {sorted(val_cities)}")
        print(f"train cities: {sorted(train_cities)}")

        def assign_split(r):
            c = r["city"]
            if c in test_cities:
                return "test"
            if c in val_cities:
                return "val"
            return "train"
    else:
        print("Few cities — falling back to per-city spatial strip split")
        city_groups = {}
        for r in rows:
            city_groups.setdefault(r["city"], []).append(r)

        spatial_assignments = {}
        for city, city_rows in city_groups.items():
            spatial_assignments.update(_per_city_spatial_split(city_rows))

        def assign_split(r):
            # Buffer-zone tiles are absent from the map → excluded (None)
            return spatial_assignments.get(id(r))

    counts = {"train": 0, "val": 0, "test": 0}
    excluded = 0
    with open(SPLITS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["city", "x", "y", "before", "after", "mask", "split"])
        writer.writeheader()
        for r in rows:
            split = assign_split(r)
            if split is None:
                excluded += 1
                continue
            counts[split] += 1
            writer.writerow({**r, "split": split})

    print(f"\nSplit counts: {counts}  (excluded buffer tiles: {excluded})")
    print(f"Written: {SPLITS_CSV}")

    # Invalidate split-derived artifacts so they are rebuilt against this split
    for stale in (SPLITS_CSV.with_name("nodata_index.json"),
                  ROOT / "flood_mapping" / "data" / "train_stats.json"):
        if stale.exists():
            stale.unlink()
            print(f"Removed stale artifact: {stale}")


if __name__ == "__main__":
    main()
