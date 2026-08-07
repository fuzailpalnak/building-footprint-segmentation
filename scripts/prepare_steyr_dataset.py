"""Split chipped tiles + masks into train/val folders for training.

Edit the CONFIG block below, then run:

    python scripts/prepare_steyr_dataset.py

CLI flags are optional and override CONFIG when provided.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from building_footprint_segmentation.utils.script_config import apply_cli_overrides

TILE_PREFIX = "tile_"

# ---------------------------------------------------------------------------
# CONFIG — edit these paths / settings directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "images": str(PROJECT_ROOT / "data" / "steyr_512" / "images"),
    "labels": str(PROJECT_ROOT / "data" / "steyr_512" / "labels"),
    "output": str(PROJECT_ROOT / "data" / "steyr_train"),
    "val_fraction": 0.2,
    # Only keep tiles of this size; set 0 to keep all
    "only_size": 512,
}
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--val-fraction", dest="val_fraction", type=float, default=None)
    parser.add_argument("--only-size", dest="only_size", type=int, default=None)
    return parser.parse_args()


def tile_x_offset(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "tile":
        raise ValueError(f"Unexpected tile name: {path.name}")
    return int(parts[1])


def copy_pair(image_path: Path, label_path: Path, split: str, output_root: Path) -> None:
    image_dest = output_root / split / "images" / image_path.name
    label_dest = output_root / split / "labels" / label_path.name
    image_dest.parent.mkdir(parents=True, exist_ok=True)
    label_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_dest)
    shutil.copy2(label_path, label_dest)


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())

    images_dir = Path(settings["images"])
    labels_dir = Path(settings["labels"])
    output_root = Path(settings["output"])
    val_fraction = float(settings["val_fraction"])
    only_size = int(settings["only_size"])

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    image_paths = sorted(images_dir.glob(f"{TILE_PREFIX}*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No tiles found in {images_dir}")

    x_values = sorted({tile_x_offset(path) for path in image_paths})
    val_column_count = max(1, round(len(x_values) * val_fraction))
    val_x_values = set(x_values[-val_column_count:])

    train_count = 0
    val_count = 0
    skipped = 0
    size_skipped = 0

    for image_path in image_paths:
        if only_size:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping (unreadable): {image_path.name}")
                skipped += 1
                continue
            height, width = image.shape[:2]
            if width != only_size or height != only_size:
                size_skipped += 1
                continue

        label_path = labels_dir / image_path.name
        if not label_path.exists():
            print(f"Skipping (missing label): {image_path.name}")
            skipped += 1
            continue

        split = "val" if tile_x_offset(image_path) in val_x_values else "train"
        copy_pair(image_path, label_path, split, output_root)
        if split == "train":
            train_count += 1
        else:
            val_count += 1

    if train_count == 0 or val_count == 0:
        raise RuntimeError(
            f"Split failed: train={train_count}, val={val_count}. "
            "Try a smaller val_fraction or check tile names."
        )

    print(f"Output: {output_root.resolve()}")
    print(f"Train tiles: {train_count}")
    print(f"Val tiles: {val_count}")
    print(f"Skipped (missing labels): {skipped}")
    if only_size:
        print(f"Skipped (not {only_size}x{only_size}): {size_skipped}")
    print(f"Val held-out x columns ({len(val_x_values)}): {sorted(val_x_values)}")


if __name__ == "__main__":
    main()
