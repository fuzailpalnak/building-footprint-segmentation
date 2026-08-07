"""Create a small dummy dataset for local smoke tests.

Edit the CONFIG block below, then run:

    python scripts/create_dummy_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from building_footprint_segmentation.utils.script_config import apply_cli_overrides

# ---------------------------------------------------------------------------
# CONFIG — edit these paths / settings directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "output": str(PROJECT_ROOT / "data" / "dummy"),
}
# ---------------------------------------------------------------------------


def _make_sample(index: int, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(index)

    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = rng.integers(40, 90, size=(size, size), dtype=np.uint8)
    image[..., 1] = rng.integers(80, 140, size=(size, size), dtype=np.uint8)
    image[..., 2] = rng.integers(30, 70, size=(size, size), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    label = np.zeros((size, size, 3), dtype=np.uint8)
    for _ in range(3):
        width = rng.integers(60, 140)
        height = rng.integers(60, 140)
        x1 = rng.integers(20, size - width - 20)
        y1 = rng.integers(20, size - height - 20)
        x2, y2 = x1 + width, y1 + height

        building_color = int(rng.integers(150, 220))
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (building_color, building_color, building_color),
            thickness=-1,
        )
        cv2.rectangle(label, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    return image, label


def _write_split(root: Path, split: str, count: int, start_index: int) -> None:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    if split != "test":
        label_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(count):
        index = start_index + offset
        image, label = _make_sample(index)
        image_name = f"dummy_{index:03d}.png"

        cv2.imwrite(str(image_dir / image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if split != "test":
            cv2.imwrite(
                str(label_dir / image_name), cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
            )


def create_dummy_dataset(root: Path) -> None:
    _write_split(root, "train", count=4, start_index=0)
    _write_split(root, "val", count=2, start_index=100)
    _write_split(root, "test", count=2, start_index=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())
    root = Path(settings["output"])
    create_dummy_dataset(root)

    print(f"Created dummy dataset at: {root.resolve()}")
    print("Layout:")
    print("  train/images + train/labels  (4 samples)")
    print("  val/images   + val/labels    (2 samples)")
    print("  test/images                  (2 samples)")


if __name__ == "__main__":
    main()
