"""Run building-footprint inference on a folder of images.

Edit the CONFIG block below, then run:

    python scripts/run_inference_test.py

CLI flags are optional and override CONFIG when provided.
The --images path is used as-is (no forced test/images suffix).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from building_footprint_segmentation._env import configure_windows_openmp
from building_footprint_segmentation.helpers import normalizer
from building_footprint_segmentation.segmentation import init_segmentation
from building_footprint_segmentation.utils.operations import load_image
from building_footprint_segmentation.utils.py_network import (
    convert_tensor_to_numpy,
    gpu_variable,
    to_input_image_tensor,
)
from building_footprint_segmentation.utils.script_config import (
    apply_cli_overrides,
    create_next_run_dir,
)

configure_windows_openmp()

# ---------------------------------------------------------------------------
# CONFIG — edit these paths / settings directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "images": r"D:\sadoun\Devs\BuildingFootPrint\data\steyr_512\test\images",          # folder with the images themselves
    "weights": r"D:\sadoun\Devs\BuildingFootPrint\outputs\hyperparam_search_1024\trial_023_lr1em04_bs4_wd1em04_Dice_th0.2\20260728-132101\state\best.pt",
    "output": str(PROJECT_ROOT / "runs" / "inference"),
    "model": "ReFineNet",
    "threshold": 0.20,
    "batch_size": 1,
}
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    """Load images from a flat folder path (no train/val/test layout)."""

    def __init__(self, images_dir: Path):
        self.images_dir = images_dir
        self.images = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in: {images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        path = self.images[index]
        image = load_image(str(path))
        image = normalizer.divide_by_255(image)
        return {
            "images": to_input_image_tensor(image),
            "file_name": os.path.basename(str(path)),
        }


def pad_to_multiple(
    images: torch.Tensor, multiple: int = 32
) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = images.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height or pad_width:
        images = F.pad(images, (0, pad_width, 0, pad_height))
    return images, (height, width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())

    images_dir = Path(settings["images"])
    weights_path = Path(settings["weights"])
    output_dir = create_next_run_dir(settings["output"], prefix="inference")
    model_name = settings["model"]
    threshold = float(settings["threshold"])
    batch_size = int(settings["batch_size"])

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Images:  {images_dir.resolve()}")
    print(f"Weights: {weights_path.resolve()}")
    print(f"Output:  {output_dir.resolve()}")
    print(f"Model:   {model_name}")
    print(f"Threshold: {threshold}")

    segmentation = init_segmentation("binary")
    model = segmentation.load_model(model_name, transfer_weights=str(weights_path))
    model.eval()

    dataset = ImageFolderDataset(images_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Running inference on {len(dataset)} image(s)...")

    with torch.no_grad():
        for batch in loader:
            images = gpu_variable(batch["images"])
            padded_images, (orig_height, orig_width) = pad_to_multiple(images)
            predictions = model(padded_images).sigmoid()
            predictions = predictions[:, :, :orig_height, :orig_width]
            predictions = (predictions >= threshold).float()

            file_names = batch["file_name"]
            if isinstance(file_names, str):
                file_names = [file_names]

            batch_size_now, _, height, width = predictions.shape
            for index in range(batch_size_now):
                mask = convert_tensor_to_numpy(predictions[index]).reshape(height, width)
                mask_image = (mask * 255).astype(np.uint8)
                output_path = output_dir / f"{Path(file_names[index]).stem}_mask.png"
                cv2.imwrite(str(output_path), mask_image)
                print(f"Saved: {output_path}")

    print(f"Done. Predictions are in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
