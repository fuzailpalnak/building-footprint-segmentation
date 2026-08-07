"""Fine-tune ReFineNet on a prepared train/val dataset.

Edit the CONFIG block below, then run:

    python scripts/run_training.py

CLI flags are optional and override CONFIG when provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A

from building_footprint_segmentation._env import configure_windows_openmp
from building_footprint_segmentation.helpers.callbacks import (
    CallbackList,
    MetricsPlotCallback,
    PredictionSampleCallback,
    TimeCallback,
    TrainChkCallback,
    TrainStateCallback,
)
from building_footprint_segmentation.segmentation import init_segmentation
from building_footprint_segmentation.trainer import Trainer
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
    # Dataset root that contains train/ and val/ (with images/ + labels/)
    "data": str(PROJECT_ROOT / "data" / "steyr_train"),
    "weights": str(PROJECT_ROOT / "refine.pth"),
    "output": str(PROJECT_ROOT / "runs" / "training"),
    "model": "ReFineNet",
    "criterion": "BinaryCrossEntropy",
    "epochs": 300,
    "batch_size": 8,
    "lr": 1e-4,
    "sample_count": 5,
    "threshold": 0.20,
}
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--criterion", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sample-count", dest="sample_count", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())

    data_root = Path(settings["data"])
    output_dir = create_next_run_dir(settings["output"], prefix="train")
    weights_path = Path(settings["weights"])

    train_images = data_root / "train" / "images"
    val_images = data_root / "val" / "images"
    train_labels = data_root / "train" / "labels"
    val_labels = data_root / "val" / "labels"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            f"Missing train/val folders under {data_root}. "
            "Run scripts/prepare_steyr_dataset.py first."
        )
    if not train_labels.exists() or not val_labels.exists():
        raise FileNotFoundError(
            f"Missing train/val labels under {data_root}. "
            "Run scripts/rasterize_building_masks.py first."
        )

    weights = str(weights_path) if weights_path.exists() else None
    if weights is None:
        print(f"Warning: weights not found at {weights_path}, training from scratch.")

    segmentation = init_segmentation("binary")
    model = segmentation.load_model(settings["model"], transfer_weights=weights)
    criterion = segmentation.load_criterion(name=settings["criterion"])
    loader = segmentation.load_loader(
        root_folder=str(data_root),
        image_normalization="divide_by_255",
        label_normalization="binary_label",
        augmenters=A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        ),
        batch_size=int(settings["batch_size"]),
    )
    metrics = segmentation.load_metrics(
        data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
    )
    optimizer = segmentation.load_optimizer(
        model, name="Adam", lr=float(settings["lr"])
    )

    callbacks = CallbackList(
        [
            TimeCallback(log_dir=str(output_dir)),
            TrainStateCallback(log_dir=str(output_dir)),
            TrainChkCallback(log_dir=str(output_dir)),
            MetricsPlotCallback(log_dir=str(output_dir)),
            PredictionSampleCallback(
                log_dir=str(output_dir),
                num_samples=int(settings["sample_count"]),
                threshold=float(settings["threshold"]),
            ),
        ]
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loader=loader,
        metrics=metrics,
        callbacks=callbacks,
        scheduler=None,
    )

    print(f"Training data: {data_root.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(
        f"Epochs: {settings['epochs']}, batch size: {settings['batch_size']}, "
        f"lr: {settings['lr']}"
    )
    print(f"Train tiles: {len(loader.train_loader.dataset)}")
    print(f"Val tiles: {len(loader.val_loader.dataset)}")

    trainer.train(start_epoch=0, end_epoch=int(settings["epochs"]))

    print("Training complete.")
    print(f"Results chart: {output_dir / 'results.png'}")
    print(f"Results CSV: {output_dir / 'results.csv'}")
    print(f"Prediction samples: {output_dir / 'predictions.png'}")
    print(f"Best checkpoint: {output_dir / 'state' / 'best.pt'}")
    print(f"Latest weights: {output_dir / 'chk_pth' / 'chk_pth.pt'}")


if __name__ == "__main__":
    main()
