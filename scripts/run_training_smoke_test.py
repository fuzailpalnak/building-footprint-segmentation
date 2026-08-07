"""Run a 1-epoch training smoke test on dummy data.

Edit the CONFIG block below, then run:

    python scripts/run_training_smoke_test.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A

from building_footprint_segmentation._env import configure_windows_openmp
from building_footprint_segmentation.helpers.callbacks import CallbackList, TimeCallback
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
    "data": str(PROJECT_ROOT / "data" / "dummy"),
    "weights": str(PROJECT_ROOT / "refine.pth"),
    "output": str(PROJECT_ROOT / "runs" / "training"),
    "epochs": 1,
    "batch_size": 2,
}
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())
    data_root = Path(settings["data"])
    weights_path = Path(settings["weights"])
    output_dir = create_next_run_dir(settings["output"], prefix="train")

    if not (data_root / "train" / "images").exists():
        raise FileNotFoundError(
            f"Missing training data at {data_root}. "
            "Run: python scripts/create_dummy_data.py"
        )

    segmentation = init_segmentation("binary")
    weights = str(weights_path) if weights_path.exists() else None
    model = segmentation.load_model("ReFineNet", transfer_weights=weights)
    criterion = segmentation.load_criterion(name="BinaryCrossEntropy")
    loader = segmentation.load_loader(
        root_folder=str(data_root),
        image_normalization="divide_by_255",
        label_normalization="binary_label",
        augmenters=A.Compose([A.HorizontalFlip(p=0.5)]),
        batch_size=int(settings["batch_size"]),
    )
    metrics = segmentation.load_metrics(
        data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
    )
    optimizer = segmentation.load_optimizer(model, name="Adam", lr=1e-4)
    callbacks = CallbackList([TimeCallback(log_dir=str(output_dir))])

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loader=loader,
        metrics=metrics,
        callbacks=callbacks,
        scheduler=None,
    )

    print(f"Starting smoke training for {settings['epochs']} epoch(s)...")
    trainer.train(start_epoch=0, end_epoch=int(settings["epochs"]))
    print("Training smoke test complete.")


if __name__ == "__main__":
    main()
