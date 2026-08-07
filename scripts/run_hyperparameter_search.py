"""Grid-search hyperparameters for ReFineNet fine-tuning.

Edit the CONFIG block below, then run:

    python scripts/run_hyperparameter_search.py

CLI flags are optional and override CONFIG when provided.
Tries every mix of the provided hyperparameter ranges, trains each mix for a
fixed number of epochs, writes per-trial charts, then reports the best mix.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

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
    "data": str(PROJECT_ROOT / "data" / "steyr_train"),
    "weights": str(PROJECT_ROOT / "refine.pth"),
    "output": str(PROJECT_ROOT / "runs" / "hyperparameter"),
    "epochs_per_trial": 20,
    "lrs": [1e-5, 5e-5, 1e-4],
    "batch_sizes": [2, 4, 8],
    "weight_decays": [0.0, 1e-4],
    "criteria": ["BinaryCrossEntropy", "Dice"],
    "thresholds": [0.20],
    "metric": "valid_iou",
    "sample_count": 5,
    "max_trials": 0,  # 0 = run full grid
}
# ---------------------------------------------------------------------------


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--epochs-per-trial",
        dest="epochs_per_trial",
        type=int,
        default=None,
    )
    parser.add_argument("--lrs", type=_parse_floats, default=None)
    parser.add_argument(
        "--batch-sizes",
        dest="batch_sizes",
        type=_parse_ints,
        default=None,
    )
    parser.add_argument(
        "--weight-decays",
        dest="weight_decays",
        type=_parse_floats,
        default=None,
    )
    parser.add_argument("--criteria", type=_parse_strings, default=None)
    parser.add_argument("--thresholds", type=_parse_floats, default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument(
        "--sample-count",
        dest="sample_count",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-trials",
        dest="max_trials",
        type=int,
        default=None,
    )
    return parser.parse_args()


def build_augmenters() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )


def criterion_kwargs(name: str) -> dict[str, Any]:
    if name == "Dice":
        return {"dice_weight": 1.0}
    if name == "IOU":
        return {"iou_weight": 0.5}
    return {}


def trial_name(params: dict[str, Any], index: int) -> str:
    lr_tag = f"{params['lr']:.0e}".replace(".", "p").replace("-", "m")
    wd_tag = f"{params['weight_decay']:.0e}".replace(".", "p").replace("-", "m")
    return (
        f"trial_{index:03d}_lr{lr_tag}_bs{params['batch_size']}"
        f"_wd{wd_tag}_{params['criterion']}_th{params['threshold']}"
    )


def read_best_metric(results_csv: Path, metric: str) -> tuple[float | None, dict | None]:
    if not results_csv.exists():
        return None, None
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None
    if metric not in rows[0]:
        raise KeyError(
            f"Metric '{metric}' not found in {results_csv}. "
            f"Available: {list(rows[0].keys())}"
        )
    best_row = max(rows, key=lambda row: float(row[metric]))
    return float(best_row[metric]), best_row


def run_trial(
    *,
    data_root: Path,
    weights: str | None,
    trial_dir: Path,
    params: dict[str, Any],
    epochs: int,
    sample_count: int,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)

    with (trial_dir / "hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    segmentation = init_segmentation("binary")
    model = segmentation.load_model("ReFineNet", transfer_weights=weights)
    criterion = segmentation.load_criterion(
        name=params["criterion"], **criterion_kwargs(params["criterion"])
    )
    loader = segmentation.load_loader(
        root_folder=str(data_root),
        image_normalization="divide_by_255",
        label_normalization="binary_label",
        augmenters=build_augmenters(),
        batch_size=params["batch_size"],
    )
    metrics = segmentation.load_metrics(
        data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
    )
    optimizer = segmentation.load_optimizer(
        model,
        name="Adam",
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    callbacks = CallbackList(
        [
            TimeCallback(log_dir=str(trial_dir)),
            TrainStateCallback(log_dir=str(trial_dir)),
            TrainChkCallback(log_dir=str(trial_dir)),
            MetricsPlotCallback(log_dir=str(trial_dir)),
            PredictionSampleCallback(
                log_dir=str(trial_dir),
                num_samples=sample_count,
                threshold=params["threshold"],
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
    trainer.train(start_epoch=0, end_epoch=epochs)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(summary_rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for comparison charts. "
            "Install with: pip install matplotlib"
        ) from exc

    completed = [row for row in summary_rows if row.get("status") == "ok"]
    if not completed:
        return

    labels = [row["trial"] for row in completed]
    values = [float(row[metric]) for row in completed]
    best_index = max(range(len(values)), key=lambda i: values[i])

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    bars = ax.bar(range(len(labels)), values, color="#4C78A8")
    bars[best_index].set_color("#F58518")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"Hyperparameter Search — best {metric} per trial")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_curve_overlay(
    trial_dirs: list[Path],
    metric: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for comparison charts. "
            "Install with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for trial_dir in trial_dirs:
        csv_path = trial_dir / "results.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or metric not in rows[0]:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row[metric]) for row in rows]
        ax.plot(epochs, values, linewidth=1.5, label=trial_dir.name)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Hyperparameter Search — {metric} curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())

    data_root = Path(settings["data"])
    output_root = create_next_run_dir(settings["output"], prefix="hyperparameter")
    weights_path = Path(settings["weights"])
    epochs_per_trial = int(settings["epochs_per_trial"])
    metric = settings["metric"]
    sample_count = int(settings["sample_count"])
    max_trials = int(settings["max_trials"])

    train_images = data_root / "train" / "images"
    val_images = data_root / "val" / "images"
    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            f"Missing train/val under {data_root}. "
            "Prepare the dataset first (see README Training guide)."
        )

    weights = str(weights_path) if weights_path.exists() else None
    if weights is None:
        print(f"Warning: weights not found at {weights_path}, training from scratch.")

    lrs = list(settings["lrs"])
    batch_sizes = list(settings["batch_sizes"])
    weight_decays = list(settings["weight_decays"])
    criteria = list(settings["criteria"])
    thresholds = list(settings["thresholds"])

    grid = list(
        itertools.product(lrs, batch_sizes, weight_decays, criteria, thresholds)
    )
    if max_trials > 0:
        grid = grid[:max_trials]

    print(f"Search root: {output_root.resolve()}")
    print(f"Trials: {len(grid)} | epochs/trial: {epochs_per_trial}")
    print(f"Optimize: maximize {metric}")

    summary_rows: list[dict[str, Any]] = []
    trial_dirs: list[Path] = []

    for index, (lr, batch_size, weight_decay, criterion, threshold) in enumerate(grid):
        params = {
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "criterion": criterion,
            "threshold": threshold,
            "epochs": epochs_per_trial,
        }
        name = trial_name(params, index)
        trial_dir = output_root / name
        trial_dirs.append(trial_dir)

        print("\n" + "=" * 72)
        print(f"[{index + 1}/{len(grid)}] {name}")
        print(json.dumps(params, indent=2))
        print("=" * 72)

        row: dict[str, Any] = {
            "trial": name,
            "trial_dir": str(trial_dir),
            **params,
            "status": "ok",
            "error": "",
            metric: None,
            "best_epoch": None,
        }

        try:
            run_trial(
                data_root=data_root,
                weights=weights,
                trial_dir=trial_dir,
                params=params,
                epochs=epochs_per_trial,
                sample_count=sample_count,
            )
            best_value, best_row = read_best_metric(trial_dir / "results.csv", metric)
            row[metric] = best_value
            row["best_epoch"] = int(best_row["epoch"]) if best_row else None
            if best_row:
                for key, value in best_row.items():
                    if key.startswith("valid_") or key.startswith("train_"):
                        row[f"at_best_{key}"] = float(value)
            print(f"Trial best {metric}: {best_value}")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print(f"Trial FAILED: {exc}")
            traceback.print_exc()

        summary_rows.append(row)
        write_summary_csv(output_root / "search_summary.csv", summary_rows)

    completed = [
        row for row in summary_rows if row.get("status") == "ok" and row.get(metric) is not None
    ]
    if not completed:
        raise RuntimeError("All hyperparameter trials failed. See search_summary.csv.")

    best = max(completed, key=lambda row: float(row[metric]))
    best_params = {
        "lr": best["lr"],
        "batch_size": best["batch_size"],
        "weight_decay": best["weight_decay"],
        "criterion": best["criterion"],
        "threshold": best["threshold"],
        "epochs_per_trial": best["epochs"],
        "selection_metric": metric,
        "best_metric_value": best[metric],
        "best_epoch": best["best_epoch"],
        "trial": best["trial"],
        "trial_dir": best["trial_dir"],
    }

    best_json = output_root / "best_hyperparameters.json"
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)

    # Promote best trial visuals to the search root for easy access.
    best_dir = Path(best["trial_dir"])
    for filename in ("results.png", "predictions.png", "results.csv"):
        source = best_dir / filename
        if source.exists():
            shutil.copy2(source, output_root / f"best_{filename}")

    plot_comparison(summary_rows, metric, output_root / "comparison.png")
    plot_curve_overlay(trial_dirs, metric, output_root / "curves_comparison.png")

    print("\n" + "=" * 72)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 72)
    print(json.dumps(best_params, indent=2))
    print(f"\nSummary CSV: {output_root / 'search_summary.csv'}")
    print(f"Best params:  {best_json}")
    print(f"Comparison:   {output_root / 'comparison.png'}")
    print(f"Curves:       {output_root / 'curves_comparison.png'}")
    print(f"Best charts:  {output_root / 'best_results.png'}, {output_root / 'best_predictions.png'}")
    print(
        "\nSuggested full training command:\n"
        f"python scripts/run_training.py --data {data_root} --weights {weights_path} "
        f"--output runs/training --epochs 300 "
        f"--batch-size {best_params['batch_size']} --lr {best_params['lr']}"
    )


if __name__ == "__main__":
    main()
