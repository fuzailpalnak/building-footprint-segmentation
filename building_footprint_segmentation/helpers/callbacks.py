import csv
import datetime
import logging
import os
import shutil

import cv2
import torch
import warnings
import time

from py_oneliner import one_liner

from building_footprint_segmentation.utils import date_time
from building_footprint_segmentation.utils.operations import (
    is_overridden_func,
    make_directory,
)
from building_footprint_segmentation.utils.py_network import (
    adjust_model,
    gpu_variable,
    convert_tensor_to_numpy,
)


logger = logging.getLogger("segmentation")


def _get_summary_writer():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


class CallbackList(object):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        if len(callbacks) != 0:
            [
                logger.debug("Registered {}".format(c.__class__.__name__))
                for c in callbacks
            ]

    def append(self, callback):
        logger.debug("Registered {}".format(callback.__class__.__name__))
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_begin):
                logger.debug(
                    "Nothing Registered On Epoch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_end):
                logger.debug(
                    "Nothing Registered On Epoch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            logger.debug("On Batch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_begin):
                logger.debug(
                    "Nothing Registered On Batch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):

        for callback in self.callbacks:
            logger.debug("On Batch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_end):
                logger.debug(
                    "Nothing Registered On Batch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_end(batch, logs)

    def on_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_begin):
                logger.debug(
                    "Nothing Registered On Begin {}".format(callback.__class__.__name__)
                )
            callback.on_begin(logs)

    def on_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_end):
                logger.debug(
                    "Nothing Registered On End {}".format(callback.__class__.__name__)
                )
            callback.on_end(logs)

    def interruption(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("Interruption {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.interruption):
                logger.debug(
                    "Nothing Registered On Interruption {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.interruption(logs)

    def update_params(self, params):
        for callback in self.callbacks:
            if not is_overridden_func(callback.update_params):
                logger.debug(
                    "Nothing Registered On Update param {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.update_params(params)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    def __init__(self, log_dir):
        self.log_dir = os.path.join(
            log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_begin(self, logs=None):
        pass

    def on_end(self, logs=None):
        pass

    def interruption(self, logs=None):
        pass

    def update_params(self, params):
        pass


class TrainStateCallback(Callback):
    """
    Save the training state
    """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        state = make_directory(self.log_dir, "state")
        self.chk = os.path.join(state, "default.pt")
        self.best = os.path.join(state, "best.pt")

        self.previous_best = None

    def on_epoch_end(self, epoch, logs=None):
        valid_loss = logs["valid_loss"]
        my_state = logs["state"]
        if self.previous_best is None or valid_loss < self.previous_best:
            self.previous_best = valid_loss
            torch.save(my_state, str(self.best))
        torch.save(my_state, str(self.chk))
        logger.debug(
            "Successful on Epoch End {}, Saved State".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["state"]

        torch.save(my_state, str(self.chk))
        logger.debug(
            "Successful on Interruption {}, Saved State".format(self.__class__.__name__)
        )


class TensorBoardCallback(Callback):
    """
    Log tensor board events
    """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        SummaryWriter = _get_summary_writer()
        self.writer = SummaryWriter(make_directory(self.log_dir, "events"))

    def plt_scalar(self, y, x, tag):
        if type(y) is dict:
            self.writer.add_scalars(tag, y, global_step=x)
            self.writer.flush()
        else:
            self.writer.add_scalar(tag, y, global_step=x)
            self.writer.flush()

    def plt_images(self, img, global_step, tag):
        self.writer.add_image(tag, img, global_step)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        lr = logs["lr"]
        train_loss = logs["train_loss"]
        valid_loss = logs["valid_loss"]

        train_metric = logs["train_metric"]
        valid_metric = logs["valid_metric"]

        self.plt_scalar(lr, epoch, "LR/Epoch")
        self.plt_scalar(
            {"train_loss": train_loss, "valid_loss": valid_loss}, epoch, "Loss/Epoch"
        )

        metric_keys = list(train_metric.keys())
        for key in metric_keys:
            self.plt_scalar(
                {
                    "Train_{}".format(key): train_metric[key],
                    "Valid_{}".format(key): valid_metric[key],
                },
                epoch,
                "{}/Epoch".format(key),
            )

        logger.debug(
            "Successful on Epoch End {}, Data Plot".format(self.__class__.__name__)
        )

    def on_batch_end(self, batch, logs=None):
        img_data = logs["plt_img"] if "plt_img" in logs else None
        data = logs["plt_lr"]

        if img_data is not None:
            # self.plt_images(to_tensor(np.moveaxis(img_data["img"], -1, 0)), batch, img_data["tag"])
            pass

        self.plt_scalar(data["data"], batch, data["tag"])
        logger.debug(
            "Successful on Batch End {}, Data Plot".format(self.__class__.__name__)
        )


class SchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__(None)
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(epoch)
        logger.debug(
            "Successful on Epoch End {}, Lr Scheduled".format(self.__class__.__name__)
        )


class TimeCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.start_time = None

    def on_begin(self, logs=None):
        self.start_time = time.time()

    def on_end(self, logs=None):
        end_time = time.time()
        total_time = date_time.get_time(end_time - self.start_time)
        one_liner.one_line(
            tag="Run Time",
            tag_data=f"{total_time}",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )

    def interruption(self, logs=None):
        end_time = time.time()
        total_time = date_time.get_time(end_time - self.start_time)
        one_liner.one_line(
            tag="Run Time",
            tag_data=f"{total_time}",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )


class TrainChkCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.chk = os.path.join(make_directory(self.log_dir, "chk_pth"), "chk_pth.pt")

    def on_epoch_end(self, epoch, logs=None):
        my_state = logs["state"]
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on Epoch End {}, Chk Saved".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["state"]
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on interruption {}, Chk Saved".format(self.__class__.__name__)
        )


class TestDuringTrainingCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.test_path = os.path.join(self.log_dir, "test_on_epoch_end")

    def on_epoch_end(self, epoch, logs=None):
        model = logs["model"]
        test_loader = logs["test_loader"]
        model.eval()
        try:
            if os.path.exists(self.test_path):
                shutil.rmtree(self.test_path)

            for i, test_data in enumerate(test_loader):
                self.inference(
                    model,
                    gpu_variable(test_data["images"]),
                    test_data["file_name"],
                    make_directory(
                        os.path.dirname(self.test_path),
                        os.path.basename(self.test_path),
                    ),
                    epoch,
                )
                break
        except Exception as ex:
            logger.exception("Skipped Exception in {}".format(self.__class__.__name__))
            logger.exception("Exception {}".format(ex))
            pass

    def inference(self, model, image, file_name, save_path, index):
        pass


class BinaryTestCallback(TestDuringTrainingCallback):
    def __init__(self, log_dir, threshold: float = 0.20):
        super().__init__(log_dir)
        self._threshold = threshold

    @torch.no_grad()
    def inference(self, model, image, file_name, save_path, index):
        """

        :param model: the model used for training
        :param image: the images loaded by the test loader
        :param file_name: the file name of the test image
        :param save_path: path where to save the image
        :param index:
        :return:
        """
        prediction = model(image)
        prediction = prediction.sigmoid()
        prediction[prediction >= self._threshold] = 1
        prediction[prediction < self._threshold] = 0

        batch, _, h, w = prediction.shape
        for i in range(batch):
            prediction_numpy = convert_tensor_to_numpy(prediction[i])
            prediction_numpy = prediction_numpy.reshape((h, w))
            cv2.imwrite(
                os.path.join(save_path, f"{file_name[i]}.png"), prediction_numpy * 255
            )


class MetricsPlotCallback(Callback):
    """
    Log per-epoch train/val metrics to CSV and save a YOLO-style results chart.
    """

    def __init__(self, log_dir):
        # Keep a stable path (no timestamp) so results.png is easy to find.
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "results.csv")
        self.plot_path = os.path.join(self.log_dir, "results.png")
        self.history = []
        self._fieldnames = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_metric = logs.get("train_metric") or {}
        valid_metric = logs.get("valid_metric") or {}

        row = {
            "epoch": int(epoch),
            "lr": float(logs.get("lr", 0.0)),
            "train_loss": float(logs.get("train_loss", 0.0)),
            "valid_loss": float(logs.get("valid_loss", 0.0)),
        }
        for key, value in train_metric.items():
            row[f"train_{key}"] = float(value)
        for key, value in valid_metric.items():
            row[f"valid_{key}"] = float(value)

        self.history.append(row)
        self._write_csv()
        # Refresh chart every epoch so a long run still has a usable plot if interrupted.
        self._plot_results()
        logger.debug(
            "Successful on Epoch End {}, Metrics Saved".format(self.__class__.__name__)
        )

    def on_end(self, logs=None):
        self._plot_results()
        one_liner.one_line(
            tag="Results Chart",
            tag_data=self.plot_path,
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )

    def interruption(self, logs=None):
        self._plot_results()
        one_liner.one_line(
            tag="Results Chart",
            tag_data=self.plot_path,
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )

    def _write_csv(self):
        if not self.history:
            return
        fieldnames = list(self.history[0].keys())
        self._fieldnames = fieldnames
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def _plot_results(self):
        if not self.history:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for results charts. "
                "Install with: pip install matplotlib"
            ) from exc

        epochs = [row["epoch"] for row in self.history]
        panels = [
            ("Loss", [("train_loss", "train"), ("valid_loss", "val")]),
            ("Accuracy", [("train_accuracy", "train"), ("valid_accuracy", "val")]),
            ("Precision", [("train_precision", "train"), ("valid_precision", "val")]),
            ("Recall", [("train_recall", "train"), ("valid_recall", "val")]),
            ("F1", [("train_f1", "train"), ("valid_f1", "val")]),
            ("IoU", [("train_iou", "train"), ("valid_iou", "val")]),
            ("Learning Rate", [("lr", "lr")]),
        ]

        available_panels = []
        for title, series in panels:
            usable = [
                (key, label)
                for key, label in series
                if any(key in row for row in self.history)
            ]
            if usable:
                available_panels.append((title, usable))

        cols = 2
        rows = (len(available_panels) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
        fig.suptitle("Training Results", fontsize=14, fontweight="bold")

        for index, (title, series) in enumerate(available_panels):
            ax = axes[index // cols][index % cols]
            for key, label in series:
                values = [row.get(key) for row in self.history]
                ax.plot(epochs, values, marker="o", markersize=2, linewidth=1.5, label=label)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        for index in range(len(available_panels), rows * cols):
            axes[index // cols][index % cols].axis("off")

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)

class PredictionSampleCallback(Callback):
    """
    Save a visual GT vs prediction grid for a fixed set of random validation tiles.
    Updated every epoch so the file always reflects the latest model.

    Rows: image, ground truth, prediction, merge overlay
    (white=TP, blue=FP, red=FN; background stays black).
    """

    def __init__(
        self,
        log_dir,
        num_samples: int = 5,
        threshold: float = 0.20,
        seed: int = 42,
    ):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.num_samples = num_samples
        self.threshold = threshold
        self.seed = seed
        self.sample_indices = None
        self.plot_path = os.path.join(self.log_dir, "predictions.png")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        model = logs.get("model")
        val_loader = logs.get("val_loader")
        if model is None or val_loader is None:
            logger.debug(
                "Skipped {} — model or val_loader missing from logs".format(
                    self.__class__.__name__
                )
            )
            return
        self._save_prediction_grid(model, val_loader, epoch)
        logger.debug(
            "Successful on Epoch End {}, Prediction Grid Saved".format(
                self.__class__.__name__
            )
        )

    def interruption(self, logs=None):
        logs = logs or {}
        model = logs.get("model")
        val_loader = logs.get("val_loader")
        if model is None or val_loader is None:
            return
        epoch = logs.get("state", {}).get("start_epoch", "?")
        self._save_prediction_grid(model, val_loader, epoch)

    @staticmethod
    def _pad_to_multiple(images, multiple: int = 32):
        import torch.nn.functional as F

        _, _, height, width = images.shape
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        if pad_height or pad_width:
            images = F.pad(images, (0, pad_width, 0, pad_height))
        return images, (height, width)

    def _ensure_indices(self, dataset_size: int):
        if self.sample_indices is not None:
            return
        count = min(self.num_samples, dataset_size)
        rng = __import__("random").Random(self.seed)
        self.sample_indices = rng.sample(range(dataset_size), count)

    def _save_prediction_grid(self, model, val_loader, epoch):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for prediction grids. "
                "Install with: pip install matplotlib"
            ) from exc

        dataset = val_loader.dataset
        if len(dataset) == 0:
            return

        self._ensure_indices(len(dataset))
        model.eval()

        images = []
        grounds = []
        preds = []
        merges = []

        with torch.no_grad():
            for index in self.sample_indices:
                sample = dataset[index]
                image = sample["images"].unsqueeze(0)
                ground = sample["ground_truth"]

                image = gpu_variable(image)
                padded, (orig_h, orig_w) = self._pad_to_multiple(image)
                prediction = model(padded).sigmoid()
                prediction = prediction[:, :, :orig_h, :orig_w]
                prediction = (prediction >= self.threshold).float()

                image_np = convert_tensor_to_numpy(image[0])
                if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                    image_np = np.moveaxis(image_np, 0, -1)
                image_np = np.clip(image_np, 0.0, 1.0)

                ground_np = convert_tensor_to_numpy(ground)
                if ground_np.ndim == 3:
                    ground_np = ground_np.reshape(ground_np.shape[-2], ground_np.shape[-1])

                pred_np = convert_tensor_to_numpy(prediction[0])
                if pred_np.ndim == 3:
                    pred_np = pred_np.reshape(pred_np.shape[-2], pred_np.shape[-1])

                gt_bin = ground_np >= 0.5
                pred_bin = pred_np >= 0.5
                merge_np = np.zeros((*gt_bin.shape, 3), dtype=np.float32)
                # True positive: building correctly predicted → white
                merge_np[gt_bin & pred_bin] = (1.0, 1.0, 1.0)
                # False positive: predicted building on background → blue
                merge_np[~gt_bin & pred_bin] = (0.0, 0.45, 1.0)
                # False negative: missed ground-truth building → red
                merge_np[gt_bin & ~pred_bin] = (1.0, 0.15, 0.15)

                images.append(image_np)
                grounds.append(ground_np)
                preds.append(pred_np)
                merges.append(merge_np)

        cols = len(self.sample_indices)
        fig, axes = plt.subplots(4, cols, figsize=(3.0 * cols, 11.5), squeeze=False)
        fig.suptitle(
            "Validation Samples — Epoch {} "
            "(image / GT / prediction / merge: white=TP, blue=FP, red=FN)".format(
                epoch
            ),
            fontsize=11,
            fontweight="bold",
        )

        for col, (image_np, ground_np, pred_np, merge_np) in enumerate(
            zip(images, grounds, preds, merges)
        ):
            axes[0][col].imshow(image_np)
            axes[0][col].set_title(f"#{self.sample_indices[col]}", fontsize=9)
            axes[0][col].axis("off")

            axes[1][col].imshow(ground_np, cmap="gray", vmin=0, vmax=1)
            if col == 0:
                axes[1][col].set_ylabel("Ground Truth", fontsize=10)
            axes[1][col].axis("off")

            axes[2][col].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            if col == 0:
                axes[2][col].set_ylabel("Prediction", fontsize=10)
            axes[2][col].axis("off")

            axes[3][col].imshow(merge_np)
            if col == 0:
                axes[3][col].set_ylabel("Merge", fontsize=10)
            axes[3][col].axis("off")

        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


def load_default_callbacks(log_dir: str):
    return [
        TrainChkCallback(log_dir),
        TimeCallback(log_dir),
        TensorBoardCallback(log_dir),
        TrainStateCallback(log_dir),
        MetricsPlotCallback(log_dir),
        PredictionSampleCallback(log_dir),
    ]


def load_callback(log_dir: str, callback: str) -> Callback:
    """
    :param log_dir:
    :param callback:
    :return:
    """
    return eval(callback)(log_dir)
