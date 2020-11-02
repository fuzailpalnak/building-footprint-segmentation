import logging
import random
import time
from typing import Tuple, Union

import numpy as np
import torch
import tqdm
from py_oneliner import one_liner

from building_footprint_segmentation.helpers.callbacks import SchedulerCallback
from building_footprint_segmentation.utils.operations import (
    handle_dictionary,
    compute_eta,
    dict_to_string,
)
from building_footprint_segmentation.utils.py_network import gpu_variable, extract_state

logger = logging.getLogger("segmentation")


class Trainer:
    def __init__(
        self, model, criterion, optimizer, loader, metrics, callbacks, scheduler
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader = loader
        self.metrics = metrics
        self.callbacks = callbacks

        if scheduler is not None:
            self.callbacks.append(SchedulerCallback(scheduler))

    def previous_state(self, state):
        assert {
            "model",
            "optimizer",
            "step",
            "start_epoch",
            "end_epoch",
            "bst_vld_loss",
        } == set(list(state.keys())), (
            "Expected 'state' to have ['model', 'optimizer', 'step', 'start_epoch', 'end_epoch', 'bst_vld_loss']'"
            "got %s",
            (list(state.keys()),),
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        return (
            state["step"],
            state["start_epoch"],
            state["end_epoch"],
            state["bst_vld_loss"],
        )

    def resume(self, state: str, new_end_epoch=None):
        step, start_epoch, end_epoch, bst_vld_loss = self.previous_state(
            extract_state(state)
        )
        if new_end_epoch is not None:
            logger.debug(f"Overriding end epoch from {end_epoch} to {new_end_epoch}")
            end_epoch = new_end_epoch

        logger.debug(
            f"Resuming State {state}, with, step: {step}, start_epoch: {start_epoch}"
        )
        self.train(start_epoch, end_epoch, step, bst_vld_loss)

    def train(self, start_epoch, end_epoch, step: int = 0, bst_vld_loss: float = None):
        logger.debug(f"Training Begin")
        self.callbacks.on_begin()
        for ongoing_epoch in range(start_epoch, end_epoch):
            epoch_logs = dict()
            random.seed()

            lr = self.optimizer.param_groups[0]["lr"]

            progress_bar = tqdm.tqdm(
                total=(
                    len(self.loader.train_loader) * self.loader.train_loader.batch_size
                )
            )
            progress_bar.set_description("Epoch {}, lr {}".format(ongoing_epoch, lr))

            try:
                logger.debug("Setting Learning rate : {}".format(lr))
                epoch_logs = handle_dictionary(epoch_logs, "lr", lr)

                if not self.model.training:
                    self.model.train()

                train_loss, train_metric, step, progress_bar = self.state_train(
                    step, progress_bar
                )
                progress_bar.close()

                self.model.eval()
                valid_loss, valid_metric = self.state_validate()

                epoch_logs = handle_dictionary(epoch_logs, "train_loss", train_loss)
                epoch_logs = handle_dictionary(epoch_logs, "valid_loss", valid_loss)

                epoch_logs = handle_dictionary(epoch_logs, "train_metric", train_metric)
                epoch_logs = handle_dictionary(epoch_logs, "valid_metric", valid_metric)

                if (bst_vld_loss is None) or (valid_loss < bst_vld_loss):
                    bst_vld_loss = valid_loss

                epoch_logs = handle_dictionary(epoch_logs, "model", self.model)
                epoch_logs = handle_dictionary(
                    epoch_logs, "test_loader", self.loader.test_loader
                )

                self.callbacks.on_epoch_end(
                    ongoing_epoch,
                    logs={
                        **epoch_logs,
                        **self.collect_state(
                            ongoing_epoch, end_epoch, step, bst_vld_loss, "complete"
                        ),
                    },
                )

                logger.debug(
                    "Train Loss {}, Valid Loss {}".format(train_loss, valid_loss)
                )
                logger.debug("Train Metric {}".format(train_metric))
                logger.debug("Valid Metric {}".format(valid_metric))

                one_liner.one_line(
                    tag="Loss",
                    tag_data=f"train: {train_loss}, validation: {valid_loss}",
                    tag_color="cyan",
                    to_reset_data=True,
                )

                one_liner.one_line(
                    tag="Train Metric",
                    tag_data=dict_to_string(train_metric),
                    tag_color="cyan",
                    to_reset_data=True,
                    to_new_line_data=True,
                )
                one_liner.one_line(
                    tag="Valid Metric",
                    tag_data=dict_to_string(valid_metric),
                    tag_color="cyan",
                    to_new_line_data=True,
                    to_reset_data=True,
                )

            except KeyboardInterrupt:
                progress_bar.close()
                self.callbacks.interruption(
                    logs={
                        **epoch_logs,
                        **self.collect_state(
                            ongoing_epoch, end_epoch, step, bst_vld_loss, "interruption"
                        ),
                    }
                )

                one_liner.one_line(
                    tag="KeyBoard Interrupt",
                    tag_data=f"State Saved at epoch {ongoing_epoch}",
                    tag_color="cyan",
                    to_reset_data=True,
                )
                raise KeyboardInterrupt
            except Exception as ex:
                progress_bar.close()
                one_liner.one_line(
                    tag="Exception",
                    tag_data=str(ex),
                    tag_color="cyan",
                    to_reset_data=True,
                    to_new_line_data=True,
                )
                raise ex

        one_liner.one_line(
            tag="Training Complete",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )
        self.callbacks.on_end()

    def state_train(
        self, step: int, progress_bar
    ) -> Tuple[Union[int, float], dict, int, tqdm.std.tqdm]:

        report_each = 100
        batch_loss = []
        mean_loss = 0

        for train_data in self.loader.train_loader:
            batch_logs = dict()
            self.callbacks.on_batch_begin(step, logs=batch_logs)

            train_data = gpu_variable(train_data)

            prediction = self.model(train_data["images"])
            calculated_loss = self.criterion(train_data["ground_truth"], prediction)
            self.optimizer.zero_grad()
            calculated_loss.backward()
            self.optimizer.step()

            batch_loss.append(calculated_loss.item())
            mean_loss = np.mean(batch_loss[-report_each:])
            batch_logs = handle_dictionary(
                batch_logs, "plt_lr", {"data": mean_loss, "tag": "Loss/Step"}
            )
            batch_logs = handle_dictionary(batch_logs, "model", self.model)
            batch_logs = handle_dictionary(
                batch_logs, "test_loader", self.loader.test_loader
            )
            self.callbacks.on_batch_end(step, logs=batch_logs)
            progress_bar.update(self.loader.train_loader.batch_size)
            progress_bar.set_postfix(loss="{:.5f}".format(mean_loss))
            step += 1
            self.metrics.get_metrics(
                ground_truth=train_data["ground_truth"], prediction=prediction
            )

        return mean_loss.item(), self.metrics.compute_mean(), step, progress_bar

    @torch.no_grad()
    def state_validate(self) -> Tuple[np.ndarray, dict]:
        logger.debug("Validation In Progress")
        losses = []
        start = time.time()
        for ongoing_count, val_data in enumerate(self.loader.val_loader):
            ongoing_count += 1

            one_liner.one_line(
                tag="Validation",
                tag_data=f"{ongoing_count}/{len(self.loader.val_loader)} "
                f"ETA -- {compute_eta(start, ongoing_count, len(self.loader.val_loader))}",
                tag_color="cyan",
                to_reset_data=True,
            )
            val_data = gpu_variable(val_data)

            prediction = self.model(val_data["images"])
            loss = self.criterion(val_data["ground_truth"], prediction)

            losses.append(loss.item())
            self.metrics.get_metrics(
                ground_truth=val_data["ground_truth"], prediction=prediction
            )

        return np.mean(losses), self.metrics.compute_mean()

    def collect_state(
        self,
        ongoing_epoch: int,
        end_epoch: int,
        step: int,
        bst_vld_loss: float,
        run_state: str,
    ) -> dict:
        assert run_state in ["interruption", "complete"], (
            "Expected state to save ['interruption', 'complete']" "got %s",
            (run_state,),
        )

        state_data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
            if self.optimizer is not None
            else "NA",
            "start_epoch": ongoing_epoch + 1
            if run_state == "complete"
            else ongoing_epoch,
            "step": step,
            "bst_vld_loss": bst_vld_loss if bst_vld_loss is not None else "NA",
            "end_epoch": end_epoch,
        }
        return {"state": state_data}
