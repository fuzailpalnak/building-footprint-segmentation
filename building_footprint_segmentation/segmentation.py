from types import SimpleNamespace
from typing import List

import torch
import yaml

from building_footprint_segmentation.utils.py_network import load_parallel_model


class Segmentation:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def load_model(self, name: str, **param):
        return load_parallel_model(self.segmentation.create_network(name, **param))

    def load_criterion(self, name: str, **param):
        return self.segmentation.create_criterion(name, **param)

    def load_loader(
        self,
        root_folder: str,
        image_normalization: str,
        label_normalization: str,
        augmenters: dict,
        batch_size: int,
    ):
        return self.segmentation.create_loader(
            root_folder,
            image_normalization,
            label_normalization,
            augmenters,
            batch_size,
        )

    def load_metrics(self, data_metrics: List[str]):
        return self.segmentation.create_metrics(data_metrics)

    @staticmethod
    def load_optimizer(model, name: str, **param):
        return getattr(torch.optim, name)(
            filter(lambda p: p.requires_grad, model.parameters()), **param
        )


def init_segmentation(segmentation_type: str):
    if segmentation_type == "binary":
        from building_footprint_segmentation.ml.binary.factory import BinaryFactory

        return Segmentation(BinaryFactory())
    else:
        raise NotImplementedError


def parse_segmentation_trainer_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    assert {"Model", "Criterion", "Loader", "Metrics", "Optimizer", "log_dir"} == set(
        list(config.keys())
    ), (
        "Expected config to have ['Model', 'Criterion', 'Loader', 'Metrics', 'Optimizer', 'log_dir']"
        "got %s",
        (config.keys(),),
    )
    config = SimpleNamespace(**config)
    return config
