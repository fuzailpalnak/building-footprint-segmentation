from typing import List

import torch

from building_segmentation.utils.py_network import load_parallel_model


class Extractor:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def load_model(self, model_name: str, **model_param):
        return load_parallel_model(
            self.segmentation.create_network(model_name, **model_param)
        )

    def load_criterion(self, criterion_name: str, **criterion_param):
        return self.segmentation.create_criterion(criterion_name, **criterion_param)

    def load_loader(
        self,
        root_folder: str,
        image_normalization: str,
        label_normalization: str,
        batch_size: int,
    ):
        return self.segmentation.create_loader(
            root_folder, image_normalization, label_normalization, batch_size
        )

    def load_metrics(self, data_metrics: List[str]):
        return self.segmentation.create_metrics(data_metrics)

    @staticmethod
    def load_optimizer(model, optimizer_name: str, **optimizer_param):
        return getattr(torch.optim, optimizer_name)(
            filter(lambda p: p.requires_grad, model.parameters()), **optimizer_param
        )


def init_extractor(segmentation_type: str):
    if segmentation_type == "binary":
        from building_segmentation.ml.binary.factory import BinaryFactory

        return Extractor(BinaryFactory())
    else:
        raise NotImplementedError
