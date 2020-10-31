from typing import List

from building_segmentation.helpers.metrics import MetricList
from building_segmentation.ml.base_factory import Factory
from building_segmentation.ml.binary.loader import BinaryLoader
from building_segmentation.ml.binary import metrics
from building_segmentation.ml.binary import criterion
from building_segmentation.ml.binary import models


class BinaryFactory(Factory):
    def create_loader(
        self,
        root_folder: str,
        image_normalization: str,
        ground_truth_normalization: str,
        batch_size: int,
    ):
        return BinaryLoader.get_data_loader(
            root_folder, image_normalization, ground_truth_normalization, batch_size
        )

    def create_network(self, model_name: str, **model_param):
        return getattr(models, model_name)(**model_param)

    def create_criterion(self, criterion_name: str, **criterion_param):
        return getattr(criterion, criterion_name)(**criterion_param)

    def create_metrics(self, data_metrics: List[str]) -> MetricList:
        _metrics = list()
        for metric in data_metrics:
            if hasattr(metrics, metric):
                _metrics.append(getattr(metrics, metric))
        return MetricList(_metrics)
