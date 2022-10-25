from typing import List

from albumentations import Compose

from building_footprint_segmentation.helpers.metrics import BinaryMetric
from building_footprint_segmentation.seg.base_factory import Factory
from building_footprint_segmentation.seg.binary.loader import BinaryLoader
from building_footprint_segmentation.seg.binary import metrics
from building_footprint_segmentation.seg.binary import criterion
from building_footprint_segmentation.seg.binary import models


class BinaryFactory(Factory):
    def create_loader(
        self,
        root_folder: str,
        image_normalization: str,
        ground_truth_normalization: str,
        augmenters: Compose,
        batch_size: int,
    ):
        return BinaryLoader.get_data_loader(
            root_folder,
            image_normalization,
            ground_truth_normalization,
            augmenters,
            batch_size,
        )

    def create_network(self, name: str, **kwargs):
        return getattr(models, name)(**kwargs)

    def create_criterion(self, name: str, **kwargs):
        return getattr(criterion, name)(**kwargs)

    def create_metrics(self, data_metrics: List[str]) -> BinaryMetric:
        _metrics = list()
        for metric in data_metrics:
            if hasattr(metrics, metric):
                _metrics.append(getattr(metrics, metric))
        return BinaryMetric(_metrics)
