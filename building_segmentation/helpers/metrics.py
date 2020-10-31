import logging
from typing import Union

import numpy as np
from torch import Tensor

from building_segmentation.utils.operations import handle_dictionary

EPSILON = 1e-11

logger = logging.getLogger("segmentation")


class MetricList:
    def __init__(self, metrics: list):
        metrics = metrics or []
        self.metrics = [c for c in metrics]
        if len(metrics) != 0:
            [
                logger.debug("Registered {}".format(c.__class__.__name__))
                for c in metrics
            ]
        self.metric_value = dict()

    def append(self, callback):
        logger.debug("Registered {}".format(callback.__class__.__name__))
        self.metrics.append(callback)

    def get_metrics(
        self,
        ground_truth: Union[np.ndarray, Tensor],
        prediction: Union[np.ndarray, Tensor],
    ):
        """

        :param ground_truth:
        :param prediction:
        :return:
        """
        computed_metric = self.compute_metric(ground_truth, prediction)
        for key, value in computed_metric.items():
            self.metric_value = handle_dictionary(self.metric_value, key, value)

    def compute_metric(
        self,
        ground_truth: Union[np.ndarray, Tensor],
        prediction: Union[np.ndarray, Tensor],
    ) -> dict:
        """

        :param ground_truth:
        :param prediction:
        :return:
        """
        logger.debug("Computing Metrics")
        computed_metric = dict()
        for metric in self.metrics:
            logger.debug(f"Metric Computation {metric.__name__}")
            value = metric(ground_truth, prediction)
            computed_metric[metric.__name__] = value
        return computed_metric

    def compute_mean(self) -> dict:
        logger.debug("Computing Metrics Mean")
        mean_metric = dict()
        for key, value in self.metric_value.items():
            assert type(value) in [float, list, np.float64], (
                "Expected to have either ['float', 'list', 'np.float64']" "got %s",
                (type(value),),
            )
            mean_value = np.mean(value)
            mean_metric = handle_dictionary(mean_metric, key, mean_value)
        self.metric_value = dict()
        return mean_metric
