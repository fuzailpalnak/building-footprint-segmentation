import numpy as np
from typing import Union

from torch import Tensor

from building_footprint_segmentation.helpers.metrics import confusion_matrix

EPSILON = 1e-11


def accuracy(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    tn, fp, fn, tp = confusion_matrix(prediction, ground_truth)
    num = tp + tn
    den = tp + tn + fp + fn
    return num / (den + EPSILON)


def f1(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """

    _, fp, fn, tp = confusion_matrix(prediction, ground_truth)
    return (2 * tp) / (((2 * tp) + fp + fn) + EPSILON)


def recall(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    _, _, fn, tp = confusion_matrix(prediction, ground_truth)
    return tp / (tp + fn + EPSILON)


def precision(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    _, fp, _, tp = confusion_matrix(prediction, ground_truth)
    return tp / (tp + fp + EPSILON)


def iou(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    _, fp, fn, tp = confusion_matrix(prediction, ground_truth)
    denominator = tp + fp + fn
    if denominator == 0:
        value = 0
    else:
        value = float(tp) / (denominator + EPSILON)
    return value


def get_metrics():
    return [accuracy, precision, recall, f1, iou]
