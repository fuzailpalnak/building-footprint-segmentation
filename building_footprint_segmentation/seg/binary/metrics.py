import numpy as np
from typing import Union

from sklearn.metrics import confusion_matrix
from torch import Tensor

from building_footprint_segmentation.utils.operations import get_numpy, to_binary


EPSILON = 1e-11


def accuracy(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    prediction = get_numpy(prediction).flatten()
    ground_truth = get_numpy(ground_truth).flatten()
    prediction = to_binary(prediction)
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction, labels=[0, 1]).ravel()
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
    prediction = get_numpy(prediction).flatten()
    ground_truth = get_numpy(ground_truth).flatten()
    prediction = to_binary(prediction)
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction, labels=[0, 1]).ravel()
    num = 2 * tp
    den = (2 * tp) + fp + fn
    return num / (den + EPSILON)


def recall(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    prediction = get_numpy(prediction).flatten()
    ground_truth = get_numpy(ground_truth).flatten()
    prediction = to_binary(prediction)
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction, labels=[0, 1]).ravel()
    return tp / (tp + fn + EPSILON)


def precision(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    prediction = get_numpy(prediction).flatten()
    ground_truth = get_numpy(ground_truth).flatten()
    prediction = to_binary(prediction)
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction, labels=[0, 1]).ravel()
    return tp / (tp + fp + EPSILON)


def iou(
    ground_truth: Union[np.ndarray, Tensor], prediction: Union[np.ndarray, Tensor]
) -> float:
    """

    :param ground_truth:
    :param prediction:
    :return:
    """
    prediction = get_numpy(prediction).flatten()
    ground_truth = get_numpy(ground_truth).flatten()
    prediction = to_binary(prediction)
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction, labels=[0, 1]).ravel()
    denominator = tp + fp + fn
    if denominator == 0:
        value = 0
    else:
        value = float(tp) / (denominator + EPSILON)
    return value


def get_metrics():
    return [accuracy, precision, recall, f1, iou]
