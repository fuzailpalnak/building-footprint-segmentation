import numpy as np
import cv2


def divide_by_255(img: np.ndarray) -> np.ndarray:
    return img / 255


def binary_label(mask) -> np.ndarray:
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    normalized_mask = divide_by_255(mask)
    return np.expand_dims(normalized_mask, -1)
