import os
import sys
import time
import traceback
from typing import Union, Tuple, Any

import numpy as np
import cv2
from torch import Tensor

from building_footprint_segmentation.utils.date_time import get_time
from building_footprint_segmentation.utils.py_network import convert_tensor_to_numpy


def handle_dictionary(input_dictionary: dict, key: Any, value: Any) -> dict:
    """

    :param input_dictionary:
    :param key:
    :param value:
    :return:
    """
    if key not in input_dictionary:
        input_dictionary[key] = value
    elif type(input_dictionary[key]) == list:
        input_dictionary[key].append(value)
    else:
        input_dictionary[key] = [input_dictionary[key], value]

    return input_dictionary


def dict_to_string(input_dict: dict, separator=", ") -> str:
    """

    :param input_dict:
    :param separator:
    :return:
    """
    combined_list = list()
    for key, value in input_dict.items():
        individual = "{} : {:.5f}".format(key, value)
        combined_list.append(individual)
    return separator.join(combined_list)


def make_directory(current_dir: str, folder_name: str) -> str:
    """

    :param current_dir:
    :param folder_name:
    :return:
    """
    new_dir = os.path.join(current_dir, folder_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def is_overridden_func(func):
    # https://stackoverflow.com/questions/9436681/how-to-detect-method-overloading-in-subclasses-in-python
    obj = func.__self__
    base_class = getattr(super(type(obj), obj), func.__name__)
    return func.__func__ != base_class.__func__


def extract_detail():
    """Extracts failing function name from Traceback
    by Alex Martelli
    http://stackoverflow.com/questions/2380073/how-to-identify-what-function-call-raise-an-exception-in-python
    """
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, -1)[0]
    return "{} in {} line num {} on line {} ".format(
        stk.name, stk.filename, stk.lineno, stk.line
    )


def get_details(fn):
    class_name = vars(sys.modules[fn.__module__])[
        fn.__qualname__.split(".")[0]
    ].__name__
    fn_name = fn.__name__
    if class_name == fn_name:
        return None, fn_name
    else:
        return class_name, fn_name


def crop_image(
    input_image: np.ndarray, crop_to_dimension: tuple, random_coord: tuple
) -> np.ndarray:
    """

    :param input_image:
    :param crop_to_dimension:
    :param random_coord:
    :return:
    """
    model_height, model_width = crop_to_dimension
    height, width = random_coord

    input_image = input_image[
        height : height + model_height, width : width + model_width
    ]

    return input_image


def get_random_crop_x_and_y(
    crop_to_dimension: tuple, base_dimension: tuple
) -> Tuple[int, int]:
    """

    :param crop_to_dimension:
    :param base_dimension:
    :return:
    """
    crop_height, crop_width = crop_to_dimension
    base_height, base_width = base_dimension
    h_start = np.random.randint(0, base_height - crop_height)
    w_start = np.random.randint(0, base_width - crop_height)

    return h_start, w_start


def get_pad_limit(model_input_dimension: tuple, image_input_dimension: tuple) -> int:
    """

    :param model_input_dimension:
    :param image_input_dimension:
    :return:
    """
    model_height, model_width = model_input_dimension
    image_height, image_width = image_input_dimension

    limit = (model_height - image_height) // 2
    return limit


def pad_image(img: np.ndarray, limit: int) -> np.ndarray:
    """

    :param img:
    :param limit:
    :return:
    """
    img = cv2.copyMakeBorder(
        img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101
    )
    return img


def perform_scale(
    img: np.ndarray, dimension: tuple, interpolation=cv2.INTER_NEAREST
) -> np.ndarray:
    """

    :param img:
    :param dimension:
    :param interpolation:
    :return:
    """
    new_height, new_width = dimension
    img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return img


def load_image(path: str):
    """

    :param path:
    :return:
    """
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_binary(
    prediction: Union[np.ndarray, Tensor], cutoff=0.40
) -> Union[np.ndarray, Tensor]:
    """

    :param prediction:
    :param cutoff:
    :return:
    """
    prediction[prediction >= cutoff] = 1
    prediction[prediction < cutoff] = 0
    return prediction


def get_numpy(data: Union[Tensor, np.ndarray]) -> np.ndarray:
    """

    :param data:
    :return:
    """
    return convert_tensor_to_numpy(data) if type(data) == Tensor else data


def compute_eta(start, current_iter, total_iter):
    """

    :param start:
    :param current_iter:
    :param total_iter:
    :return:
    """
    e = time.time() - start
    eta = e * total_iter / current_iter - e
    return get_time(eta)


def handle_image_size(input_image: np.ndarray, dimension: tuple):
    """

    :param input_image:
    :param dimension:
    :return:
    """
    assert input_image.ndim == 3, (
        "Image should have 3 dimension '[HxWxC]'" "got %s",
        (input_image.shape,),
    )
    assert len(dimension) == 2, (
        "'dimension' should have 'Hxw' " "got %s",
        (dimension,),
    )

    h, w, _ = input_image.shape

    if dimension < (h, w):
        random_height, random_width = get_random_crop_x_and_y(dimension, (h, w))
        input_image = crop_image(input_image, dimension, (random_height, random_width))

    elif dimension > (h, w):
        limit = get_pad_limit(dimension, (h, w))
        input_image = pad_image(input_image, limit)

    return input_image
