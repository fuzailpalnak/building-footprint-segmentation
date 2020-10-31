import os
import sys
import time
import traceback
import numpy as np
import cv2
from torch import Tensor

from building_segmentation.utils.date_time import get_time
from building_segmentation.utils.py_network import convert_tensor_to_numpy


def handle_dictionary(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

    return dictionary


def dict_to_string(input_dict, separator=", "):
    combined_list = list()
    for key, value in input_dict.items():
        individual = "{} : {:.5f}".format(key, value)
        combined_list.append(individual)
    return separator.join(combined_list)


def make_directory(current_dir, folder_name):
    new_dir = os.path.join(current_dir, folder_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def create_state_path(pth, version):
    version_path = make_directory(pth, version)
    state_path = make_directory(version_path, "state")
    default_weights_path = os.path.join(state_path, "default_state.pt")
    best_weights_path = os.path.join(state_path, "best_state.pt")
    return default_weights_path, best_weights_path


def create_chk_path(pth, exp_name, model, version):
    version_path = make_directory(pth, version)
    chk_path = make_directory(version_path, "chk_pt")
    weights_path = os.path.join(
        chk_path, "{}_{}_{}_chk.pt".format(exp_name, model, version)
    )
    return weights_path


def level_2_folder_creation(root, level_1, level_2):
    root_folder = make_directory(os.getcwd(), "exp_zoo/" + root)
    level_1_folder_path = make_directory(root_folder, level_1)
    level_2_folder_path = make_directory(level_1_folder_path, level_2)

    return root_folder, level_1_folder_path, level_2_folder_path


def create_version(directory):
    subdir = os.listdir(directory)
    if len(subdir) == 0:
        version_number = 1
    else:
        existing_version = list()
        for sub in subdir:
            version_number = sub[1:]
            existing_version.append(int(version_number))
        existing_version.sort()
        version_number = existing_version[-1] + 1
    current_version = "v" + str(version_number)

    return current_version


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


def crop_image(img: np.ndarray, model_input_dimension: tuple, random_crop_coord: tuple):
    model_height, model_width = model_input_dimension
    height, width = random_crop_coord

    img = img[height : height + model_height, width : width + model_width]

    return img


def get_random_crop_x_and_y(model_input_dimension: tuple, image_input_dimension: tuple):
    model_height, model_width = model_input_dimension
    image_height, image_width, _ = image_input_dimension
    h_start = np.random.randint(0, image_height - model_height)
    w_start = np.random.randint(0, image_width - model_height)

    return h_start, w_start


def get_pad_limit(model_input_dimension: tuple, image_input_dimension: tuple):
    model_height, model_width = model_input_dimension
    image_height, image_width, _ = image_input_dimension

    limit = (model_height - image_height) // 2
    return limit


def pad_image(img: np.ndarray, limit: int):
    img = cv2.copyMakeBorder(
        img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101
    )
    return img


def perform_scale(img, dimension, interpolation=cv2.INTER_NEAREST):
    new_height, new_width = dimension
    img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return img


def handle_image_size(img, mask, dimension):
    if dimension < (img.shape[0], img.shape[1]):
        height, width = get_random_crop_x_and_y(dimension, img.shape)
        img = crop_image(img, dimension, (height, width))
        if mask is not None:
            mask = crop_image(mask, dimension, (height, width))
        return img, mask

    elif dimension > (img.shape[0], img.shape[1]):
        limit = get_pad_limit(dimension, img.shape)
        img = pad_image(img, limit)
        if mask is not None:
            mask = pad_image(mask, limit)
        return img, mask
    else:
        return img, mask


def load_image(path: str):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_binary(prediction, cutoff=0.40):
    prediction[prediction >= cutoff] = 1
    prediction[prediction < cutoff] = 0
    return prediction


def get_numpy(ip):
    if type(ip) == Tensor:
        return convert_tensor_to_numpy(ip)
    elif type(ip) == np.ndarray:
        return ip


def compute_eta(start, current_iter, total_iter):
    e = time.time() - start
    eta = e * total_iter / current_iter - e
    return get_time(eta)
