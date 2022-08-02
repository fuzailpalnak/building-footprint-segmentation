from collections import OrderedDict
from typing import Union, List, Any, Tuple, Dict

import torch

import numpy as np
from torch import Tensor


def get_gpu_device_ids():
    """
    Get list of available GPU devices
    :return:
    """

    device_id = list()
    separator = ","
    gpu_device_available = torch.cuda.device_count()
    for i in range(gpu_device_available):
        device_id.append(str(i))
    device_id = separator.join(device_id)
    return device_id


def load_parallel_model(model) -> Union[Any, torch.nn.DataParallel]:
    """

    :param model:
    :return:
    """
    if torch.cuda.is_available():
        device_ids = get_gpu_device_ids()
        if device_ids:
            device_ids = list(map(int, device_ids.split(",")))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model
    else:
        return model


def adjust_model(state: dict) -> OrderedDict:
    """
    # WhenEver a model is trained on multi gpu using DataParallel, module keyword is added

    :param state:
    :return:
    """
    model = {
        ([".".join(key.split(".")[1:])][0] if "module" in key.split(".")[0] else key): (
            value
        )
        for key, value in state.items()
    }
    return OrderedDict(model.items())


def gpu_variable(
    input_variable: Union[List, Tuple, Dict, Tensor]
) -> Union[List[Tensor], Tuple[Tensor], Dict, Tensor]:
    """
    :param input_variable:
    :return:
    """
    if isinstance(input_variable, (list, tuple)):
        return [gpu_variable(y) for y in input_variable]

    if isinstance(input_variable, dict):
        for k, v in input_variable.items():
            input_variable[k] = gpu_variable(v)
        return input_variable

    return input_variable.cuda() if torch.cuda.is_available() else input_variable


def to_input_image_tensor(
    img: Union[List[np.ndarray], np.ndarray]
) -> Union[List[Tensor], Tensor]:
    """

    :param img:
    :return:
    """
    if isinstance(img, list):
        return [to_input_image_tensor(image) for image in img]
    return to_tensor(np.moveaxis(img, -1, 0))


def to_label_image_tensor(mask: np.ndarray):
    return to_tensor(np.expand_dims(mask, 0))


def to_tensor(data: np.ndarray) -> Tensor:
    return torch.from_numpy(data).float()


def to_multi_output_label_image_tensor(mask: np.ndarray) -> Tensor:
    return to_tensor(np.moveaxis(mask, -1, 0))


def add_extra_dimension(
    data: Union[List[Tensor], Tensor]
) -> Union[List[Tensor], Tensor]:
    if isinstance(data, (list, tuple)):
        return [torch.unsqueeze(y.cuda(), dim=0) for y in data]
    return torch.unsqueeze(data, dim=0)


def convert_tensor_to_numpy(input_variable: Tensor) -> np.ndarray:
    return (
        input_variable.data.cpu().numpy()
        if input_variable.is_cuda
        else input_variable.data.numpy()
    )


def extract_state(weight_path: str) -> dict:
    state = torch.load(str(weight_path), map_location="cpu")
    return state
