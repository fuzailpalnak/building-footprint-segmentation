from typing import Union, List

import torch

import numpy as np
from torch import Tensor


def get_gpu_device_ids():
    device_id = list()
    separator = ","
    gpu_device_available = torch.cuda.device_count()
    for i in range(gpu_device_available):
        device_id.append(str(i))
    device_id = separator.join(device_id)
    return device_id


def load_parallel_model(model):
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


def adjust_model(state):
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
    return model


def get_prediction_as_per_instance(outputs):
    """

    :param outputs:
    :return:
    """
    if isinstance(outputs, dict):
        assert "final_image" in outputs, "while passing image use key-final_image"
        return outputs["final_image"]
    elif isinstance(outputs, torch.Tensor):
        return outputs
    else:
        raise NotImplementedError


def gpu_variable(x):
    """
    :param x:
    :return:
    """
    if isinstance(x, (list, tuple)):
        return [gpu_variable(y) for y in x]

    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = gpu_variable(v)
        return x

    return x.cuda() if torch.cuda.is_available() else x


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
    return torch.unsqueeze(data.cuda(), dim=0)


def convert_tensor_to_numpy(ip):
    if ip.is_cuda:
        return ip.data.cpu().numpy()
    else:
        return ip.data.numpy()


def extract_state(weight_path: str):
    state = torch.load(str(weight_path), map_location="cpu")
    return state


def model_state(model, state_path: str):
    model_adjusted = adjust_model(extract_state(state_path))
    model.load_state_dict(model_adjusted)
    return model
