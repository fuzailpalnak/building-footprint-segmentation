import logging
import sys

import torch.optim.lr_scheduler as lr_scheduler


logger = logging.getLogger()


def get_scheduler(scheduler: str, **kwargs):
    if hasattr(lr_scheduler, scheduler):
        return getattr(lr_scheduler, scheduler)(**kwargs)
    else:
        return str_to_class(scheduler)(**kwargs)


def str_to_class(class_name: str):
    class_obj = getattr(sys.modules[__name__], class_name)
    return class_obj
