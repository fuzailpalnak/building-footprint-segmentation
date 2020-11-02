from typing import List


class Factory:
    def __init__(self):
        pass

    def create_loader(
        self,
        root_folder,
        image_normalization,
        label_normalization,
        augmenters,
        batch_size,
    ):
        raise NotImplementedError

    def create_network(self, name, **kwargs):
        raise NotImplementedError

    def create_criterion(self, name, **kwargs):
        raise NotImplementedError

    def create_metrics(self, data_metrics: List[str]):
        raise NotImplementedError
