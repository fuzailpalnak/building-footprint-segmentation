from abc import abstractmethod


class BaseCriterion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, ground_truth, predictions):
        return self.compute_criterion(ground_truth, predictions)

    @abstractmethod
    def compute_criterion(self, ground_truth, predictions):
        raise NotImplementedError
