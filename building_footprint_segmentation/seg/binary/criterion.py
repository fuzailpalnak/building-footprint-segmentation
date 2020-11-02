import torch
from torch import nn

from building_footprint_segmentation.seg.base_criterion import BaseCriterion


class BinaryCrossEntropy(BaseCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()

    def compute_criterion(self, ground_truth, prediction):

        loss = self.nll_loss(prediction, ground_truth)
        return loss


class IOU(BaseCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.iou_weight = kwargs["iou_weight"]
        self.binary_weight = 1 if self.iou_weight == 1 else 1 - self.iou_weight

    def compute_criterion(self, ground_truth, prediction):

        loss = self.binary_weight * self.nll_loss(prediction, ground_truth)

        if self.iou_weight:
            loss -= self.iou_weight * self.calculate_iou(
                prediction.sigmoid(), (ground_truth == 1).float()
            )
        return loss

    @staticmethod
    def calculate_iou(prediction, ground_truth):
        eps = 1e-15

        intersection = (prediction * ground_truth).sum()
        union = prediction.sum() + ground_truth.sum()

        iou = torch.log((intersection + eps) / (union - intersection + eps))
        return iou


class Dice(BaseCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weights = kwargs["dice_weight"]

    def compute_criterion(self, ground_truth, prediction):

        bce_loss = self.nll_loss(prediction, ground_truth)

        dice_loss = self.calculate_dice(prediction.sigmoid(), ground_truth)
        loss = bce_loss + self.dice_weights * (1 - dice_loss)
        return loss

    @staticmethod
    def calculate_dice(prediction, ground_truth):
        dice = (2.0 * (prediction * ground_truth).sum() + 1) / (
            prediction.sum() + ground_truth.sum() + 1
        )
        return dice
