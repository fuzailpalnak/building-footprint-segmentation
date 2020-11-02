import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SpatialPooling(nn.Module):
    def __init__(self):
        super(SpatialPooling, self).__init__()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(5, 5))
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(6, 6))

    def forward(self, input_feature: Tensor) -> Tensor:
        _, _, w, h = input_feature.shape
        p1 = F.interpolate(self.max_pool_1(input_feature), size=(w, h), mode="bilinear")
        p2 = F.interpolate(self.max_pool_2(input_feature), size=(w, h), mode="bilinear")
        p3 = F.interpolate(self.max_pool_3(input_feature), size=(w, h), mode="bilinear")
        p4 = F.interpolate(self.max_pool_4(input_feature), size=(w, h), mode="bilinear")

        out = torch.cat([p1, p2, p3, p4, input_feature], 1)
        return out
