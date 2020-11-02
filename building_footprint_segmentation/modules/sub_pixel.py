from torch import nn, Tensor


class SubPixel(nn.Module):
    def __init__(self, down_factor: int, in_features: int, num_classes: int):
        super(SubPixel, self).__init__()
        features = (down_factor ** 2) * num_classes
        self.convolution = nn.Conv2d(in_features, features, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(features)
        self.non_linearity = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, input_feature: Tensor) -> Tensor:
        input_feature = self.convolution(input_feature)
        input_feature = self.bn(input_feature)
        input_feature = self.non_linearity(input_feature)
        input_feature = self.pixel_shuffle(input_feature)
        return input_feature
